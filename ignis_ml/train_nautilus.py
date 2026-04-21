#/ignis_ml/train_nautilus.py
import os, math, random, gc, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

from src.utils.config import load_config
from src.models.convlstm_unet import ConvLSTMUNet, ARCH_VERSION
from src.data.dataset import NpzTileDataset
# ---- Fast-math toggles (helpful on NVIDIA L4) ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
U_IDX = 1
V_IDX = 2
# ------------------------- GPU Monitoring -------------------------
def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved | "
              f"Peak: {max_allocated:.2f}GB")
        torch.cuda.reset_peak_memory_stats()

def clear_gpu_cache():
    """Clear CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ------------------------- Path Fixing -------------------------
def fix_split_paths(split_file: Path, tiles_dir: Path):
    """Fix absolute paths in split files to match current system"""
    if not split_file.exists():
        return
    
    print(f"Checking paths in {split_file.name}...")
    
    # Read all lines
    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        return
    
    # Check first line to see if paths need fixing
    first_path = Path(lines[0])
    
    # If path doesn't exist or is from different system (contains /Users/ or different root)
    needs_fixing = not first_path.exists() or '/Users/' in str(first_path)
    
    if needs_fixing:
        print(f"  Fixing paths in {split_file.name}...")
        fixed_lines = []
        
        for line in lines:
            old_path = Path(line)
            # Extract just the filename
            filename = old_path.name
            # Create new path relative to tiles_dir
            new_path = tiles_dir / filename
            
            if new_path.exists():
                fixed_lines.append(str(new_path.resolve()))
            else:
                print(f"  Warning: Could not find {filename}")
        
        # Write back fixed paths
        with open(split_file, 'w') as f:
            for line in fixed_lines:
                f.write(line + '\n')
        
        print(f"  ✓ Fixed {len(fixed_lines)} paths in {split_file.name}")
    else:
        print(f"  ✓ Paths in {split_file.name} are already correct")

# ------------------------- Utils -------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_device():
    if torch.cuda.is_available(): 
        print(f"🚀 CUDA Device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
        return "mps"
    return "cpu"

def compute_pixel_prior(files, max_files=2000):
    files = files[:max_files]; pos=0; tot=0
    for p in files:
        y = np.load(p, mmap_mode="r")["y"]
        pos += (y>0.5).sum(); tot += y.size
    return float((pos/tot) if tot>0 else 0.01)

def make_pos_weight(ratio_pos: float, cap: float = 25.0, device: str = "cpu"):
    if ratio_pos <= 0 or ratio_pos >= 1: w = 1.0
    else: w = (1 - ratio_pos) / max(ratio_pos, 1e-12)
    return torch.tensor([min(w, cap)], device=device, dtype=torch.float32)

def init_final_bias_to_prior(model, prior_pos: float):
    prior = min(max(prior_pos, 1e-6), 1-1e-6)
    bias = math.log(prior/(1-prior))
    with torch.no_grad():
        if hasattr(model,"dec") and hasattr(model.dec,"out") and hasattr(model.dec.out,"bias"):
            model.dec.out.bias.fill_(bias)

def build_train_sampler(train_files, target_pos_fraction=None, num_bins=6, seed=42):
    """
    Fire-fraction–balanced sampler.
    Accepts the old 'target_pos_fraction' kwarg (ignored) so existing calls keep working.
    Returns (sampler, mean_fire_frac).
    """
    rng = np.random.default_rng(seed)

    fracs = []
    for p in train_files:
        y = np.load(p, mmap_mode="r")["y"]
        fracs.append(float((y > 0.5).mean()))
    fracs = np.asarray(fracs)

    # bin by quantiles so each difficulty bucket is equally likely
    qs = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(fracs, qs)
    edges[0]  = min(edges[0],  fracs.min()) - 1e-9
    edges[-1] = max(edges[-1], fracs.max()) + 1e-9
    bins = np.digitize(fracs, edges[1:-1], right=False)

    counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[bins]

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights_t, num_samples=len(weights_t), replacement=True)
    return sampler, float(fracs.mean())
# ------------------------- Loss & Metrics -------------------------
def _dilate_labels(y: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0: return y
    k = px*2 + 1
    return F.max_pool2d(y, kernel_size=k, stride=1, padding=px)

def loss_bce_dice_focal(logits, y, *, pos_weight, dice_lambda=0.5, focal=None, dice_dilate_px=1):
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    y_d = _dilate_labels(y, int(dice_dilate_px))
    p = torch.sigmoid(logits)
    inter = (p*y_d).sum(dim=(1,2,3))
    dice = 1 - (2*inter + 1e-6)/(p.sum(dim=(1,2,3)) + y_d.sum(dim=(1,2,3)) + 1e-6)
    loss = bce + dice_lambda * dice.mean()
    if focal and focal.get("enable", False):
        gamma = float(focal.get("gamma", 2.0))
        alpha = float(focal.get("alpha", 0.85))
        pt = torch.where(y>0.5, p, 1-p).clamp(1e-6, 1-1e-6)
        focal_term = ((1-pt)**gamma) * F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        focal_w = torch.where(y>0.5, alpha, 1-alpha)
        loss = loss + 0.25*(focal_w*focal_term).mean()
    return loss

def _resize_to_target(logits, y):
    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return logits

@torch.no_grad()
def _rot90_vec_torch(xd: torch.Tensor, k: int, u_idx: int, v_idx: int) -> torch.Tensor:
    """
    xd: [B,T,C,H,W]; rotate spatially by k*90° CCW and rotate wind vector (u,v).
    """
    xd = torch.rot90(xd, k, dims=(-2, -1))
    if (u_idx is not None) and (v_idx is not None):
        u = xd[:, :, u_idx].clone()
        v = xd[:, :, v_idx].clone()
        if   k == 1: u2, v2 = -v,  u
        elif k == 2: u2, v2 = -u, -v
        elif k == 3: u2, v2 =  v, -u
        else:        u2, v2 =  u,  v
        xd[:, :, u_idx] = u2
        xd[:, :, v_idx] = v2
    return xd

@torch.no_grad()
def _predict_tta(model, x_d, x_s):
    """
    TTA over 0,90,180,270 deg. x_d:[B,T,C,H,W], x_s:[B,Cs,H,W]
    """
    logits_sum = 0
    for k in (0, 1, 2, 3):
        xd_k = _rot90_vec_torch(x_d.clone(), k, U_IDX, V_IDX)
        xs_k = torch.rot90(x_s, k, dims=(-2, -1))
        out  = model(xd_k, xs_k)
        logits_sum = logits_sum + torch.rot90(out, -k, dims=(-2, -1))
    return logits_sum / 4.0

@torch.no_grad()
def eval_metrics(model, loader, device, thresh=0.5, pos_weight=None, dice_lambda=0.5, use_tta=False, use_amp=False):
    model.eval(); n=0; tot_loss=tot_iou=tot_dice=tot_csi=0.0
    for x_d, x_s, y in loader:
        x_d = x_d.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y   = y.to(device, non_blocking=True)
        
        with autocast("cuda", enabled=use_amp):
            logits = _predict_tta(model, x_d, x_s) if use_tta else model(x_d,x_s)
            logits = _resize_to_target(logits, y)
            loss = loss_bce_dice_focal(logits, y, pos_weight=pos_weight, dice_lambda=dice_lambda, 
                                      focal=None, dice_dilate_px=0)
        
        pbin = (torch.sigmoid(logits) >= thresh).float()
        inter = (pbin*y).sum(); union = (pbin + y - pbin*y).sum()
        iou = (inter/(union+1e-6)).item()
        dice= (2*inter/(pbin.sum()+y.sum()+1e-6)).item()
        tp = inter; fp = (pbin*(1-y)).sum(); fn = ((1-pbin)*y).sum()
        csi = (tp/(tp+fp+fn+1e-6)).item()
        bs = y.size(0)
        tot_loss += loss.item()*bs; tot_iou += iou*bs; tot_dice += dice*bs; tot_csi += csi*bs; n += bs
    return tot_loss/max(n,1), tot_iou/max(n,1), tot_dice/max(n,1), tot_csi/max(n,1)

@torch.no_grad()
def find_best_threshold(model, loader, device, candidates=None, pos_weight=None, dice_lambda=0.5, 
                       use_tta=False, use_amp=False):
    if candidates is None:
        # v2 uses relaxed focal (gamma=2, alpha=0.75), so the model no longer
        # saturates near 1.0. Search where CSI actually peaks for a balanced loss.
        candidates = np.linspace(0.3, 0.7, 9)
    best = (-1.0, 0.5)
    for th in candidates:
        _,_,_,csi = eval_metrics(model, loader, device, thresh=float(th), pos_weight=pos_weight, 
                                dice_lambda=dice_lambda, use_tta=use_tta, use_amp=use_amp)
        if csi > best[0]: best = (csi, float(th))
    return best

# ------------------------- Main -------------------------
def main():
    global U_IDX, V_IDX
    CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
    CFG = load_config(CONFIG_PATH)
    try:
        U_IDX = CFG.channels.dynamic_order.index("u")
        V_IDX = CFG.channels.dynamic_order.index("v")
    except Exception:
        print("[warn] falling back to u=1, v=2; verify dynamic_order!")
        U_IDX, V_IDX = 1, 2
    set_seed(CFG.training.seed)
    device = get_device()
    use_amp = (device == "cuda") and bool(CFG.training.mixed_precision)
    
    if use_amp:
        print("✓ Mixed Precision (FP16) enabled - 2x faster training!")
    
    # CUDA optimizations
    if device == "cuda":
        print_gpu_memory()

    # Optimal settings for Nautilus cluster
    BATCH_SIZE = CFG.training.batch_size  # Use 8-12 from config
    NUM_WORKERS = min(8, CFG.training.num_workers)  # Parallel data loading
    print(f"📊 Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

    tiles_dir: Path = CFG.datasets["mndws"].tiles_dir
    if not tiles_dir.exists(): raise FileNotFoundError(f"Tiles dir not found: {tiles_dir}")
    files = sorted([p.resolve() for p in tiles_dir.glob("*.npz")])
    if not files: raise FileNotFoundError(f"No .npz tiles found in {tiles_dir}")

    splits_dir = CFG.root / "data" / "splits"
    tr_txt, va_txt = splits_dir/"train.txt", splits_dir/"val.txt"
    
    if tr_txt.exists() and va_txt.exists():
        print(f"Using existing splits: {tr_txt} | {va_txt}")
        # FIX PATHS AUTOMATICALLY
        fix_split_paths(tr_txt, tiles_dir)
        fix_split_paths(va_txt, tiles_dir)
    else:
        idx = np.arange(len(files)); rng = np.random.default_rng(CFG.training.seed); rng.shuffle(idx)
        n_val = max(1, int(len(files)*0.15))
        train_files = [files[i] for i in idx[n_val:]]
        val_files   = [files[i] for i in idx[:n_val]]
        splits_dir.mkdir(parents=True, exist_ok=True)
        with open(tr_txt,"w") as f: [f.write(str(p)+"\n") for p in train_files]
        with open(va_txt,"w") as f: [f.write(str(p)+"\n") for p in val_files]
        print(f"Created new splits: {len(train_files)} train | {len(val_files)} val")

    ds_tr = NpzTileDataset(
        tr_txt, augment=True,
        expected_dyn_order=CFG.channels.dynamic_order,
        expected_stat_order=CFG.channels.static_order,
        seq_len=CFG.training.seq_len, res_m=CFG.grid.res_m,
        normalize=True, fire_boost=5.0, compute_slope_aspect=True,
    )
    ds_va = NpzTileDataset(
        va_txt, augment=False,
        expected_dyn_order=CFG.channels.dynamic_order,
        expected_stat_order=CFG.channels.static_order,
        seq_len=CFG.training.seq_len, res_m=CFG.grid.res_m,
        normalize=True, fire_boost=5.0, compute_slope_aspect=True,
    )

    with open(tr_txt) as f:
        train_files = [Path(line.strip()) for line in f if line.strip()]
    sampler, pos_tile_frac = build_train_sampler(train_files, target_pos_fraction=0.5)
    print(f"Train tiles: {len(train_files)} | Val tiles: {sum(1 for _ in open(va_txt))} | "
          f"Train tiles with any fire: {pos_tile_frac*100:.2f}%")

    # CUDA optimization: pin_memory + workers
    pin = (device == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=(NUM_WORKERS>0))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=(NUM_WORKERS>0))

    x_d, x_s, _ = next(iter(dl_tr))
    Cd, Cs, tile_size = x_d.shape[2], x_s.shape[1], x_d.shape[-1]
    print(f"Model channels — dynamic: {Cd}, static: {Cs}, tile: {tile_size}x{tile_size}")

    # Larger model - we have the GPU memory for it!
    HIDDEN_CHANNELS = CFG.training.hidden_channels
    model = ConvLSTMUNet(Cd=Cd, Cs=Cs, hidden=HIDDEN_CHANNELS).to(device)
    print(f"✓ Model with hidden={HIDDEN_CHANNELS} channels on {device}")
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if device == "cuda":
        print_gpu_memory()

    ckpt_path = CFG.checkpoints.model
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # v2 architecture has encoder downsampling + skip connections + smaller output
    # resolution. v1 state_dicts share ~no layer names, so loading one with
    # strict=False silently re-initializes the whole model — worse, it wastes
    # time printing "Warm-started" when nothing was actually restored. Only
    # warm-start from a checkpoint whose stored arch_version matches.
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device)
            ckpt_arch = state.get("arch_version")
            if ckpt_arch == ARCH_VERSION:
                model.load_state_dict(state["state_dict"], strict=False)
                print(f"✓ Warm-started from {ckpt_path} (arch={ckpt_arch})")
            else:
                print(
                    f"[info] Skipping warm-start: checkpoint arch={ckpt_arch!r} "
                    f"does not match model arch={ARCH_VERSION!r}. Training fresh."
                )
        except Exception as e:
            print(f"[warn] Could not inspect checkpoint: {e}")

    prior_pos = compute_pixel_prior(train_files, max_files=2000)
    pos_weight = make_pos_weight(prior_pos, cap=15.0, device=device)
    init_final_bias_to_prior(model, prior_pos)
    print(f"Pixel prior ≈ {prior_pos:.5f} → pos_weight={float(pos_weight.item()):.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.training.lr, weight_decay=CFG.training.weight_decay)
    if CFG.training.warmup_epochs and CFG.training.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < CFG.training.warmup_epochs:
                return float(epoch + 1) / float(max(1, CFG.training.warmup_epochs))
            progress = (epoch - CFG.training.warmup_epochs) / max(1, (CFG.training.epochs - CFG.training.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress,0.0),1.0)))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.training.epochs)

    scaler = GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    dice_lambda     = float(CFG.training.loss.get("dice_weight", 0.5))
    dice_dilate_px  = int(CFG.training.loss.get("dice_dilate_px", 1))
    focal_cfg       = CFG.training.loss.get("focal", {"enable": False})
    do_tta          = bool(CFG.training.metrics.get("tta", False))

    best_val_loss = float("inf"); best_val_csi = -1.0
    best_th = float(CFG.training.metrics.get("threshold", 0.5))
    patience = 0

    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80 + "\n")

    for ep in range(1, CFG.training.epochs + 1):
        model.train()
        tr_loss = 0.0
        epoch_start = time.time()
        
        for x_d, x_s, y in dl_tr:
            # x_d is 5D [B,T,C,H,W] → leave layout as-is
            x_d = x_d.to(device, non_blocking=True)
            # x_s is 4D [B,Cs,H,W] → channels_last helps on CUDA
            x_s = x_s.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y   = y.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            # Mixed precision training
            with autocast("cuda", enabled=use_amp):
                logits = model(x_d, x_s)
                logits = _resize_to_target(logits, y)
                loss = loss_bce_dice_focal(
                    logits, y, pos_weight=pos_weight,
                    dice_lambda=dice_lambda, focal=focal_cfg,
                    dice_dilate_px=dice_dilate_px,
                )
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.training.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.training.grad_clip)
                opt.step()
            
            tr_loss += loss.item()
        
        tr_loss /= max(1, len(dl_tr))
        sch.step()
        epoch_time = time.time() - epoch_start

        # Validation
        va_loss, va_iou, va_dice, va_csi = eval_metrics(
            model, dl_va, device, thresh=best_th, pos_weight=pos_weight,
            dice_lambda=dice_lambda, use_tta=do_tta, use_amp=use_amp,
        )
        csi_search, th_search = find_best_threshold(
            model, dl_va, device, candidates=np.linspace(0.30, 0.70, 9),
            pos_weight=pos_weight, dice_lambda=dice_lambda, use_tta=do_tta, use_amp=use_amp,
        )

        print(f"Epoch {ep:03d} ({epoch_time:.1f}s) | "
              f"train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"IoU {va_iou:.3f} | Dice {va_dice:.3f} | CSI {va_csi:.3f} | "
              f"bestCSI {csi_search:.3f}@{th_search:.2f}")
        
        if device == "cuda" and ep % 5 == 0:
            print_gpu_memory()

        improved = False
        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss; improved = True
        if csi_search > best_val_csi + 1e-4:
            best_val_csi = csi_search; best_th = th_search; improved = True

        if improved:
            patience = 0
            # Stamp arch_version into the filename so v2 checkpoints don't
            # collide with v1 files already in checkpoints/ (and so tilesvc /
            # downstream loaders can refuse to load the wrong arch).
            name = (
                f"convlstm_unet_{ARCH_VERSION}_Cd{Cd}_Cs{Cs}_"
                f"H{HIDDEN_CHANNELS}_T{CFG.training.seq_len}_nautilus.pt"
            )
            ckpt_out = Path(CFG.checkpoints.model).with_name(name)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch_version": ARCH_VERSION,
                    "cd": Cd, "cs": Cs, "tile_size": tile_size,
                    "hidden": HIDDEN_CHANNELS,
                    "dyn_order": CFG.channels.dynamic_order,
                    "stat_order": CFG.channels.static_order,
                    "prior_pos": prior_pos, "best_threshold": best_th,
                    "val_loss": best_val_loss, "val_csi": best_val_csi,
                },
                ckpt_out,
            )
            print(f"  ✓ Saved → {ckpt_out.name}  (CSI={best_val_csi:.3f}, th={best_th:.2f})")
        else:
            patience += 1
            if patience >= CFG.training.early_stop:
                print("Early stopping."); break

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print(f"Best Val CSI: {best_val_csi:.3f} @ threshold {best_th:.2f}")
    print("="*80)

if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()