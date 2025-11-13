import torch, torch.nn as nn

# ---- toggles ----
USE_GROUPNORM = True     # GN tends to be more stable than BN on small batches/MPS
GN_GROUPS     = 8
USE_ASPP      = True     # light context pyramid; set False to disable

def Norm2d(num_channels: int):
    if USE_GROUPNORM:
        # make sure num_channels % GN_GROUPS == 0
        groups = GN_GROUPS if num_channels % GN_GROUPS == 0 else 1
        return nn.GroupNorm(groups, num_channels)
    else:
        return nn.BatchNorm2d(num_channels)

def block(in_c, out_c, drop=0.1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        Norm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        Norm2d(out_c), nn.ReLU(inplace=True),
        nn.Dropout2d(drop),
    )

class ASPP(nn.Module):
    """Very light atrous spatial pyramid pooling."""
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4), mid=None, drop=0.0):
        super().__init__()
        mid = mid or (in_ch // 4)
        self.branches = nn.ModuleList()
        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), Norm2d(mid), nn.ReLU(inplace=True)
        ))
        # dilated branches
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=r, dilation=r, bias=False),
                Norm2d(mid), nn.ReLU(inplace=True)
            ))
        fused_in = mid * (1 + len(rates))
        self.proj = nn.Sequential(
            nn.Conv2d(fused_in, out_ch, 1, bias=False),
            Norm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.proj(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__(); p=k//2
        self.x2h = nn.Conv2d(in_ch, 4*hid_ch, k, padding=p, bias=True)
        self.h2h = nn.Conv2d(hid_ch, 4*hid_ch, k, padding=p, bias=True)
        self.hid_ch = hid_ch
    def init(self, B,H,W,device):
        h = torch.zeros(B,self.hid_ch,H,W,device=device); c = torch.zeros_like(h); return (h,c)
    def forward(self, x, state):
        h,c = state
        i,f,o,g = torch.chunk(self.x2h(x)+self.h2h(h), 4, dim=1)
        i,f,o,g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f*c + i*g; h = o*torch.tanh(c)
        return h,(h,c)

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, layers=1):
        super().__init__()
        self.cells = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch) for i in range(layers)])
    def forward(self, x):  # x: [B,T,C,H,W]
        B,T,C,H,W = x.shape; h = x
        for cell in self.cells:
            state = cell.init(B,H,W,x.device); outs=[]
            for t in range(T):
                out, state = cell(h[:,t], state); outs.append(out)
            h = torch.stack(outs, dim=1)
        return h[:,-1]

class UNetDecoder(nn.Module):
    def __init__(self, base=64, drop=0.1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base, base//2, 2, 2)
        self.c1  = block(base//2, base//2, drop)
        self.up2 = nn.ConvTranspose2d(base//2, base//4, 2, 2)
        self.c2  = block(base//4, base//4, drop)
        self.out = nn.Conv2d(base//4, 1, 1)  # logits
    def forward(self, z):
        z = self.up1(z); z = self.c1(z)
        z = self.up2(z); z = self.c2(z)
        return self.out(z)

class ConvLSTMUNet(nn.Module):
    def __init__(self, Cd, Cs, hidden=64, lstm_layers=2, drop=0.1):
        super().__init__()
        self.enc_dyn  = block(Cd, hidden, drop)
        self.enc_stat = block(Cs, hidden, drop)
        self.rnn      = ConvLSTM(hidden, hidden, layers=lstm_layers)
        self.fuse     = nn.Conv2d(hidden*2, hidden, 1, bias=False)
        self.fuse_bn  = Norm2d(hidden)
        self.aspp     = ASPP(hidden, hidden, rates=(1,2,4), drop=drop) if USE_ASPP else nn.Identity()
        self.act      = nn.ReLU(inplace=True)
        self.dec      = UNetDecoder(hidden, drop)
    def forward(self, x_dyn, x_stat):
        B,T,Cd,H,W = x_dyn.shape
        dyn = self.enc_dyn(x_dyn.reshape(B*T,Cd,H,W)).reshape(B,T,-1,H,W)
        rnn = self.rnn(dyn)
        stat= self.enc_stat(x_stat)
        z   = torch.cat([rnn,stat], dim=1)
        z   = self.act(self.fuse_bn(self.fuse(z)))
        z   = self.aspp(z)
        return self.dec(z)