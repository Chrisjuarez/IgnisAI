#!/usr/bin/env bash
# tools/cleanup-remote-branches.sh
#
# Interactively delete remote branches after a secrets-rotation event,
# or as routine housekeeping. Always prompts before deleting; never
# touches `main`, the current branch, or branches matching $PROTECTED_RE.
#
# Usage:
#   bash tools/cleanup-remote-branches.sh                # interactive
#   bash tools/cleanup-remote-branches.sh --dry-run      # preview only
#   bash tools/cleanup-remote-branches.sh --merged-only  # only show branches merged into main
#   REMOTE=upstream bash tools/cleanup-remote-branches.sh
#
# Notes
# -----
# Deleting a remote branch does NOT remove the secret from GitHub's
# reflog or from any cached fork. If you're cleaning up after a leak,
# you still need to rotate the credential — see docs/SECRETS_ROTATION.md.

set -euo pipefail

REMOTE="${REMOTE:-origin}"
DRY_RUN=0
MERGED_ONLY=0

# Patterns we will never delete, even if the user says yes. Adjust if
# your team uses a different default branch convention.
PROTECTED_RE='^(main|master|release/.*|hotfix/.*|prod|production)$'

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --merged-only) MERGED_ONLY=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not inside a git repository." >&2
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Remote '$REMOTE' is not configured. Set REMOTE=<name> to override." >&2
  exit 1
fi

current_branch="$(git symbolic-ref --quiet --short HEAD || echo '')"

echo "==> Fetching $REMOTE (with prune)..."
git fetch --prune "$REMOTE"

# Build the candidate list. Strip the leading 'remote/origin/' prefix.
mapfile -t candidates < <(
  git for-each-ref --format='%(refname:short)' "refs/remotes/$REMOTE/" \
    | sed -E "s#^$REMOTE/##" \
    | grep -Ev '^HEAD$' \
    | sort -u
)

if [[ ${#candidates[@]} -eq 0 ]]; then
  echo "No remote branches found on $REMOTE."
  exit 0
fi

# Optional: only show branches already merged into main.
if [[ $MERGED_ONLY -eq 1 ]]; then
  if ! git rev-parse --verify "$REMOTE/main" >/dev/null 2>&1; then
    echo "--merged-only requires $REMOTE/main to exist." >&2
    exit 1
  fi
  filtered=()
  for b in "${candidates[@]}"; do
    if git merge-base --is-ancestor "$REMOTE/$b" "$REMOTE/main"; then
      filtered+=("$b")
    fi
  done
  candidates=("${filtered[@]}")
fi

# Drop protected branches and the current branch from the candidate list
# entirely — we won't even prompt for these.
deletable=()
for b in "${candidates[@]}"; do
  if [[ "$b" =~ $PROTECTED_RE ]]; then continue; fi
  if [[ "$b" == "$current_branch" ]]; then continue; fi
  deletable+=("$b")
done

if [[ ${#deletable[@]} -eq 0 ]]; then
  echo "Nothing to clean up after applying protections."
  exit 0
fi

echo
echo "Candidate branches on $REMOTE:"
for b in "${deletable[@]}"; do
  # Show last commit short info to help the human decide.
  info=$(git log -1 --format='%h %ad %an %s' --date=short "$REMOTE/$b" 2>/dev/null || echo '?')
  printf '  %-40s  %s\n' "$b" "$info"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "[--dry-run] Not deleting anything."
  exit 0
fi

echo
echo "You will be prompted [y/N] for each branch. Default is NO."
echo "Type 'q' at any prompt to quit without further deletes."
echo

deleted_count=0
for b in "${deletable[@]}"; do
  read -r -p "Delete $REMOTE/$b ? [y/N/q] " ans
  case "${ans:-N}" in
    y|Y|yes|YES)
      echo "  -> deleting $REMOTE/$b"
      if git push "$REMOTE" --delete "$b"; then
        deleted_count=$((deleted_count + 1))
      else
        echo "  !! failed to delete $b" >&2
      fi
      ;;
    q|Q)
      echo "Stopping early."
      break
      ;;
    *)
      echo "  -> keeping $b"
      ;;
  esac
done

echo
echo "Done. Deleted $deleted_count branch(es) from $REMOTE."
echo "Reminder: deleting a branch does not retroactively scrub leaked secrets."
echo "         If this cleanup is post-leak, finish docs/SECRETS_ROTATION.md."
