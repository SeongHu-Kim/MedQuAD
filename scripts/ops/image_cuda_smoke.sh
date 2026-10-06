#!/usr/bin/env bash
# CUDA smoke INSIDE a built API image (its own torch), with the same least-privilege flags as the stack.
# Checks: torch.cuda available, bf16 matmul, and the rotary-style batched outer product that needs
# TORCH_DISABLE_NATIVE_JIT=1 (D-030). Result: artifacts/service/gpu_image_smoke.json
# Usage: scripts/ops/image_cuda_smoke.sh [image]   Owner: service-platform-engineer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${1:-medquad-api:gpu}"
OUT="$ROOT/artifacts/service/gpu_image_smoke.json"
set +e
out=$(docker run --rm --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp --network none --device nvidia.com/gpu=all "$IMAGE" python -c '
import os, torch
assert torch.cuda.is_available(), "cuda not available"
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
mm = float((a @ a).float().sum())
outer = torch.randn(1, 64, 1, device="cuda") @ torch.randn(1, 1, 32, device="cuda")
torch.cuda.synchronize()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0),
      "TORCH_DISABLE_NATIVE_JIT=" + os.environ.get("TORCH_DISABLE_NATIVE_JIT", "unset"),
      "matmul_ok", "outer_ok", tuple(outer.shape), "uid=%d" % os.getuid())
' 2>&1)
rc=$?
set -e
echo "$out"
python3 - "$OUT" "$IMAGE" "$rc" "$out" <<'PY'
import json, sys, datetime, subprocess
out, image, rc, text = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
image_id = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", image], capture_output=True, text=True).stdout.strip()
json.dump({"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "image": image, "image_id": image_id, "exit_code": rc, "output": text.strip()[-800:],
           "result": "pass" if rc == 0 else "fail"}, open(out, "w"), indent=2)
PY
exit "$rc"
