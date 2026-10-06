#!/usr/bin/env bash
# Validate GPU access from a container via CDI (D-017). Read-only with respect to the host:
# no runtime/daemon/socket changes, no sudo. Owner: service-platform-engineer.
#
# Stage 1: nvidia-smi in the pinned CUDA base image with --device nvidia.com/gpu=all.
# Stage 2: a real CUDA matmul with the host's already-installed torch (mounted read-only into the
#          pinned python image), so the CUDA path is proven before building the ~7 GB GPU API image.
# Exit 0 only if both stages pass. Result JSON: artifacts/service/gpu_validation.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/artifacts/service"
OUT="$OUT_DIR/gpu_validation.json"
CUDA_IMAGE="nvidia/cuda:13.0.1-base-ubuntu24.04@sha256:f8ef28f579ea42a44b415d2c5d46f788e6a9b395c6c83f2929416e1fc192c143"
PY_IMAGE="python:3.12-slim-bookworm@sha256:7753c33391fc9f01d1984375bf375eb6686d52ba10db6043a86634a5ccf90dcf"
SITE="$ROOT/.venv/lib/python3.12/site-packages"
HARDEN=(--rm --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only --network none
        --device nvidia.com/gpu=all)
mkdir -p "$OUT_DIR"

stage1="fail"; stage2="fail"; smi=""; torch_out=""

echo "[gpu-check] CDI spec dirs: $(docker info --format '{{json .CDISpecDirs}}')"
if ! compgen -G "/etc/cdi/*.yaml" >/dev/null && ! compgen -G "/var/run/cdi/*.yaml" >/dev/null; then
  echo "[gpu-check] no CDI spec found (expected /var/run/cdi/nvidia.yaml)" >&2
fi

docker pull -q "$CUDA_IMAGE" >/dev/null
docker pull -q "$PY_IMAGE" >/dev/null

echo "[gpu-check] stage 1: nvidia-smi via CDI in $CUDA_IMAGE"
if smi=$(docker run "${HARDEN[@]}" "$CUDA_IMAGE" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1); then
  stage1="pass"
fi
echo "$smi"

echo "[gpu-check] stage 2: CUDA matmul with host torch (read-only mount) in $PY_IMAGE"
if [[ -d "$SITE/torch" ]]; then
  if torch_out=$(docker run "${HARDEN[@]}" -e PYTHONPATH=/site -e HOME=/tmp --tmpfs /tmp \
      -v "$SITE:/site:ro" "$PY_IMAGE" python -c '
import torch
assert torch.cuda.is_available(), "cuda not available"
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
s = float((a @ a).float().sum())
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "matmul_ok", round(s, 2))
' 2>&1); then
    stage2="pass"
  fi
else
  torch_out="host torch not found at $SITE/torch"
fi
echo "$torch_out"

python3 - "$OUT" "$stage1" "$stage2" "$smi" "$torch_out" "$CUDA_IMAGE" "$PY_IMAGE" <<'EOF'
import json, sys, datetime
out, s1, s2, smi, torch_out, cuda_img, py_img = sys.argv[1:8]
json.dump({
    "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "stage1_nvidia_smi": s1, "stage1_output": smi.strip()[-500:],
    "stage2_torch_cuda": s2, "stage2_output": torch_out.strip()[-500:],
    "cuda_image": cuda_img, "python_image": py_img,
    "result": "pass" if s1 == s2 == "pass" else "fail",
}, open(out, "w"), indent=2)
EOF
echo "[gpu-check] stage1=$stage1 stage2=$stage2 -> $OUT"
[[ "$stage1" == "pass" && "$stage2" == "pass" ]]
