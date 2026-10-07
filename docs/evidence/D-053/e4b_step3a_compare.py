"""E4b step 3(a) comparison (read-only): rebuilt sft-mix-v2 vs the original build. Prints hashes, key names and
paths only (no record text; the compared files contain IDs and counts only).

Rule (user-approved):
- Byte identity (SHA-256) for: sft_record_ids.jsonl, val_record_ids.jsonl, sft_build_train.json,
  sft_build_val.json, sft_build_val_rag.json.
- build_summary.json: remove ONLY the key "out" (the build directory path); everything else must be identical.
  ("versions.exports_manifest" is a fixed repo path and is NOT removed.)
- protected_check.json: in "inputs" (a dict keyed by input file path -> sha256), replace ONLY the build-directory
  prefix of the keys with "<BUILD>"; the sha256 values and every other field must be identical.
- Any other difference fails (exit 1). Path-only differences are printed old/new.
Usage: e4b_step3a_compare.py <old_build_dir> <new_build_dir>
"""

import hashlib
import json
import sys
from pathlib import Path

old, new = Path(sys.argv[1]), Path(sys.argv[2])
fail = []
DATA = ("sft_record_ids.jsonl", "val_record_ids.jsonl", "sft_build_train.json", "sft_build_val.json",
        "sft_build_val_rag.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


print("== byte identity (5 data files)")
for f in DATA:
    a, b = sha(old / f), sha(new / f)
    print(f"  {'MATCH   ' if a == b else 'MISMATCH'} {f}  old={a}  new={b}")
    if a != b:
        fail.append(f)

print("== build_summary.json (only 'out' removed)")
so, sn = (json.loads((d / "build_summary.json").read_text()) for d in (old, new))
print(f"  out: old={so.get('out')!r}  new={sn.get('out')!r}")
so.pop("out", None)
sn.pop("out", None)
if so == sn:
    print("  IDENTICAL after removing 'out'")
else:
    keys = sorted(k for k in set(so) | set(sn) if so.get(k) != sn.get(k))
    print(f"  DIFFERENT keys after removing 'out': {keys}")
    fail.append("build_summary.json")

print("== protected_check.json (build-dir prefix of 'inputs' keys -> <BUILD>)")
po, pn = (json.loads((d / "protected_check.json").read_text()) for d in (old, new))


def norm(d: dict, build: Path) -> dict:
    out = dict(d)
    ins = {}
    for k, v in d.get("inputs", {}).items():
        nk = k.replace(str(build), "<BUILD>", 1) if k.startswith(str(build)) else k
        if nk != k:
            print(f"  inputs key path: {k!r} -> {nk!r}")
        ins[nk] = v
    out["inputs"] = ins
    return out


po_n, pn_n = norm(po, old), norm(pn, new)
if po_n == pn_n:
    print("  IDENTICAL after normalising the build-dir prefix of 'inputs' keys (sha256 values compared)")
else:
    keys = sorted(k for k in set(po_n) | set(pn_n) if po_n.get(k) != pn_n.get(k))
    print(f"  DIFFERENT keys after normalisation: {keys}")
    fail.append("protected_check.json")

extra = sorted({p.name for p in new.iterdir()} ^ {p.name for p in old.iterdir()})
if extra:
    print(f"== file sets differ: {extra}")
    fail.append("file set")
print("RESULT:", "PASS" if not fail else f"FAIL {fail}")
sys.exit(1 if fail else 0)
