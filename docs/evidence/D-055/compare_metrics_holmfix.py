"""Compare E4 metrics.json before/after the Holm-family fix: raw values must be identical.

Usage: python compare_metrics_holmfix.py <old metrics.json> <new metrics.json>   (prints paths and counts only)
Allowed differences: removed sets.*.paired_holm_adjusted_p; added sets.*.paired_holm_adjusted_p_primary /
_exploratory; added provenance. Anything else differing is a FAILURE (exit 1).
"""

import json
import sys


def leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


old = dict(leaves(json.load(open(sys.argv[1]))))
new = dict(leaves(json.load(open(sys.argv[2]))))
ALLOWED_REMOVED = ("/paired_holm_adjusted_p/",)
ALLOWED_ADDED = ("/paired_holm_adjusted_p_primary/", "/paired_holm_adjusted_p_exploratory/")
common = sorted(set(old) & set(new))
differ = [p for p in common if old[p] != new[p]]
removed = sorted(set(old) - set(new))
added = sorted(set(new) - set(old))
bad_removed = [p for p in removed if not any(a in p for a in ALLOWED_REMOVED)]
bad_added = [p for p in added if not (any(a in p for a in ALLOWED_ADDED) or p.startswith("/provenance/"))]
report = {
    "old_leaf_paths": len(old),
    "new_leaf_paths": len(new),
    "compared_common_paths": len(common),
    "identical": len(common) - len(differ),
    "differing": differ,
    "removed": removed,
    "added_holm": [p for p in added if any(a in p for a in ALLOWED_ADDED)],
    "added_provenance_count": sum(p.startswith("/provenance/") for p in added),
    "unexpected_removed": bad_removed,
    "unexpected_added": bad_added,
}
report["passed"] = not differ and not bad_removed and not bad_added
print(json.dumps(report, indent=1))
sys.exit(0 if report["passed"] else 1)
