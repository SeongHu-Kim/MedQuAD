"""E4b step 3(b) v2b report (read-only). Reads ONLY the v2b build directory (manifests = IDs/flags/counts; build
reports = counts) plus the gate output file. Prints counts, shares and hashes only: no record text, no record IDs.
Usage: e4b_step3b_report.py <v2b_build_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path

B = Path(sys.argv[1])
SERVING_DIFF_TYPE = {0: 23.5, 1: 21.5, 2: 17.3, 3: 22.2, 4: 15.5}  # D-031, gold retrieved, n=1,570 (e4b_projection.log)
SERVING_ALL_TYPES = {"0": 20.2, "1": 18.7, "2": 16.4, "3+": 44.8}  # D-031 other on-topic, all types (e4b_d031_recount.log)
fail = []


def load_rows(name):
    return [json.loads(x) for x in (B / name).open(encoding="utf-8") if x.strip()]


def pct(c, n):
    return f"{c} ({c / n:.1%})" if n else f"{c}"


def cue_from_rows(rows):
    by = {}
    for r in rows:
        by.setdefault(int(r["on_topic_total"]), Counter())[r["format"]] += 1
    n = sum(sum(c.values()) for c in by.values())
    tot = Counter()
    for c in by.values():
        tot.update(c)
    return (sum(max(c.values()) for c in by.values()) / n, max(tot.values()) / n, n) if n else (None, None, 0)


for split, man, rep_name in (("train", "sft_record_ids.jsonl", "sft_build_train.json"),
                              ("validation_rag", "val_record_ids.jsonl", "sft_build_val_rag.json")):
    rows = load_rows(man)
    rep = json.loads((B / rep_name).read_text())
    v = rep.get("v2b")
    print(f"\n######## {split}: {rep_name} (mix_version {rep.get('mix_version')})")
    if v is None:
        print("  NO v2b block in report")
        fail.append(f"{split}: no v2b block")
        continue
    rag = [r for r in rows if r["format"] != "closed_book"]
    ans = [r for r in rag if r["format"] == "rag_answerable"]
    sen = [r for r in rag if r["format"] == "rag_insufficient"]
    same = [r for r in sen if r["sentinel_kind"] == "sentinel_same_topic"]
    info = [r for r in sen if r["sentinel_kind"] == "sentinel_info_offtopic"]

    # --- manifest self-consistency (independent of the report)
    bad = sum(r["on_topic_total"] != sum(bool(x) for x in r["evidence_on_topic"]) for r in rag)
    bad += sum(len(r["evidence_record_ids"]) > 5 or len(r["evidence_record_ids"]) != len(r["evidence_chunk_ids"]) for r in rag)
    bad += sum(r["record_id"] not in r["evidence_record_ids"] for r in ans)
    bad += sum(r["on_topic_total"] != 0 for r in info)
    print(f"  manifest consistency violations: {bad}")
    if bad:
        fail.append(f"{split}: {bad} manifest inconsistencies")

    print("  plan vs realised:", json.dumps(v["plan_vs_realised"]))
    print("  assignment skips:", v["assignment_skips"])
    print("  question-type mix by format:", json.dumps(v["qtype_mix_by_format"]))

    # --- PRIMARY: realised added same-topic blocks in answerable prompts vs serving DIFFERENT-type split
    n = len(ans)
    real = Counter(int(r["same_topic_realised_final"]) for r in ans)
    drawn = Counter(int(r["same_topic_drawn"]) for r in ans)
    print(f"  PRIMARY answerable added same-topic blocks (n={n}): value: realised-final | drawn | serving different-type %")
    for k in range(5):
        print(f"    {k}: {pct(real[k], n)} | {pct(drawn[k], n)} | {SERVING_DIFF_TYPE[k]}%")
    print(f"    context only, serving ALL types other on-topic (incl. same-type, excluded by design): {SERVING_ALL_TYPES}")

    print("  on_topic_total exact 0-5 per class (report):", json.dumps(v["on_topic_total_hist"]))
    for name, grp in (("rag_answerable", ans), ("rag_insufficient", sen), ("sentinel_same_topic", same),
                      ("sentinel_info_offtopic", info)):
        mine = {str(k): sum(int(r["on_topic_total"]) == k for r in grp) for k in range(6)}
        if mine != v["on_topic_total_hist"].get(name):
            fail.append(f"{split}: on_topic hist mismatch {name}")
            print(f"    MISMATCH recomputed {name}: {mine}")
    print("  same-topic sentinels with on_topic_total == 0 after length drops:",
          v["sentinel_same_topic_zero_on_topic_after_length_drops"])
    print("  zero-candidate per class:", v["zero_candidate"], "| by sentinel qtype:", v["zero_candidate_by_sentinel_qtype"])

    cc = v["cue_check"]
    print("  CUE CHECK (rule = best answer/refuse rule from on_topic_total alone):")
    for key in ("all", "excluding_zero_candidate"):
        c = cc[key]
        print(f"    {key}: n={c.get('n')} rule_accuracy={c.get('rule_accuracy'):.3f} majority_baseline={c.get('majority_baseline'):.3f}"
              f" excess={c.get('rule_accuracy') - c.get('majority_baseline'):+.3f} counts={c.get('counts_by_on_topic_total')}")
    ra, mb, nn = cue_from_rows(rag)
    print(f"    recomputed from manifest (all): rule_accuracy={ra:.3f} majority_baseline={mb:.3f} n={nn}")
    if abs(ra - cc["all"]["rule_accuracy"]) > 1e-9:
        fail.append(f"{split}: cue check mismatch")
    print("    per sentinel question type (answerable + sentinel rows of that type):")
    for qt, c in cc["per_sentinel_qtype"].items():
        if c.get("n"):
            print(f"      {qt}: n={c['n']} rule_accuracy={c['rule_accuracy']:.3f} majority={c['majority_baseline']:.3f}"
                  f" excess={c['rule_accuracy'] - c['majority_baseline']:+.3f}")

    print("  REFUSE SHARE by on_topic_total (descriptive; sentinels / all RAG prompts at that count):")
    for k in range(0, 6):
        at = [r for r in rag if int(r["on_topic_total"]) == k]
        s = sum(r["format"] == "rag_insufficient" for r in at)
        if at:
            print(f"    {k}: {s}/{len(at)} = {s / len(at):.1%}")
    print("  guard drops (candidate chunks, first failing guard):", json.dumps(v["guard_drops_candidate_chunks"]))
    g7 = v["g7_drops_by_pair"]
    print(f"  G7 drops by pair ({len(g7)} pairs, total {sum(g7.values())}):", json.dumps(g7))
    print("  length drops:", v["length_drops"])
    print("  gold position:", rep.get("gold_position"))
    print("  RAG prompt tokens:", rep.get("rag_prompt_tokens"), "| distractor drop events:", rep.get("distractors_dropped_for_length"))
    tr = Counter((r.get("sentinel_kind") or r["format"], bool(r["truncated"])) for r in rows)
    print("  truncated rows (kind, truncated): count:", {f"{k[0]},{k[1]}": c for k, c in sorted(tr.items())})
    cb = rep.get("closed_book") or {}
    if cb:
        print(f"  closed-book builder: n_examples={cb.get('n_examples')} n_truncated={cb.get('n_truncated')} max_seq_len={cb.get('max_seq_len')}")

print("\n######## contrast: original v2 build sft-mix-v2-20261007-035957, refuse share by on-topic total (D-031)")
print("  source: docs/evidence/D-053/e4b_d031_recount.log section A (no new reads): train answerable 2,697 with exactly")
print("  1 on-topic block (gold; 0 on-topic distractors), sentinels 500 with 0; validation answerable 222 with 1, 1 with 0")
print("  (topic_key None), sentinels 31 with 0.")
print("    train:      0: 500/500 = 100.0% | 1: 0/2697 = 0.0% | 2-5: none")
print("    validation: 0: 31/32 = 96.9%   | 1: 0/222 = 0.0%  | 2-5: none")

summ = json.loads((B / "build_summary.json").read_text())
print("\n######## build_summary:", json.dumps({k: summ[k] for k in ("train_examples", "train_tokens", "val_sets")}),
      "| versions:", json.dumps({k: summ["versions"].get(k) for k in ("mix_version", "split_version", "corpus_version",
                                                                         "message_builder", "rag_prompt_version")}),
      "| embedder revision:", summ["versions"].get("embedder", {}).get("revision"))
pc = B / "protected_check.json"
if pc.exists():
    g = json.loads(pc.read_text())
    print("gate:", json.dumps({k: g[k] for k in ("passed", "rows", "protected_record_ids", "train_probe_sft_overlap")}),
          "errors:", len(g["errors"]), "warnings:", len(g["warnings"]))
    if not g["passed"]:
        fail.append("gate not passed")
else:
    fail.append("protected_check.json missing")
print("REPORT:", "OK" if not fail else f"FAIL {fail}")
sys.exit(1 if fail else 0)
