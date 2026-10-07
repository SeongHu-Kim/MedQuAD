"""E4b step 2 report (DRAFT). Read-only. Prints IDs/counts only, never record text.

Reads: the build dir, data/manifests/split_manifest{.jsonl,.meta.json} (IDs/splits only),
records_train.jsonl and records_validation.jsonl (for the no-text check). Never reads the test export.
Exit 1 on any hard failure; the >3% distractor-drop share is a FLAG (printed), not a failure."""

import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
fail: list[str] = []
ID = re.compile(r"mq-[0-9a-f]{16}")
ROW_KEYS = {"record_id", "split_group_id", "format", "evidence_record_ids", "truncated", "split_version"}
FORMATS = {"closed_book", "rag_answerable", "rag_insufficient"}
PINNED_BGE = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
V1_CLOSED = 12636  # v1 sft_build_train.json n_examples
DROP_FLAG = 0.03

frozen = json.loads(Path("data/manifests/split_manifest.meta.json").read_text())["split_version"]
split_of = {}
for line in Path("data/manifests/split_manifest.jsonl").open(encoding="utf-8"):
    m = json.loads(line)
    split_of[m["record_id"]] = m["split"]

texts = set()  # train + validation questions/answers only
for split in ("train", "validation"):
    for line in Path(f"data/processed/exports/records_{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        texts.update(t.strip() for t in (r.get("question", ""), r.get("answer", "")) if len(t.strip()) >= 20)


def strings(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield f"{path}/{k}", str(k)
            yield from strings(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from strings(v, f"{path}/{i}")
    elif isinstance(o, str):
        yield path, o


def load(f: Path) -> list:
    if f.suffix == ".jsonl":
        return [json.loads(x) for x in f.open(encoding="utf-8") if x.strip()]
    return [json.loads(f.read_text(encoding="utf-8"))]


# 1. ID manifests: schema, ID format, split_version
rows = {n: load(out / n) for n in ("sft_record_ids.jsonl", "val_record_ids.jsonl")}
for name, rs in rows.items():
    for i, r in enumerate(rs):
        if set(r) != ROW_KEYS:
            fail.append(f"{name}:{i} keys differ: {sorted(set(r) ^ ROW_KEYS)}")
            continue
        ids = [r["record_id"], *r["evidence_record_ids"]]
        if not all(ID.fullmatch(x) for x in ids):
            fail.append(f"{name}:{i} non-ID value")
        if r["format"] not in FORMATS:
            fail.append(f"{name}:{i} unknown format")
        if r["split_version"] != frozen:
            fail.append(f"{name}:{i} split_version mismatch")
print("split_version in manifests:", sorted({r.get("split_version") for v in rows.values() for r in v}), "| frozen:", frozen)

# 2. distractor/evidence pools: train build -> train records only; validation build -> validation only
for name, want in (("sft_record_ids.jsonl", "train"), ("val_record_ids.jsonl", "validation")):
    ids = {x for r in rows[name] for x in (r["record_id"], *r["evidence_record_ids"])}
    bad = sorted(x for x in ids if split_of.get(x) != want)
    print(f"pool check {name}: {len(ids)} distinct IDs, all in {want}: {'OK' if not bad else f'{len(bad)} NOT'}")
    if bad:
        fail.append(f"{name}: {len(bad)} IDs outside {want} (first: {bad[:3]})")

# 3. every output file: no record text, no test-split record ID
for f in sorted(out.iterdir()):
    if f.suffix not in (".json", ".jsonl"):
        fail.append(f"unexpected output file type: {f.name}")
        continue
    text_hits = test_hits = unknown = 0
    for o in load(f):
        for p, s in strings(o):
            if len(s.split()) >= 8 or (len(s) >= 20 and s.strip() in texts):
                text_hits += 1
                if text_hits <= 3:
                    fail.append(f"text-like value in {f.name} at {p} (value not printed)")
            for rid in ID.findall(s):
                sp = split_of.get(rid)
                test_hits += sp == "test"
                unknown += sp is None
    print(f"{f.name}: text-like={text_hits} test-split IDs={test_hits} unknown IDs={unknown}")
    if test_hits or unknown:
        fail.append(f"{f.name}: {test_hits} test-split IDs, {unknown} unknown IDs")

# 4. embedder actually loaded (build_summary records SentenceTransformerEmbedder.revision = resolved snapshot)
summ = json.loads((out / "build_summary.json").read_text())
emb = summ["versions"].get("embedder", {})
print("embedder loaded:", emb.get("model_id"), "@", emb.get("revision"))
if emb.get("revision") != PINNED_BGE:
    fail.append(f"embedder revision {emb.get('revision')} != pinned {PINNED_BGE}")
print("versions:", {k: summ["versions"].get(k) for k in ("mix_version", "rag_prompt_version", "message_builder", "split_version", "corpus_version")})

# 5. counts vs pre-declared targets
tr = json.loads((out / "sft_build_train.json").read_text())
c = tr["counts"]
n_a, n_i, n_c = c.get("rag_answerable", 0), c.get("rag_insufficient", 0), c.get("closed_book", 0)
print("train counts:", c, "| plan:", {k: tr["plan"][k] for k in ("n_rag_answerable", "n_rag_insufficient", "n_closed_book")})
print(f"sentinel share of RAG examples: {n_i}/{n_a + n_i} = {n_i / max(1, n_a + n_i):.3f} (pre-declared ~0.16)")
print(f"closed-book vs v1: {n_c}/{V1_CLOSED} = {n_c / V1_CLOSED:.3f} (pre-declared 'about half')")
print("skipped (train):", tr["skipped"], "| val sets:", summ["val_sets"])

# 6. truncation, token limits, and the 2048 (train) vs 3072 (serving) budget gap
k = tr["plan"]["k"]
for name, rep_file in (("sft_record_ids.jsonl", "sft_build_train.json"), ("val_record_ids.jsonl", "sft_build_val_rag.json")):
    rep = json.loads((out / rep_file).read_text())
    rag = [r for r in rows[name] if r["format"] != "closed_book"]
    dropped = sum(len(r["evidence_record_ids"]) < k for r in rag)
    share = dropped / max(1, len(rag))
    flag = "  <-- FLAG: above 3%, user decision needed before step 3" if share > DROP_FLAG else ""
    print(f"[{rep_file}] RAG examples with distractors dropped to fit the training budget: {dropped}/{len(rag)} = {share:.3%}{flag}")
    print(f"[{rep_file}]   drop events: {rep['distractors_dropped_for_length']}; skipped: {rep['skipped']}")
    tgt = [r for r in rag if r["format"] == "rag_answerable"]
    print(f"[{rep_file}]   answerable targets cut at {rep['plan']['max_target_tokens']} tokens: {sum(r['truncated'] for r in tgt)}/{len(tgt)}")
    print(f"[{rep_file}]   RAG prompt tokens: {rep['rag_prompt_tokens']}; gold position: {rep['gold_position']}")
print("note: every drop is counted as caused by the 2048 budget; this is an upper bound for drops that serving's 3072 limit would not also make")
cb = tr["closed_book"] or {}
print(f"closed-book truncated at max_seq_len {cb.get('max_seq_len')}: {cb.get('n_truncated')}/{cb.get('n_examples')}")
print("limits: train RAG prompt+target <= 2048 (lora_sft_mixed_v2.yaml), target <= 256, closed-book <= 1024 (as v1);"
      " serving: max_input_tokens 3072 (prompt only), max_new_tokens 256")

# 7. leakage gate summary
g = json.loads((out / "protected_check.json").read_text())
print("leakage gate:", {kk: g[kk] for kk in ("passed", "rows", "protected_record_ids", "train_probe_sft_overlap")}, "errors:", len(g["errors"]))
if not g["passed"]:
    fail.append("leakage gate not passed")

for f_ in fail[:20]:
    print("FAIL:", f_)
print("REPORT:", "FAIL" if fail else "OK")
sys.exit(1 if fail else 0)
