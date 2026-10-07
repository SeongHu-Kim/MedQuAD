import hashlib, json, sys
sys.path.insert(0, "tests/training")
import medquad_qa.training.sft_rag_data as m
print("module file:", m.__file__)
from test_training_answerability import _records
from medquad_qa.models.testing import build_tiny_tokenizer
from medquad_qa.retrieval.embedding import HashingEmbedder
plan = m.MixPlan(n_rag_answerable=8, n_rag_insufficient=4, n_closed_book=6, k=3, max_target_tokens=400)
examples, report, rows = m.MixedSFTBuilder(build_tiny_tokenizer(), HashingEmbedder(), max_seq_len=4096).build(_records("tr", 15, 0), plan)
h = hashlib.sha256()
for ex, row in zip(examples, rows, strict=True):
    h.update(json.dumps([row, ex.input_ids, ex.labels], sort_keys=True).encode())
print("v2 hash:", h.hexdigest())
