from __future__ import annotations

from typing import Any

from training_fixtures import synthetic_record

from medquad_qa.training.lora import PadCollator
from medquad_qa.training.sft_data import IGNORE_INDEX, SFTBuilder, default_closed_book_messages


def _full_template_ids(tok: Any, question: str, answer: str) -> list[int]:
    msgs = [m.model_dump() for m in default_closed_book_messages(question)]
    msgs.append({"role": "assistant", "content": answer})
    ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def test_prompt_masked_and_matches_chat_template(tiny_tokenizer: Any) -> None:
    rec = synthetic_record(1, "Anemia", "symptoms", "Fatigue and pale skin.", "g1")
    ex = SFTBuilder(tiny_tokenizer, max_seq_len=512).build_one(rec)
    assert ex is not None and not ex.truncated
    prompt_len = sum(1 for lab in ex.labels if lab == IGNORE_INDEX)
    assert ex.labels[prompt_len:] == ex.input_ids[prompt_len:]
    assert all(lab == IGNORE_INDEX for lab in ex.labels[:prompt_len])
    # supervised part = answer + <|im_end|>; whole sequence is the template's rendering minus the trailing newline
    eot = tiny_tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert ex.input_ids[-1] == eot
    full = _full_template_ids(tiny_tokenizer, rec.question, rec.answer)
    assert full[: len(ex.input_ids)] == ex.input_ids


def test_truncation_at_sentence_boundary_without_eos(tiny_tokenizer: Any) -> None:
    answer = " ".join(f"Sentence number {i} has a few words." for i in range(60))
    rec = synthetic_record(2, "Gout", "treatment", answer, "g2")
    builder = SFTBuilder(tiny_tokenizer, max_seq_len=200)
    ex = builder.build_one(rec)
    assert ex is not None and ex.truncated
    assert len(ex.input_ids) <= 200
    eot = tiny_tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert eot not in ex.labels
    kept_text = tiny_tokenizer.decode([t for t in ex.labels if t != IGNORE_INDEX])
    assert kept_text.rstrip().endswith(".") and answer.startswith(kept_text.strip())


def test_build_report_counts_and_exclusions(tiny_tokenizer: Any) -> None:
    long_answer = " ".join(f"Line {i} is here." for i in range(200))
    recs = [
        synthetic_record(1, "A", "symptoms", "Short answer.", "g"),
        synthetic_record(2, "B", "symptoms", "Boilerplate text.", "g", flags=["boilerplate_answer"]),
        synthetic_record(3, "C", "treatment", long_answer, "g"),
    ]
    examples, report = SFTBuilder(tiny_tokenizer, max_seq_len=256).build(recs)
    d = report.to_dict()
    assert d["n_records_in"] == 3 and d["n_examples"] == 2 == len(examples)
    assert d["excluded"] == {"boilerplate_answer": 1}
    assert d["truncated_by_source"] == {"SYN": 1} and d["answer_tokens_dropped_by_truncation"] > 0


def test_pad_collator() -> None:
    batch = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}]
    out = PadCollator(pad_id=0)(batch)
    assert out["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert out["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
