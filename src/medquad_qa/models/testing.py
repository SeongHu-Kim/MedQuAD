"""Offline fixtures: a tiny random-init Qwen3 model with a ChatML template, built without network access.

Used by tests (models, training, and any teammate needing a real HF code path). Outputs are meaningless.
"""

from __future__ import annotations

from pathlib import Path

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

_CORPUS = [
    "You are a careful assistant for general medical information.",
    "What are the symptoms of anemia? Fatigue, pale skin and shortness of breath.",
    "Use only the evidence. Cite [E1] or [E2]. INSUFFICIENT_EVIDENCE if nothing applies.",
    "How is high blood pressure treated? Lifestyle changes and medicines such as diuretics.",
    "The quick brown fox jumps over the lazy dog 0123456789 .,;:!?()[]<>=\"'-",
]


def build_tiny_tokenizer() -> object:
    """Byte-level BPE trained on a few synthetic sentences, with ChatML special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(_CORPUS * 4, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",  # noqa: S106
        pad_token="<|endoftext|>",  # noqa: S106
        additional_special_tokens=specials[1:],
    )
    fast.chat_template = CHATML_TEMPLATE
    return fast


def build_tiny_qwen(seed: int = 0) -> tuple[object, object]:
    """Return (model, tokenizer): a 2-layer Qwen3ForCausalLM with random weights."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tokenizer = build_tiny_tokenizer()
    config = Qwen3Config(
        vocab_size=len(tokenizer),  # type: ignore[arg-type]
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # type: ignore[attr-defined]
        pad_token_id=tokenizer.pad_token_id,  # type: ignore[attr-defined]
        bos_token_id=None,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)
    model.generation_config.eos_token_id = config.eos_token_id
    model.generation_config.pad_token_id = config.pad_token_id
    model.eval()
    return model, tokenizer


def save_tiny_qwen(path: Path, seed: int = 0) -> Path:
    """Write the tiny model + tokenizer as a local HF directory (usable as GeneratorConfig.model_id)."""
    model, tokenizer = build_tiny_qwen(seed)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)  # type: ignore[attr-defined]
    tokenizer.save_pretrained(path)  # type: ignore[attr-defined]
    return path
