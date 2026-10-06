"""TensorFlow/Keras answerability classifier: siamese BiGRU over (question, evidence) + lexical features.

CPU-only TensorFlow (D-014). Imported only by offline training/evaluation scripts, never by the API or the
generator. Labels come from the synthetic pair rules in ``answerability_pairs`` (non-clinical).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from medquad_qa.contracts.answerability import AnswerabilityPair, PredictorOutput
from medquad_qa.contracts.interfaces import ArtifactUnavailableError
from medquad_qa.models.answerability import LexicalFeaturizer
from medquad_qa.training.answerability_train import eval_blocks, labels, threshold_block


class KerasConfig:
    vocab_size = 30000
    question_len = 32
    evidence_len = 256
    embed_dim = 128
    gru_units = 64
    dense_units = 64
    dropout = 0.3
    learning_rate = 1e-3
    batch_size = 128
    max_epochs = 8
    patience = 2
    seed = 42

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v) and k != "as_dict"}


def _lex(
    featurizer: LexicalFeaturizer, pairs: list[AnswerabilityPair], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    x = featurizer.matrix([p.question for p in pairs], [p.evidence_text for p in pairs])
    return ((x - mean) / std).astype("float32")


def build_model(train_texts: list[str], n_lex: int, cfg: type[KerasConfig] = KerasConfig) -> Any:
    import keras
    from keras import layers

    vocab_layer = layers.TextVectorization(max_tokens=cfg.vocab_size, output_mode="int")
    vocab_layer.adapt(train_texts)
    vocab = vocab_layer.get_vocabulary()
    q_vec = layers.TextVectorization(vocabulary=vocab, output_sequence_length=cfg.question_len, name="vectorize_q")
    e_vec = layers.TextVectorization(vocabulary=vocab, output_sequence_length=cfg.evidence_len, name="vectorize_e")
    q_in = keras.Input(shape=(1,), dtype="string", name="question")
    e_in = keras.Input(shape=(1,), dtype="string", name="evidence")
    lex_in = keras.Input(shape=(n_lex,), dtype="float32", name="lexical")

    embed = layers.Embedding(len(vocab), cfg.embed_dim, mask_zero=True, name="embed")
    encoder = layers.Bidirectional(layers.GRU(cfg.gru_units), name="bigru")
    u = encoder(embed(q_vec(layers.Reshape((), name="q_squeeze")(q_in))))
    v = encoder(embed(e_vec(layers.Reshape((), name="e_squeeze")(e_in))))
    diff = layers.Subtract(name="diff")([u, v])
    prod = layers.Multiply(name="prod")([u, v])
    h = layers.Concatenate(name="concat")([u, v, diff, prod, lex_in])
    h = layers.Dense(cfg.dense_units, activation="relu", name="dense")(h)
    h = layers.Dropout(cfg.dropout, name="dropout")(h)
    out = layers.Dense(1, activation="sigmoid", name="answerable")(h)
    model = keras.Model([q_in, e_in, lex_in], out, name="answerability_bigru")
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def _string_inputs(questions: list[str], evidence: list[str], lex: np.ndarray) -> dict[str, Any]:
    import tensorflow as tf

    return {
        "question": tf.constant([[q] for q in questions], dtype=tf.string),
        "evidence": tf.constant([[e] for e in evidence], dtype=tf.string),
        "lexical": tf.constant(lex, dtype=tf.float32),
    }


def _inputs(pairs: list[AnswerabilityPair], lex: np.ndarray) -> dict[str, Any]:
    return _string_inputs([p.question for p in pairs], [p.evidence_text for p in pairs], lex)


def train_keras(
    pairs: dict[str, list[AnswerabilityPair]], out_dir: Path, cfg: type[KerasConfig] = KerasConfig
) -> dict[str, Any]:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(cfg.seed)
    tf.config.experimental.enable_op_determinism()
    train, val, test = pairs["train"], pairs["validation"], pairs.get("test", [])

    featurizer = LexicalFeaturizer.fit({p.evidence_record_id: p.evidence_text for p in train}.values())
    x_tr = featurizer.matrix([p.question for p in train], [p.evidence_text for p in train])
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    std[std == 0] = 1.0
    lex_tr = ((x_tr - mean) / std).astype("float32")
    lex_val = _lex(featurizer, val, mean, std)

    texts = [p.question for p in train] + list({p.evidence_record_id: p.evidence_text for p in train}.values())
    model = build_model(texts, lex_tr.shape[1], cfg)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=cfg.patience, restore_best_weights=True
    )
    history = model.fit(
        _inputs(train, lex_tr),
        labels(train).astype("float32"),
        validation_data=(_inputs(val, lex_val), labels(val).astype("float32")),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=[stop],
        shuffle=True,
        verbose=2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    (out_dir / "lexical_norm.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "featurizer": featurizer.to_dict()}), encoding="utf-8"
    )
    p_val = model.predict(_inputs(val, lex_val), batch_size=512, verbose=0).ravel()
    model_version = f"answerability-keras-bigru@{_sha12(out_dir / 'model.keras')}"
    thr = threshold_block(labels(val), p_val, model_version)
    (out_dir / "threshold.json").write_text(
        json.dumps({k: thr[k] for k in ("threshold", "threshold_version", "threshold_rule")}, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(out_dir / "val_scores.npy", p_val)
    result: dict[str, Any] = {
        "model": "keras_bigru_siamese_plus_lexical",
        "model_version": model_version,
        "config": cfg.as_dict(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "params": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "threshold": thr,
        **eval_blocks("val", val, p_val, thr["threshold"]),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if test:
        p_test = model.predict(_inputs(test, _lex(featurizer, test, mean, std)), batch_size=512, verbose=0).ravel()
        result.update(eval_blocks("test", test, p_test, thr["threshold"]))
        np.save(out_dir / "test_scores.npy", p_test)
    return result


def _sha12(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class KerasAnswerabilityPredictor:
    """``AnswerabilityPredictor`` over the saved Keras model (offline evaluation only; needs TensorFlow)."""

    def __init__(self, directory: Path) -> None:
        import keras

        paths = [directory / n for n in ("model.keras", "threshold.json", "lexical_norm.json")]
        if not all(p.is_file() for p in paths):
            raise ArtifactUnavailableError(f"keras answerability model not found in {directory}")
        self.model = keras.models.load_model(paths[0])
        thr = json.loads(paths[1].read_text(encoding="utf-8"))
        norm = json.loads(paths[2].read_text(encoding="utf-8"))
        self.featurizer = LexicalFeaturizer(norm["featurizer"]["idf"], norm["featurizer"]["default_idf"])
        self.mean, self.std = np.asarray(norm["mean"]), np.asarray(norm["std"])
        self.threshold = float(thr["threshold"])
        self.threshold_version = str(thr["threshold_version"])
        self.model_version = f"answerability-keras-bigru@{_sha12(paths[0])}"

    def predict(self, question: str, evidence: list[str]) -> PredictorOutput:
        if not evidence:
            return PredictorOutput(
                answerability_score=0.0,
                predicted_label=False,
                threshold_version=self.threshold_version,
                model_version=self.model_version,
            )
        x = self.featurizer.matrix([question] * len(evidence), evidence)
        lex = ((x - self.mean) / self.std).astype("float32")
        inputs = _string_inputs([question] * len(evidence), list(evidence), lex)
        scores = [float(s) for s in self.model.predict(inputs, verbose=0).ravel()]
        best = max(scores)
        return PredictorOutput(
            answerability_score=best,
            predicted_label=best >= self.threshold,
            threshold_version=self.threshold_version,
            model_version=self.model_version,
            per_evidence_scores=scores,
        )
