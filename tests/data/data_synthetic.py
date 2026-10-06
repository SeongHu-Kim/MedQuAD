"""SYNTHETIC data for data-pipeline tests. No MedQuAD text is used; every row below is invented."""

from __future__ import annotations

import csv
from pathlib import Path

HEADER = ["question", "answer", "source", "focus_area"]

LONG_A = (
    "Alphaitis is a SYNTHETIC condition used only for tests. It affects about 1 in 10,000 people and does not "
    "spread between people. Treatment is not always needed, but a doctor may suggest rest and fluids."
)
LONG_B = (
    "Betaosis is a SYNTHETIC disorder for tests. Symptoms include mild fever, joint pain and tiredness that "
    "last 2 to 3 weeks. It is not inherited and has no known cure, yet most people recover fully."
)
TEMPLATE = (
    "{t} is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell "
    "carry a SYNTHETIC change. The parents each carry one copy but usually show no signs."
)
BOILER = "SYNTHETIC boilerplate: these resources address diagnosis and management."

# (question, answer, source, focus_area) - SYNTHETIC
SYNTHETIC_ROWS: list[tuple[str, str, str, str]] = [
    ("What is (are) Alphaitis ?", LONG_A, "SRC1", "Alphaitis"),
    ("What are the treatments for Alphaitis ?", "Rest  and   fluids are advised; do NOT use 5 mg doses.", "SRC1",
     "Alphaitis"),
    ("What is (are) Alphaitis ?", LONG_A, "SRC2", "alphaitis"),  # same topic, other source/case
    ("What is (are) Betaosis ?", LONG_B, "SRC1", "Betaosis"),
    ("What is (are) Betaosis ?", LONG_B, "SRC1", "Betaosis"),  # exact full-row duplicate
    ("What causes Betaosis ?", "", "SRC1", "Betaosis"),  # empty answer
    ("Is Gammaemia inherited ?", TEMPLATE.format(t="Gammaemia"), "SRC2", "Gammaemia"),
    ("Is Deltapathy inherited ?", TEMPLATE.format(t="Deltapathy"), "SRC2", "Deltapathy"),  # template near-dup
    ("Is Epsilonoma inherited ?", "Is Epsilonoma inherited?", "SRC2", "Epsilonoma"),  # question-only answer
    ("How to prevent Zetaosis ?", "Topics", "SRC3", "Zetaosis"),  # heading-only answer
    ("How many people are affected by Zetaosis ?", "Zetaosis is rare.", "SRC3", "Zetaosis"),  # valid short
    ("What is (are) Etaitis ?", BOILER, "SRC3", "Etaitis"),
    ("What is (are) Thetaitis ?", BOILER, "SRC3", "Thetaitis"),
    ("What is (are) Iotaitis ?", BOILER, "SRC3", "Iotaitis"),
    ("Do you have information about Kappa", "Kappa is a SYNTHETIC topic about safe sleep for babies.", "SRC4", ""),
]
for i in range(30):  # filler topics so every split receives groups
    SYNTHETIC_ROWS.append(
        (f"What is (are) Filler{i} ?", f"Filler{i} is SYNTHETIC filler number {i} with unique words w{i}a w{i}b.",
         f"SRC{i % 3 + 1}", f"Filler{i}")
    )


def write_csv(path: Path, rows: list[tuple[str, str, str, str]], header: list[str] = HEADER) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


