from __future__ import annotations

from pathlib import Path


def load_sms_spam_collection(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in raw:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Bad line (expected label<TAB>text): {line!r}")

        label, text = parts[0].strip().lower(), parts[1].strip()
        if label not in {"ham", "spam"}:
            raise ValueError(f"Unexpected Label {label!r} in line: {line!r}")

        texts.append(text)
        labels.append(label)

    if not texts:
        raise ValueError(f"No rows loaded from {path}")

    return texts, labels
