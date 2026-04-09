from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sms_spam_shield.ml.data import load_sms_spam_collection
from sms_spam_shield.ml.pipeline import build_pipeline


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = repo_root()
    data_path = root / "data" / "SMSSpamCollection"
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"

    texts, labels = load_sms_spam_collection(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
