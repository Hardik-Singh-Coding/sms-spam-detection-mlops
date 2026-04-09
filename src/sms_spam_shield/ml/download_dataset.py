from __future__ import annotations

import io
import ssl
import zipfile
from pathlib import Path
from urllib.request import urlopen

import certifi

UCI_SMS_SPAM_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    target_path = data_dir / "SMSSpamCollection"
    if target_path.exists():
        print(f"Dataset already exists at: {target_path}")
        return

    print(f"Downloading dataset from: {UCI_SMS_SPAM_URL}")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urlopen(UCI_SMS_SPAM_URL, context=ctx) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(data_dir)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Expected {target_path} after execution, but it was not found."
        )

    print(f"Saved dataset to: {target_path}")


if __name__ == "__main__":
    main()
