import hashlib
from pathlib import Path


def generate_document_id(file_path: Path) -> str:
    """Generate a stable SHA-256 ID from the PDF contents."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()