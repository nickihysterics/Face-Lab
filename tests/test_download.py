import hashlib
import io

import pytest

from scripts.download_weights import download


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.test/weights.pt"


def test_download_verified_and_no_overwrite(tmp_path, monkeypatch):
    payload = b"trusted fixture, not model weights"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(payload))
    destination = tmp_path / "detector.pt"
    unrelated = tmp_path / "detector.download"
    unrelated.write_bytes(b"keep")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download("https://example.test/weights.pt", "0" * 64, destination)
    assert not destination.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".weights-*"))
    download("https://example.test/weights.pt", hashlib.sha256(payload).hexdigest(), destination)
    assert destination.read_bytes() == payload
    with pytest.raises(FileExistsError):
        download("https://example.test/weights.pt", "0" * 64, destination)
    assert destination.read_bytes() == payload


@pytest.mark.parametrize(
    "url,checksum", [("http://example.test/x", "0" * 64), ("https://example.test/x", "bad")]
)
def test_download_rejects_invalid_source(tmp_path, url, checksum):
    with pytest.raises(ValueError):
        download(url, checksum, tmp_path / "x.pt")
