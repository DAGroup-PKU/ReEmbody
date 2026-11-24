"""Original sha256 support routine, without unrelated initialization machinery."""
import hashlib
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
