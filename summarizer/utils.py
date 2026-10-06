"""Small I/O helpers used by the CLI and examples."""

from pathlib import Path


def read_text_file(path):
    """Read a UTF-8 text file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8")


def write_text_file(path, text):
    """Write ``text`` to ``path`` (UTF-8), creating parent folders as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)
