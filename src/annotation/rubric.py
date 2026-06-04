"""Load annotation rubric from data/annotation/rubric.md."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUBRIC_PATH = _REPO_ROOT / "data" / "annotation" / "rubric.md"


def load_rubric(path: Path | None = None) -> str:
    p = path or DEFAULT_RUBRIC_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Rubric not found: {p}")
    return p.read_text(encoding="utf-8")


def build_system_prompt(rubric: str) -> str:
    return (
        "You are an expert annotator for bias research datasets.\n"
        "Follow the rubric below exactly.\n\n"
        f"{rubric.strip()}\n"
    )
