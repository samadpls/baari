"""Label batches for the LLM agents, and their validated results.

Each agent reads one batch (whole sessions, segments in order) and writes one JSON
line per segment saying whether the speaker had finished their turn at its end.
"""

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

Key = tuple[str, int]


class SegmentLabel(BaseModel):
    session: str
    chunk: int
    complete: bool
    confidence: float = Field(ge=0, le=1)


def sessions(index: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in index:
        out[r["session"]].append(r)
    for chunks in out.values():
        chunks.sort(key=lambda r: r["chunk"])
    return dict(out)


def write_batches(index: list[dict], out_dir: Path, max_segments: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, current, size = [], [], 0
    for sid, chunks in sorted(sessions(index).items()):
        if current and size + len(chunks) > max_segments:
            groups.append(current)
            current, size = [], 0
        current.append((sid, chunks))
        size += len(chunks)
    if current:
        groups.append(current)
    paths = []
    for i, group in enumerate(groups):
        lines = [f"# Batch {i:03d}", ""]
        for sid, chunks in group:
            lines += [f"## Session {sid}", ""]
            lines += [f"[{c['chunk']}] ({c['duration']:.1f}s) {c['text'].strip()}" for c in chunks]
            lines.append("")
        path = out_dir / f"batch-{i:03d}.md"
        path.write_text("\n".join(lines))
        paths.append(path)
    return paths


def load_results(results_dir: Path, index: list[dict]) -> dict[Key, SegmentLabel]:
    """Validate every line and require exactly one label per indexed segment."""
    labels: dict[Key, SegmentLabel] = {}
    for path in sorted(results_dir.glob("batch-*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            label = SegmentLabel.model_validate_json(line)
            key = (label.session, label.chunk)
            if key in labels:
                raise ValueError(f"{path.name}:{n}: duplicate label for {key}")
            labels[key] = label
    wanted = {(r["session"], r["chunk"]) for r in index}
    missing, extra = wanted - labels.keys(), labels.keys() - wanted
    if missing or extra:
        raise ValueError(f"{results_dir}: {len(missing)} segments unlabelled, {len(extra)} unknown labels")
    return labels
