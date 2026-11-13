import json
import random
import zlib
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from baari import clips, indicvoices, labels
from baari.settings import Settings

app = typer.Typer(no_args_is_help=True)


def load_index(s: Settings, split: str) -> list[dict]:
    rows = pq.read_table(s.index_dir / f"{split}.parquet").to_pylist()
    if split == "valid":
        return rows
    return indicvoices.without_test(rows, pq.read_table(s.index_dir / "valid.parquet").to_pylist())


@app.command()
def scan(split: str = "valid") -> None:
    """Index the conversation segments of one IndicVoices Urdu split (text only, no audio)."""
    s = Settings()
    rows = [
        r
        for shard in indicvoices.shards(split, s.hf_token)
        for r in indicvoices.index_shard(shard, s.hf_token)
    ]
    s.index_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), s.index_dir / f"{split}.parquet")
    hours = sum(r["duration"] for r in rows) / 3600
    typer.echo(
        f"{split}: {len(rows)} segments, {len({r['session'] for r in rows})} sessions, "
        f"{len({r['speaker_id'] for r in rows})} speakers, {hours:.1f} h"
    )


@app.command()
def batches(split: str = "valid", max_segments: int = 350) -> None:
    """Write label batches for the LLM agents."""
    s = Settings()
    paths = labels.write_batches(load_index(s, split), s.labels_dir / split / "batches", max_segments)
    typer.echo(f"wrote {len(paths)} batches to {paths[0].parent}")


def split_of(split: str, speaker_id: str, agree: bool) -> str:
    """valid becomes the test set; any other split is cut by speaker into train and a 5% dev set."""
    if split == "valid":
        return "test" if agree else "test_ambiguous"
    return "dev" if zlib.crc32(speaker_id.encode()) % 20 == 0 else "train"


@app.command()
def build(split: str = "valid", labelers: str = "a,b", seed: int = 0, delete_shards: bool = False) -> None:
    """Cut 8 s clips for every labelled segment, one shard at a time, and write them to data/clips/<split>.parquet.

    For valid, agreeing labels go to 'test' and the rest to 'test_ambiguous'. Labeler order matters: the
    first labeler's label is the clip's label and sets the context gaps.
    """
    s = Settings()
    index = load_index(s, split)
    names = labelers.split(",")
    results = {n: labels.load_results(s.labels_dir / split / "results" / n, index) for n in names}
    sessions = labels.sessions(index)
    rows_of, need, in_shard = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in index:
        rows_of[r["shard"]].add(r["row"])
        need[r["session"]].add(r["shard"])
        in_shard[r["shard"]].add(r["session"])
    rng, audio, counts, writer = random.Random(seed), {}, defaultdict(int), None
    path = s.clips_dir / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    for n, shard in enumerate(sorted(rows_of), 1):
        got = indicvoices.shard_audio(shard, rows_of[shard], s.hf_token, s.hf_cache, delete=delete_shards)
        audio |= {(shard, i): x for i, x in got.items()}
        out = []
        # A few sessions span two shards; cut them once all their audio is in.
        for sid in [
            sid for sid in sessions if sid in in_shard[shard]
        ]:  # index order keeps the gaps reproducible
            need[sid].discard(shard)
            if need[sid]:
                continue
            chunks = sessions[sid]
            wavs = {
                c["chunk"]: audio.pop((c["shard"], c["row"]))
                for c in chunks
                if (c["shard"], c["row"]) in audio
            }
            for c, clip in zip(chunks, clips.build(chunks, wavs, results[names[0]], rng), strict=True):
                if clip is None:
                    continue
                votes = {n: results[n][(sid, c["chunk"])] for n in names}
                row = {
                    "audio": clip,
                    "endpoint_bool": votes[names[0]].complete,
                    "split": split_of(split, c["speaker_id"], len({v.complete for v in votes.values()}) == 1),
                    "session": sid,
                    "chunk": c["chunk"],
                    "speaker_id": c["speaker_id"],
                    "spoken_text": c["text"],
                    **{f"label_{n}": v.complete for n, v in votes.items()},
                    **{f"confidence_{n}": v.confidence for n, v in votes.items()},
                }
                counts[row["split"], row["endpoint_bool"]] += 1
                out.append(row)
        if out:
            table = clips.table(out)
            writer = writer or pq.ParquetWriter(path, table.schema)
            writer.write_table(table, row_group_size=500)
        typer.echo(f"[{n}/{len(rows_of)}] {shard}: {len(out)} clips", err=True)
    if writer:
        writer.close()
    typer.echo(
        f"{sum(counts.values())} clips: "
        + json.dumps({f"{k[0]}/{'finished' if k[1] else 'unfinished'}": v for k, v in sorted(counts.items())})
    )


@app.command()
def spotcheck(
    split: str = "valid",
    per_class: int = 25,
    unfinished_only: bool = False,
    skip: Path | None = None,
    name: str = "spotcheck",
    seed: int = 0,
) -> None:
    """Write a local listening page of clean test clips: a random sample of each label, or every
    unfinished clip. Clips already answered in the `skip` file are left out."""
    from importlib.resources import files

    s = Settings()
    done = set(json.loads(skip.read_text())) if skip else set()
    rows = [
        r
        for r in pq.read_table(s.clips_dir / f"{split}.parquet").to_pylist()
        if r["split"] == "test" and f"{r['session']}|{r['chunk']}" not in done
    ]
    rng = random.Random(seed)
    if unfinished_only:
        picked = [r for r in rows if not r["endpoint_bool"]]
    else:
        picked = [
            r
            for label in (False, True)
            for r in rng.sample([r for r in rows if r["endpoint_bool"] is label], per_class)
        ]
    rng.shuffle(picked)
    out = s.data_dir / name
    out.mkdir(parents=True, exist_ok=True)
    clips_meta = []
    for r in picked:
        wav = f"{r['session']}_{r['chunk']}.wav"
        (out / wav).write_bytes(r["audio"]["bytes"])
        clips_meta.append(
            {
                "id": f"{r['session']}|{r['chunk']}",
                "file": wav,
                "finished": r["endpoint_bool"],
                "text": r["spoken_text"],
            }
        )
    page = (
        files("baari")
        .joinpath("spotcheck.html")
        .read_text()
        .replace("__CLIPS__", json.dumps(clips_meta, ensure_ascii=False))
        .replace("__NAME__", name)
    )
    (out / "index.html").write_text(page)
    typer.echo(f"open {out / 'index.html'}")


@app.command()
def evaluate(split: str = "valid", subset: str = "test") -> None:
    """Score the existing turn detectors on the clips."""
    from baari import evaluate as ev

    s = Settings()
    report = ev.run(s.clips_dir / f"{split}.parquet", subset, s.hf_cache)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / f"{split}-{subset}.md").write_text(report)
    typer.echo(report)
