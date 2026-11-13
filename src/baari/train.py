"""Fine-tune a Smart Turn v3 layout model on Urdu clips and export it to ONNX.

The model layout, the batch-balanced loss and the int8 recipe follow Pipecat Smart Turn v3
(BSD-2-Clause, Copyright (c) 2024-2025, Daily; see LICENSE.pipecat), as used by Indic Smart Turn.
Training starts from released fp32 ONNX weights, which `recover` loads back into PyTorch.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn import functional as F
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from baari.evaluate import OnnxModel, last, metrics, score
from baari.indicvoices import SR


class SmartTurn(nn.Module):
    """Whisper encoder (400 frames = 8 s), attention pooling and an MLP head; returns logits."""

    def __init__(self, config: WhisperConfig | None = None, cache: Path | None = None):
        super().__init__()
        config = config or WhisperConfig.from_pretrained("openai/whisper-tiny", cache_dir=cache)
        config.max_source_positions = 400
        self.encoder = WhisperEncoder(config)
        h = config.d_model
        self.pool_attention = nn.Sequential(nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(h, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(input_features=input_features).last_hidden_state
        w = torch.softmax(self.pool_attention(hs), dim=1)
        return self.classifier((hs * w).sum(dim=1)).squeeze(-1)


class Exported(nn.Module):
    """What the ONNX file computes, as Pipecat expects it: P(turn complete), shape (batch, 1)."""

    def __init__(self, model: SmartTurn):
        super().__init__()
        self.inner = model  # Pipecat's wrapper uses this name, so weight names match theirs

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.inner(input_features)).unsqueeze(-1)


def to_onnx(model: SmartTurn, path: Path) -> None:
    torch.onnx.export(
        Exported(model).eval(),
        (torch.randn(2, 80, 800),),
        path,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={"input_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )


def recover(onnx_path: Path, model: SmartTurn) -> None:
    """Load the weights of a Smart Turn v3 layout fp32 ONNX file into `model`.

    The exporter names biases and norms but folds each Linear weight into an unnamed transposed
    MatMul constant. Each one is matched to its Linear by the bias added right after it; k_proj
    has no bias and is the unmatched MatMul that reads the same input as q_proj.
    """
    g = onnx.load(onnx_path).graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    state = {
        k.removeprefix("inner."): torch.from_numpy(v.copy())
        for k, v in consts.items()
        if k.startswith("inner.")
    }
    readers: dict[str, list[tuple[str, str | None]]] = {}
    users: dict[str, list] = {}
    for n in g.node:
        for x in n.input:
            users.setdefault(x, []).append(n)
    for n in g.node:
        if n.op_type == "MatMul" and n.input[1] in consts and not n.input[1].startswith("inner."):
            bias = [
                x
                for u in users.get(n.output[0], [])
                if u.op_type == "Add"
                for x in u.input
                if x.endswith(".bias")
            ]
            if len(bias) > 1:
                raise ValueError(f"{n.name}: several biases {bias}")
            readers.setdefault(n.input[0], []).append((n.input[1], bias[0] if bias else None))
    for pairs in readers.values():
        q = [b for _, b in pairs if b and b.endswith("q_proj.bias")]
        for weight, bias in pairs:
            if bias is None and len(q) != 1:
                raise ValueError(f"cannot place bias-free weight {weight}")
            name = (
                bias.removesuffix(".bias") + ".weight"
                if bias
                else q[0].replace("q_proj.bias", "k_proj.weight")
            )
            state[name.removeprefix("inner.")] = torch.from_numpy(consts[weight].T.copy())
    pos = "encoder.embed_positions.weight"
    if pos not in state:  # newer exporters fold the position table into an unnamed Add constant
        shape = tuple(model.encoder.embed_positions.weight.shape)
        found = [
            k
            for k, v in consts.items()
            if not k.startswith("inner.")
            and v.shape == shape
            and any(u.op_type == "Add" for u in users.get(k, []))
        ]
        if len(found) != 1:
            raise ValueError(f"cannot find the position table: {found}")
        state[pos] = torch.from_numpy(consts[found[0]].copy())
    model.load_state_dict(state, strict=True)


class Clips(torch.utils.data.Dataset):
    """WAV bytes and labels from clip parquet files.

    The audio (about 15 GB for the Urdu training clips alone) is copied into a temporary Arrow file and
    memory-mapped, so it lives in the OS page cache rather than process memory, and forked DataLoader
    workers share it. The file is unlinked at once: the mapping keeps it readable until the process ends.
    """

    SCHEMA = pa.schema([("wav", pa.binary()), ("y", pa.bool_())])

    def __init__(self, sources: list[tuple[Path, list[str]]]):
        fd, name = tempfile.mkstemp(prefix="baari-clips-", suffix=".arrow")
        os.close(fd)
        with pa.OSFile(name, "wb") as f, pa.ipc.new_file(f, self.SCHEMA) as writer:
            for path, splits in sources:
                wanted = pa.array(splits)
                for b in pq.ParquetFile(path).iter_batches(512, columns=["audio", "endpoint_bool", "split"]):
                    b = b.filter(pc.is_in(b.column("split"), value_set=wanted))
                    wav = pc.struct_field(b.column("audio"), "bytes").cast(pa.binary())
                    writer.write_batch(
                        pa.record_batch([wav, b.column("endpoint_bool").cast(pa.bool_())], schema=self.SCHEMA)
                    )
        t = pa.ipc.open_file(pa.memory_map(name)).read_all()
        os.unlink(name)
        self.wavs = t["wav"]
        self.y = t["y"].to_numpy(zero_copy_only=False).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def wav(self, i: int) -> np.ndarray:
        x, sr = sf.read(io.BytesIO(self.wavs[i].as_py()), dtype="float32")
        if sr != SR:
            raise ValueError(f"clip {i}: {sr} Hz")
        return last(x if x.ndim == 1 else x.mean(axis=1), 8)

    def __getitem__(self, i: int) -> tuple[np.ndarray, float]:
        return self.wav(i), self.y[i]


FE = WhisperFeatureExtractor(chunk_length=8)


def features(wavs: list[np.ndarray], device: str) -> torch.Tensor:
    return FE(
        wavs,
        sampling_rate=SR,
        return_tensors="pt",
        padding="max_length",
        max_length=8 * SR,
        truncation=True,
        do_normalize=True,
        device=device,
    ).input_features.to(device)


def collate(batch):
    return [w for w, _ in batch], torch.tensor([y for _, y in batch])


@torch.no_grad()
def predict(model: SmartTurn, data: Clips, device: str, batch: int = 128, workers: int = 4) -> np.ndarray:
    model.eval()
    out = []
    for wavs, _ in torch.utils.data.DataLoader(
        data, batch_size=batch, num_workers=workers, collate_fn=collate
    ):
        out.append(torch.sigmoid(model(features(wavs, device)).float()).cpu().numpy())
    return np.concatenate(out)


def memory() -> str:
    """Free memory on Linux, for the Kaggle log. MemAvailable includes page cache the kernel can drop."""
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    except OSError:
        return "memory n/a"
    gb = {k: int(info[k].split()[0]) / 1e6 for k in ("MemAvailable", "MemTotal")}
    return f"{gb['MemAvailable']:.1f} of {gb['MemTotal']:.1f} GB memory free"


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    m = metrics(p, y)
    return {k: (np.asarray(v).round(4).tolist()) for k, v in m.items()}


def export(state: dict, path: Path, cache: Path | None, check: Clips) -> None:
    """Export from a fresh model (exporting changes how the exported object computes afterwards)."""
    model = SmartTurn(cache=cache)
    model.load_state_dict(state)
    to_onnx(model, path)
    ref = SmartTurn(cache=cache)
    ref.load_state_dict(state)
    wavs = [check.wav(i) for i in range(16)]
    with torch.no_grad():
        want = torch.sigmoid(ref.eval()(features(wavs, "cpu"))).numpy()
    got = OnnxModel(str(path))(wavs)
    if np.abs(got - want).max() > 1e-3:
        raise RuntimeError(f"ONNX export differs from PyTorch by {np.abs(got - want).max():.2e}")


def quantize(fp32: Path, out: Path, calib: Clips, n: int = 1024) -> None:
    """Static int8 (QDQ, per-channel MinMax on Conv/MatMul/Gemm), calibrated on shuffled training clips."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    order = np.random.default_rng(42).permutation(len(calib))[:n]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(order)

        def get_next(self):
            i = next(self.it, None)
            return None if i is None else {"input_features": features([calib.wav(int(i))], "cpu").numpy()}

    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_optimization=False, skip_symbolic_shape=True)
    quantize_static(
        model_input=str(pre),
        model_output=str(out),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    pre.unlink()


def run(
    urdu: Path,
    test: Path,
    init: Path,
    out: Path,
    extra: list[Path],
    epochs: float = 3,
    lr: float = 2e-5,
    batch: int = 64,
    seed: int = 0,
    workers: int = 4,
    cache: Path | None = None,
    max_steps: int = 0,
) -> dict:
    """Train on Urdu 'train' (+ the 'train' split of each extra clip file), pick the epoch with the best
    Urdu dev balanced accuracy, export fp32 and int8 ONNX and score both on the Urdu test clips."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    train = Clips([(urdu, ["train"])] + [(p, ["train"]) for p in extra])
    dev, tst = Clips([(urdu, ["dev"])]), Clips([(test, ["test"])])
    print(f"loaded {len(train)} train clips in {time.time() - t_load:.0f}s; {memory()}", flush=True)
    log = {
        "config": {k: str(v) for k, v in locals().items() if k in run.__code__.co_varnames[:12]},
        "sizes": {
            "train": len(train),
            "train_unfinished": int((train.y == 0).sum()),
            "dev": len(dev),
            "test": len(tst),
        },
        "epochs": [],
    }
    model = SmartTurn(cache=cache)
    recover(init, model)
    model.to(device)
    log["start"] = {
        "dev": summary(predict(model, dev, device, workers=workers), dev.y),
        "test": summary(predict(model, tst, device, workers=workers), tst.y),
    }
    print(json.dumps(log["start"]), flush=True)

    per_epoch = len(train) // batch
    total = max_steps or int(epochs * per_epoch)
    warm = max(1, total // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0,
    )
    best, best_state, step, t0 = -1.0, None, 0, time.time()
    while step < total:
        model.train()
        losses = []
        for wavs, y in loader:
            x, y = features(wavs, device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            # Pipecat's balancing: weight the 'complete' class by the batch's incomplete/complete ratio.
            pos_weight = ((y == 0).sum() / (y == 1).sum().clamp(min=1)).clamp(0.1, 10.0)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(
                    f"step {step}/{total} loss {np.mean(losses[-100:]):.4f} {time.time() - t0:.0f}s; {memory()}",
                    flush=True,
                )
            if step >= total:
                break
        row = {
            "step": step,
            "loss": float(np.mean(losses)),
            "dev": summary(predict(model, dev, device, workers=workers), dev.y),
            "test": summary(predict(model, tst, device, workers=workers), tst.y),
        }
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
        if row["dev"]["balanced"] > best:
            best = row["dev"]["balanced"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log["best_step"] = step

    torch.save(best_state, out / "baari.pt")
    export(best_state, out / "baari-fp32.onnx", cache, tst)
    quantize(out / "baari-fp32.onnx", out / "baari-int8.onnx", train)
    wavs = [tst.wav(i) for i in range(len(tst))]
    for name in ("fp32", "int8"):
        log[f"onnx_{name}_test"] = summary(score(OnnxModel(str(out / f"baari-{name}.onnx")), wavs), tst.y)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "metrics.json").write_text(json.dumps(log, indent=1))
    return log
