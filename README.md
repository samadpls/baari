<p align="center">
  <img alt="baari: knows when an Urdu speaker has finished talking" src="https://raw.githubusercontent.com/samadpls/baari/main/assets/hero.svg" width="100%">
</p>

<p align="center">
  <a href="https://samadpls.com"><img alt="by samadpls.com" src="https://img.shields.io/badge/by-samadpls.com-0E6B4F"></a>
  <a href="https://github.com/samadpls/baari/blob/main/LICENSE"><img alt="Code license: BSD-2-Clause" src="https://img.shields.io/badge/code-BSD--2--Clause-0E6B4F"></a>
  <a href="https://huggingface.co/datasets/ai4bharat/IndicVoices"><img alt="Data: IndicVoices, CC BY 4.0" src="https://img.shields.io/badge/data-IndicVoices%20(CC%20BY%204.0)-F2B544?logo=huggingface&logoColor=000"></a>
  <a href="https://pypi.org/project/baari/"><img alt="PyPI" src="https://img.shields.io/pypi/v/baari?color=0E6B4F&label=pypi"></a>
  <a href="https://pepy.tech/projects/baari"><img alt="Downloads" src="https://img.shields.io/pepy/dt/baari?label=downloads&color=0E6B4F"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=fff">
</p>

<p align="center">
  <b>End-of-turn detection for Urdu voice agents.</b><br>
  At every pause, it decides whether the speaker has finished (reply now) or is only pausing mid-sentence (keep listening).
</p>

---

baari (باری) means "turn" in Urdu. The table below shows that today's turn detectors often cut Urdu
speakers off mid-sentence, or wait after they have finished. baari is a small audio model trained to
get this right. It follows the [Smart Turn](https://github.com/pipecat-ai/smart-turn) recipe and takes
the same input and output, so it can stand in for Smart Turn.

**Status: work in progress.** Done so far: the Urdu test set, the scores of existing models, labels
for 61,251 training segments, and the `baari` package on PyPI. The first model is training now.

## 🎧 Do existing models understand Urdu?

1,588 clips of real Urdu phone conversations, cut from the official held-out split of IndicVoices: 1,470
where the speaker had finished and 118 where they were pausing mid-sentence. Because most clips are
"finished", a model that always says "finished" gets 92.6% plain accuracy, so the fair score is the
average of the two middle columns (balanced accuracy).

| Model | Balanced accuracy (95% CI) | Waits when unfinished | Replies when finished |
|---|---|---|---|
| Always says "finished" (reference) | 50.0% | 0.0% | 100.0% |
| Smart Turn v3.2 (CPU, int8) | 63.4% (58.8–68.1) | 57.6% | 69.2% |
| Smart Turn v3.2 (GPU, fp32) | 58.8% (54.8–63.0) | 76.3% | 41.3% |
| Indic Smart Turn tiny (int8) | 73.4% (69.0–77.8) | 61.0% | 85.9% |
| Indic Smart Turn tiny (fp32) | 70.0% (65.8–74.4) | 51.7% | 88.4% |
| Indic Smart Turn base (fp32) | 70.9% (66.5–75.4) | 47.5% | 94.4% |
| PuristanLabs1/urdu-turn-v2 (Urdu) | 51.8% (48.4–55.4) | 16.9% | 86.7% |

Scored on a MacBook Pro (M3 Pro), with the same audio preparation Pipecat uses. int8 models are
sensitive: their scores shift by a few points between Arm and x86 CPUs, and even tiny changes in how
the audio is prepared flip about 1 answer in 100. fp32 results stay the same.

<details>
<summary>How the test set was made</summary>

- **Source:** the conversation recordings in the `valid` split of
  [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) Urdu (CC BY 4.0): 1,654 segments,
  81 calls, 80 speakers, 1.7 hours. Each segment ends at a pause.
- **Labels:** two LLM agents each read every call's transcript in order and marked whether the speaker
  had finished at the end of each segment ([prompt](https://github.com/samadpls/baari/blob/main/prompts/label.md)). They agreed on 96.0%. Agreed
  segments form the test set; the 66 others are kept apart as `test_ambiguous`.
- **Clips:** the last 8 s of 16 kHz audio ending at the segment, with the speaker's preceding segments
  as context, the same recipe as [Indic Smart Turn](https://github.com/adimyth/indic-smart-turn).
- **Checked by ear:** a native speaker listened to 50 random clips. All 25 "finished" labels were
  right, and 19 of 25 "unfinished" labels were. In the wrong ones the words stop mid-sentence but the
  voice sounds finished, which a transcript cannot show. So roughly 1 in 4 "unfinished" clips may
  really be finished, and the "waits when unfinished" column is a rough measure.
- **Leakage:** 70 of the 80 test speakers, and the other side of 57 of the 81 test calls, also appear in
  the IndicVoices `train` split. Any training data will leave out those speakers and calls.

</details>

## Use it

```bash
pip install baari
```

```python
from baari import TurnDetector

detector = TurnDetector(threads=4)  # downloads baari's 9 MB int8 model once
detector.is_finished(audio)  # 16 kHz mono audio (float or int16) that ends at the pause
detector.probability(audio)  # the same decision as a number from 0 to 1
```

It needs only numpy and onnxruntime. One call takes about 12 ms on 4 CPU threads and 37 ms on one
(MacBook Pro, M3 Pro). Use `threads=1` on a server that handles many calls at once.

In [Pipecat](https://github.com/pipecat-ai/pipecat), baari replaces Smart Turn with one line, because it
takes the same input and gives the same output:

```python
from baari import model_path
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

turn_analyzer = LocalSmartTurnAnalyzerV3(smart_turn_model_path=model_path())
```

`TurnDetector(path=...)` runs a local ONNX file instead of downloading one.

## 🛠️ Run it

```bash
uv sync --extra pipeline
echo "HF_TOKEN=..." > .env          # accept the IndicVoices terms on Hugging Face first
uv run baari scan --split valid     # index the conversation segments (text only)
uv run baari batches                # label batches for the LLM agents
# the agents write data/labels/valid/results/{a,b}/batch-*.jsonl
uv run baari build                  # cut the 8 s clips
uv run baari evaluate               # score the existing models
```

<details>
<summary>Training</summary>

The same steps on the IndicVoices `train` split, with one labeler. Test speakers and both sides of
test calls are left out, and about 5% of the remaining speakers form a dev split that picks the best
epoch.

```bash
uv run baari scan --split train
uv run baari batches --split train
# the agents write data/labels/train/results/a/batch-*.jsonl
uv run baari build --split train --labelers a --delete-shards
uv run baari train                  # fine-tune Indic Smart Turn tiny, export fp32 and int8 ONNX
```

Training starts from Indic Smart Turn tiny. That model ships only as ONNX, so `baari train` first
reads its weights back into PyTorch. The [kaggle](https://github.com/samadpls/baari/tree/main/kaggle) folder runs the clip build and the
training on Kaggle.

</details>

---

<p align="center">
  <a href="https://github.com/samadpls/baari/blob/main/docs/CREDITS.md">Credits</a> · <a href="https://github.com/samadpls/baari/blob/main/CONTRIBUTING.md">Contributing</a> · <a href="https://github.com/samadpls/baari/blob/main/LICENSE">BSD-2-Clause license</a><br>
  Made by <a href="https://samadpls.com">Abdul Samad</a> (abdulsamadsid1@gmail.com)
</p>
