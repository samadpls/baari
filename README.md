# urdu-turn

End-of-turn detection for Urdu voice agents: at every pause, decide whether the speaker has finished
(reply now) or is only pausing mid-sentence (keep listening).

**Status: work in progress.** The Urdu test set and the scores of existing models are done. No model of
our own yet.

## Do existing models understand Urdu?

1,588 clips of real Urdu phone conversations, cut from the official held-out split of IndicVoices: 1,470
where the speaker had finished and 118 where they were pausing mid-sentence. Because most clips are
"finished", a model that always says "finished" gets 92.6% plain accuracy, so the fair score is the
average of the two middle columns (balanced accuracy).

| Model | Balanced accuracy (95% CI) | Waits when unfinished | Replies when finished |
|---|---|---|---|
| Always says "finished" (reference) | 50.0% | 0.0% | 100.0% |
| Smart Turn v3.2 (CPU, int8) | 63.8% (59.3–68.5) | 58.5% | 69.2% |
| Smart Turn v3.2 (GPU, fp32) | 58.8% (54.8–63.0) | 76.3% | 41.3% |
| Indic Smart Turn tiny (int8) | 74.0% (69.5–78.3) | 61.9% | 86.1% |
| Indic Smart Turn tiny (fp32) | 70.0% (65.8–74.4) | 51.7% | 88.4% |
| Indic Smart Turn base (fp32) | 70.9% (66.5–75.4) | 47.5% | 94.4% |
| PuristanLabs1/urdu-turn-v2 (Urdu) | 51.8% (48.4–55.4) | 16.9% | 86.7% |

Scored on a MacBook Pro (M3 Pro). int8 results shift by a few points between Arm and x86 CPUs; fp32
results do not.

<details>
<summary>How the test set was made</summary>

- **Source:** the conversation recordings in the `valid` split of
  [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) Urdu (CC BY 4.0): 1,654 segments,
  81 calls, 80 speakers, 1.7 hours. Each segment ends at a pause.
- **Labels:** two LLM agents each read every call's transcript in order and marked whether the speaker
  had finished at the end of each segment ([prompt](prompts/label.md)). They agreed on 96.0%. Agreed
  segments form the test set; the 66 others are kept apart as `test_ambiguous`.
- **Clips:** the last 8 s of 16 kHz audio ending at the segment, with the speaker's preceding segments
  as context, the same recipe as [Indic Smart Turn](https://github.com/adimyth/indic-smart-turn).
- **Not yet checked by ear:** labels come from transcripts only. A listening check of 50 clips is next.
- **Leakage:** 70 of the 80 test speakers, and the other side of 57 of the 81 test calls, also appear in
  the IndicVoices `train` split. Any training data will leave out those speakers and calls.

</details>

## Run it

```bash
uv sync --extra pipeline
echo "HF_TOKEN=..." > .env          # accept the IndicVoices terms on Hugging Face first
uv run urdu-turn scan --split valid  # index the conversation segments (text only)
uv run urdu-turn batches             # label batches for the LLM agents
# the agents write data/labels/valid/results/{a,b}/batch-*.jsonl
uv run urdu-turn build               # cut the 8 s clips
uv run urdu-turn evaluate            # score the existing models
```

## Credits

- [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) by AI4Bharat, CC BY 4.0.
- [Indic Smart Turn](https://github.com/adimyth/indic-smart-turn): the clip recipe and labelling prompt
  are adapted from it (BSD-2-Clause, see [LICENSE.indic-smart-turn](LICENSE.indic-smart-turn)).
- [Smart Turn](https://github.com/pipecat-ai/smart-turn) by Pipecat.

Made by [Abdul Samad](https://samadpls.com) (abdulsamadsid1@gmail.com).
