# Contributing to baari

Thanks for helping. baari is a small project, so the process is light.

## Where help is welcome

- **Data augmentation for training.** See the issues labelled
  [good first issue](https://github.com/samadpls/baari/labels/good%20first%20issue) and
  [help wanted](https://github.com/samadpls/baari/labels/help%20wanted).
- **Bug reports and fixes.** Open an issue first if the fix is large, so we can agree on the approach.
- **Ideas.** If you know Urdu speech, voice agents or turn-taking, ideas for better data or labels are
  very welcome.

## Set up

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and every dependency into `.venv`.

```bash
git clone https://github.com/samadpls/baari
cd baari
uv sync --extra pipeline
uv run pytest
```

The tests need no downloads. To build clips you also need a Hugging Face token that can read
[IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices): accept its terms on Hugging Face,
then put `HF_TOKEN=...` in `.env` (see [.env.example](.env.example)).

## Before you open a pull request

- Run `uv run pytest`, `uv run ruff check` and `uv run ruff format --check`.
- Keep one change per pull request. Say in plain words what changes and how you checked it.
- Add a small test for new logic. Tests in `tests/` use tiny made-up data, never real audio.

## Data rules

- Never commit audio, clips, labels or anything under `data/`. It is gitignored. IndicVoices audio is
  licensed CC BY 4.0 and reaches the public only as a released dataset.
- Never train on test speakers or test calls. `indicvoices.without_test` removes them, so keep it in
  every training path.
- Change the test set only in a pull request that explains why, and rerun `baari evaluate`.

## License

By contributing, you agree that your work is released under the [BSD-2-Clause license](LICENSE).
