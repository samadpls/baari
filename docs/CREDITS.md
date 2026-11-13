# Credits

baari builds on open work by others. Thank you to everyone who made it available.

| Project | What baari uses | License |
|---|---|---|
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) by AI4Bharat | The Urdu phone conversations and transcripts behind the test set and the training data | CC BY 4.0 |
| [Indic Smart Turn](https://github.com/adimyth/indic-smart-turn) | The clip recipe, the labelling prompt that ours is adapted from, the starting weights (tiny), and its [training data](https://huggingface.co/datasets/adimyth/indic-smart-turn-data), which is mixed into training | Code BSD-2-Clause ([LICENSE.indic-smart-turn](../LICENSE.indic-smart-turn)), data CC BY 4.0 |
| [Smart Turn](https://github.com/pipecat-ai/smart-turn) by Pipecat | The model layout, the training recipe and the ONNX export settings | BSD-2-Clause ([LICENSE.pipecat](../LICENSE.pipecat)) |
| [Whisper](https://huggingface.co/openai/whisper-tiny) by OpenAI | The audio encoder that Smart Turn is built on | Apache-2.0 |
| [Bricolage Grotesque](https://fonts.google.com/specimen/Bricolage+Grotesque), [Geist](https://fonts.google.com/specimen/Geist), [Noto Nastaliq Urdu](https://fonts.google.com/noto/specimen/Noto+Nastaliq+Urdu) | The lettering in the banner and the logo | SIL Open Font License 1.1 |

The labels were written by LLM agents following [prompts/label.md](../prompts/label.md), and a native
Urdu speaker checked a sample by ear.
