# Label Urdu segment ends: finished turn or not

Adapted from the labelling prompt of [Indic Smart Turn](https://github.com/adimyth/indic-smart-turn) (BSD-2-Clause).

You are an expert in spoken Urdu dialogue and end-of-turn detection for voice agents.

The batch file holds several sessions. Each session is ONE speaker's side of a real two-person
Urdu phone conversation, as an ordered list of transcript segments:

```
[<chunk number>] (<duration>) <transcript>
```

The segments were cut automatically at pauses, so a segment boundary is often in the MIDDLE of
a turn. The other speaker's words are not shown. Chunk numbers may have gaps (dropped segments).

For EVERY segment, decide what is true at the END of that segment:

- **complete**: the speaker finished their turn and yielded the floor. The other person could
  start a full reply now without interrupting. Typical: a finished sentence with a finite verb
  (ہے، ہیں، تھا، گا، کریں), a direct question, a short answer or acknowledgement (جی، ہاں، ٹھیک ہے، اچھا),
  a closing phrase.
- **incomplete**: the speaker was mid-thought or still holding the floor. Typical: it ends on a
  conjunction or connective (اور، لیکن، تو، پھر، کیونکہ، کہ), a dependent, conditional or time
  clause (اگر، جب), a participle (کر کے، ہوئے), an open list, a self-correction, a lead-in such as
  "میں یہ کہنا چاہتا تھا کہ", trailing filler (وہ، یعنی، ام), or the NEXT segment plainly continues the
  same sentence.

Use the following segments as evidence: if segment N+1 plainly continues the sentence of N, then
N is incomplete. But the other person may have spoken between segments, so a natural,
self-contained statement is complete even if the speaker goes on about the same topic later.
Judge the transcript as spoken Urdu (mixing in English is normal) and ignore spelling errors.

## Output

Write one JSON object per line to the results file, covering every segment of every session in
the batch exactly once, in order, and nothing else:

```
{"session": "<session id>", "chunk": <chunk number>, "complete": true, "confidence": 0.8}
```

`confidence` has one decimal: how sure you are that your `complete` value is right, from 0.5 (a
guess) to 1.0 (certain). It is never below 0.5, also when `complete` is false.
