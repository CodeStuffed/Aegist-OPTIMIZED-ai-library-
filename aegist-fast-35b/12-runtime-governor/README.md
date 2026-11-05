# Step 12: Runtime governor

Gain: not measured as a speed or score change: it lowers the temperature when too many sentences get marked, and raises it again when few do.

Builds on step 05: it reads the probe's marks.

Previous: [step 11, Modes and idea mode](../11-modes-and-idea-mode/). Next: [step 13, Core engine and learning time](../13-core-engine/). Main README: [../README.md](../README.md).

The governor watches the brain's own output and nudges sampling:

| track | rule |
|---|---|
| answers | once 12 or more sentences are scored, cool by 0.05 a step while more than 30% (smoothed) are marked unsure, warm back when fewer than 15% are |
| fact prompts (step 04) | their own track, from 0.3 down to 0, which is greedy |
| other help | uses the answers track |
| creative prompts | no track |
| idea mode | raises the temperature when ideas come out too alike, lowers it when they fall apart |

Its state is kept in `governor.json` between sessions, and `/governor` in chat shows it.

Code: [`aegist_rs/src/governor.rs`](../aegist_rs/src/governor.rs).
