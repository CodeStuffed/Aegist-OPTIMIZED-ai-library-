# Step 08: Shorter answers, less thinking

Gain: 1.25× faster on the 177B alone (1,738 s to 1,388 s), with 53% fewer answer tokens (4,893 to 2,323) and 98% fewer thinking tokens (937 to 23).

Builds on step 07: shorter replies make both brains faster.

Previous: [step 07, Two-brain routing](../07-two-brain-routing/). Next: [step 09, Test-and-fix loop](../09-test-and-fix-loop/). Main README: [../README.md](../README.md).

Normal mode asks for answers as short as the question allows: no restated question, no closing summary,
no headings it doesn't need. Short questions with only a weak cue ("how many", "why does", "compare",
"code") are answered without thinking. The thinking budget is 256 + 384 per cue + length/4, at most 1,536
tokens. Before, it was 512 + 512 per cue + length/2, at most 4,096.

On the same 16 prompts as step 07, this brain alone took 265 s and wrote 4,978 answer tokens and 5,169
thinking tokens before the change. Spider legs took 614 thinking tokens before and none after. The
rainbow answer went from 668 tokens to 227, the mortgage answer from 237 to 87.

Code: [`aegist_rs/src/chat.rs`](../aegist_rs/src/chat.rs), [`aegist_rs/src/persona.rs`](../aegist_rs/src/persona.rs).
