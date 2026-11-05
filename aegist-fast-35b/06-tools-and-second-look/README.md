# Step 06: Tools and second look

Gain: AQ up from 86 to 90: maths 83 to 100, letters 61 to 94. Measured on the 177B brain, which runs the same code.

Builds on step 05: a mark on a fact answer triggers the second look.

Previous: [step 05, Confidence probe](../05-confidence-probe/). Next: [step 07, Two-brain routing](../07-two-brain-routing/). Main README: [../README.md](../README.md).

The brain calls tools in Qwen's own tool-call format, up to 4 rounds a reply:

- `calculate` does exact arithmetic with fractions, so long multiplication is computed.
- `spell` gives a word letter by letter, letter counts, the word reversed, its length and digit sums.
  The model reads tokens, not letters, so it has to ask.
- `word_parts` knows about 90 English prefixes and suffixes and what they mean.
- `web_search` reads a Wikipedia intro or up to 4 arXiv papers. Results are marked as untrusted text.
  `AEGIST_WEB=off` removes it.
- `run_python` came later, in step 09.

When a fact answer gets a mark, the second look asks for a short new answer from the start that says
plainly when it doesn't know and uses a tool if one fits. With two brains loaded, the 177B brain gives
that second answer (step 07). `--no-tools` and `--no-second-look` turn these off.

On the 177B brain the AQ test went from 86 (raw) to 90 with tools and the second look. This brain
hasn't been measured separately.

Code: [`aegist_rs/src/tools.rs`](../aegist_rs/src/tools.rs), [`aegist_rs/src/chat.rs`](../aegist_rs/src/chat.rs).
