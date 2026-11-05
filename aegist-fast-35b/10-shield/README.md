# Step 10: Shield against hidden text

Gain: a safety change, not a speed one: hidden or encoded instructions reach the brain only as quoted, flagged text.

Builds on step 02: the engine runs the scan on each message before prefill.

Previous: [step 09, Test-and-fix loop](../09-test-and-fix-loop/). Next: [step 11, Modes and idea mode](../11-modes-and-idea-mode/). Main README: [../README.md](../README.md).

Text can carry instructions a person reading it never sees. The shield scans every user message, removes
what a reader can't see, and reports what it found, including any message that was hidden. It covers:

| kind | example |
|---|---|
| invisible characters | zero-width spaces and joiners, BOM, soft hyphen, Hangul and Braille fillers |
| Unicode tag characters | the U+E0000 block, which can spell a hidden ASCII message (decoded and shown) |
| variation-selector smuggling | bytes hidden as VS1-256 after a visible character |
| bidirectional overrides | text that displays differently from how it reads |
| look-alike words | Latin mixed with Cyrillic or Greek letters, such as "раypal" |
| encoded blobs | base64 or hex that decodes to readable text |
| forged markup and override phrases | fake chat-control tokens, "ignore all previous instructions", "developer mode" |

When it finds something, the system prompt gets a security note telling the brain to treat that content as
untrusted text and to tell the user briefly what was hidden. The attempt itself reaches the brain only as
quoted user content, so Aegist's rules stay in force.

Code: [`aegist/shield.py`](../aegist/shield.py) (the reference, with the full list), [`aegist_rs/src/shield.rs`](../aegist_rs/src/shield.rs),
[`aegist_rs/src/persona.rs`](../aegist_rs/src/persona.rs) (the security note).
