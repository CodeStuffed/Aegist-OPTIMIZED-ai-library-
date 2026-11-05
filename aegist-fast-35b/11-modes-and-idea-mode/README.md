# Step 11: Modes and idea mode

Gain: a feature, not a speed change: idea mode uses made-up ideas on purpose and labels them, instead of hiding them behind unsure marks.

Builds on step 05: idea mode switches the probe's marks off, for a measured reason.

Previous: [step 10, Shield against hidden text](../10-shield/). Next: [step 12, Runtime governor](../12-runtime-governor/). Main README: [../README.md](../README.md).

The persona has three modes, picked with `/mode normal|blunt|idea` or `--mode`. Blunt and idea change
style only, and the hard limits apply in every mode.

Idea mode puts hallucination to work. It samples a few wild ideas at a high temperature, shown as
speculation, then writes a sober review that names what is new, what is likely false, and a cheap test for
each. Last, it develops the best idea, with guesses labelled.

Idea mode has no unsure marks. With made-up ideas in the context, the probe flagged 14 of 15 review
sentences in a measured reply, including correct critiques and plain instructions: it had learned that
sentences about made-up things are made up. The "speculation, unchecked" labels mark what is invented
instead. Idea mode also has no tools.

Code: [`aegist_rs/src/persona.rs`](../aegist_rs/src/persona.rs), [`aegist_rs/src/chat.rs`](../aegist_rs/src/chat.rs),
[`aegist/persona.py`](../aegist/persona.py).
