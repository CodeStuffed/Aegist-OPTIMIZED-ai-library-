# Step 05: Confidence probe

Gain: AUROC up from 0.778 (logit statistics alone) to 0.988. Sampling 4 extra answers scores 0.894 and costs 4× the work.

Builds on step 04: sentences in fact and other replies get marks; creative replies don't.

Previous: [step 04, Reading core](../04-reading-core/). Next: [step 06, Tools and second look](../06-tools-and-second-look/). Main README: [../README.md](../README.md).

A small probe reads layer 20 plus four logit statistics for each answer sentence and estimates the
chance it is wrong. Above its threshold, the sentence gets a mark such as
`[unsure: 53% chance this is wrong]`, and the reply ends with a count of marked sentences. A sentence
unlike anything in training is reported as not judged, so an unmarked sentence never passes silently.

The probe trained only on answers that can be checked: arithmetic, letter counts, reversals, capitals,
elements, authors, dates, things that don't exist, short paragraphs about real and made-up things, and
80 yes/no claims (true facts, myths and invented mechanisms). That is 1,172 graded rows, 122 of them wrong.

Measured on 229 held-out rows:

| method | AUROC |
|---|---|
| the probe | 0.988 |
| logit statistics alone | 0.778 |
| agreement of 4 extra sampled answers | 0.894 |
| the probe, within one kind of question | 0.850 |

At its threshold (0.29) it flags 11% of rows with precision 0.69 and recall 0.86. About 3 in 10 flags
land on right answers, and about 1 in 7 wrong answers goes unflagged. The percentage is an estimate.

Code: [`aegist/confidence.py`](../aegist/confidence.py) (training), [`aegist_rs/src/probe.rs`](../aegist_rs/src/probe.rs)
(run time), [`tests/test_confidence.py`](../tests/test_confidence.py).
