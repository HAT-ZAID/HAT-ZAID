﻿# Mohammad Zaid Khan

B.Tech CSE '26 · Jabalpur, India · Machine Learning Engineer

Applied ML engineer working on federated learning and LLM fine-tuning, with
production ML serving at DRDO's Scientific Analysis Group. Winner at Smart India
Hackathon 2024 for problem statement SIH1649 — a DDoS protection system for the
cloud, posed by DRDO under the Ministry of Defence — from a field of 49,000+
teams nationwide.

## Projects

### [Preprint Critic](https://github.com/HAT-ZAID/Preprint-Critic) — LoRA fine-tune of Qwen3-4B

Given a paper's text, generates a structured peer review. Trained on
[Re²](https://huggingface.co/datasets/Daoze/ReviewRebuttal) — 2,000 OpenReview
submissions paired with their human reviews.

A review is only worth reading if it's grounded in the paper, so that's what the
prompt asks for. Rather than "review this paper," the model gets a checklist of
what to verify: the claims and assumptions, whether the experiments actually
support them, whether the baselines are sufficient, and what limitations weaken
the conclusion. The target is the real human review, with loss masked to the
completion alone.

Reviews expand into several rows per paper, so a naive split would put two reviews
of the same paper on both sides and quietly inflate every metric. The split is
keyed on `paper_id` instead, and audited — 0 of 172 held-out test papers appear in
train.

LoRA r=8 on a 4-bit NF4 Qwen3-4B-Instruct-2507 base, 0.41% of parameters
trainable, 63 MiB adapter. One epoch on a single A100: 350 steps, validation loss
2.204 → 2.1653, held-out test 2.1518. After training, the question was which settings to ship. Fifteen generation
configurations were compared on the same weights and the same papers: length,
ROUGE-L, embedding similarity to the human review, embedding grounding in the
paper, and a rubric-fixed LLM judge scoring usefulness out of 100.

The adapter wins on cost and closeness. It writes 374 words instead of roughly
2,000, scores ROUGE-L 0.153 against 0.086, sits closer to the human review in
embedding space (0.858 against 0.846), and returns 81 usefulness points per 1k
generated tokens against 21.2.

It loses on depth. Judged usefulness is 40.2 against 71.4 for the same prompt on
the untuned base, because the human reviews in the dataset are themselves short
(median 345 words) and the adapter learned that length faithfully. Shorter
review, thinner coverage.

Shipped configuration: LoRA r=8 adapter, temperature 0.8, top_p 0.9, top_k 40,
repetition penalty 1.15. The prompt and settings live in `comparisons/`, the
comparison scripts and results in `experiments/`.
`LoRA · QLoRA · PEFT · Hugging Face Transformers · bitsandbytes`
### [Privacy-Based Customer Churn](https://github.com/HAT-ZAID/Privacy-Based-Customer-Churn) — centralized and federated

Predicting churn normally means shipping raw customer records to one server.
This builds it twice — a conventional XGBoost pipeline, and a federated version
where no row leaves the branch it came from.

The dataset is the interesting part. The wider IBM export carries an
end-of-quarter exit survey that acts as a near-deterministic label proxy: scores
1–2 always churn, 4–5 never do. Preprocessing drops four of the five
post-outcome fields; this one slipped through and was doing most of the work.
Correcting reported accuracy from an inflated 0.969 to 0.858 is the writeup.

The federated version is built the way the algorithm allows rather than the way
neural networks usually do — boosted trees are concatenated server-side, not
weight-averaged, because averaging destroys the structure the booster depends on.
Only trees and scalar metrics cross the network.

7,043 customers, 52 features, ROC-AUC 0.916 / churn-class F1 0.728 on a held-out
stratified test set. Served over FastAPI, with a Streamlit front end.

`XGBoost · Flower (flwr) · FastAPI · Streamlit · scikit-learn`

### [The Month-End](https://github.com/HAT-ZAID/the-month-end)

A financial decision assistant that answers the question people actually ask at
the checkout: *can I pay for this now without breaking the rest of my month,
and if not, what is the smartest way to do it anyway?* It reads a balance,
recurring bills, a payday and the seller's real payment options, then returns
one answer: pay in full, pay part now, take installments, wait for a date, or
don't do it.

**The model investigates and explains; Python decides.** Every machine field -
the safe amount, the status, the method, the plan, the earliest date, the
spending changes - comes from a deterministic kernel (facts, forecast, options,
solver, verifier). The agent layer reads intent, calls kernel tools and writes
the explanation; it never gets to move a number.

The kernel **fails closed**: an unknown direction, a signed amount, a
non-finite reserve, an undated cash row or a missing FX rate returns a named
reason and a zero safe amount, never a larger one. Chat answers are verified
against a result from the same turn, so an invented figure cannot reach the
screen, and a clarification is passed through rather than stonewalled.

Three provider privacy classes (local, paid, free), a consent gate before any
hosted model sees account data, provider HTTP that ignores proxy environment
variables, and prompts that treat every user string as untrusted data.

717 tests, eight evaluation gates, and an honest G2: the public sample answers
were produced by the organiser's reference implementation, and the gap is
documented rather than fitted.

`Python - pandas - Textual - SQLite - LangChain-compatible providers - pytest`

## Stack

Python · TensorFlow · Keras · scikit-learn · XGBoost · QLoRA / LoRA / PEFT ·
Hugging Face Transformers · Flower · FastAPI · Streamlit · SQL · AWS

## Certifications

AWS Certified AI Practitioner (2026) · AWS Certified Cloud Practitioner (2026) ·
NPTEL Machine Learning for Engineering Applications, IIT Madras (2025)

