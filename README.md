# Mohammad Zaid Khan

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
keyed on `paper_id` instead, and audited — 0 of 160 held-out test papers appear in
train.

LoRA r=8 on a 4-bit NF4 Qwen3-4B-Instruct-2507 base, 0.41% of parameters
trainable, 66 MB adapter. One epoch on a single A100: validation loss 2.2226 →
2.1685, held-out test 2.1629.

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

### The Month-End — HackerRank competition

A financial decision agent, built for the HackerRank *Orchestrate* challenge: for
each purchase request, decide whether the user can buy now, wait, pay partially,
use installments, or not proceed — under a 90-day cash-flow forecast, recurring
expenses, pending payments and a minimum-balance constraint.

Fully deterministic, with **zero model calls and zero tokens**. Ranking is a
specified comparator, not a sampled one; OCR falls back deterministically when
Tesseract is missing; and the conflict-resolution hierarchy across messages,
images and events is explicit and tested rather than inferred.

`Python · pandas · pytest · Tesseract (optional)`

## Stack

Python · TensorFlow · Keras · scikit-learn · XGBoost · QLoRA / LoRA / PEFT ·
Hugging Face Transformers · Flower · FastAPI · Streamlit · SQL · AWS

## Certifications

AWS Certified AI Practitioner (2026) · AWS Certified Cloud Practitioner (2026) ·
NPTEL Machine Learning for Engineering Applications, IIT Madras (2025)
