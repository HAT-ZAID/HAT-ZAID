# Mohammad Zaid Khan

B.Tech CSE '26 · Jabalpur, India · Machine Learning Engineer

Applied ML engineer working on federated learning and LLM fine-tuning, with
production ML serving at DRDO's Scientific Analysis Group. Winner at Smart India
Hackathon 2024 for problem statement SIH1649 — a DDoS protection system for the
cloud, posed by DRDO under the Ministry of Defence — from a field of 49,000+
teams nationwide.

## Projects

### Preprint Critic — QLoRA fine-tune of Qwen3-1.7B

Given a paper's text, generates a structured peer review. Trained on
ReviewRebuttal: 500 ICLR papers paired with human reviews, expanded to 1,584
training rows.

The part worth reading is the data work. Splitting on `paper_id` rather than on
rows, because reviews expand into multiple rows per paper — a row-level split
puts reviews of the same paper on both sides. Audited 50 sampled papers to
confirm zero overlap. Regex-stripping references and appendices cut input length
25.5% before tokenisation, and loss is masked to the completion only so prompt
tokens never enter the supervised signal.

4-bit NF4 base via QLoRA, 34 MB adapter, 1,372 s on a single T4,
held-out test loss 2.607.

`LoRA · QLoRA · PEFT · Hugging Face Transformers · bitsandbytes`

### Privacy-Based Customer Churn — centralized and federated

Predicting churn normally means shipping raw customer records to one server.
This builds it twice, and the second version never lets a row leave the branch
it came from.

7,043-customer dataset, 52 engineered features, 400-tree XGBoost at ROC-AUC
0.916 / churn-class F1 0.728 on a held-out stratified test set. The federated
rebuild uses Flower's `FedXgbBagging` across simulated branches — only trees and
scalar metrics cross the network, and boosted trees are concatenated rather
than weight-averaged, because you can't average trees the way you average neural
weights.

The dataset ships an end-of-quarter exit-survey field that acts as a
near-deterministic label proxy: one threshold on it alone scores 0.939 accuracy.
Finding that, and correcting reported accuracy from an inflated 0.969 to 0.858,
is the writeup.

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
