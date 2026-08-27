# Context-Aware Email Intelligence — Offline IDE Structure

This document reorganizes the Kaggle notebook code into a logical directory structure for offline IDEs (like VS Code), as requested. The original code and information have been preserved exactly [cite: 1].

## Recommended Directory Structure

```text
Email_Intelligence_Project/
├── README.md
├── setup_environment.sh
├── src/
│   ├── config.py
│   ├── 01_data_exploration.py
│   ├── 02_nlp_baseline.py
│   ├── 03_intent_training.py
│   ├── 04_summarization_training.py
│   └── 05_pipeline_demo.py
└── utils/
    └── package_outputs.py
```

---

## File Contents

### `README.md`
Consolidates the markdown documentation cells from the notebook [cite: 1].
```markdown
# Context-Aware Email Intelligence — Kaggle Training Notebook
**Project:** Samsung Collaborative Research  
**Tasks covered:**
- Data verification & exploration (Roadmap Phase 5)
- NLP baseline: NER + LDA (Phase 9)
- Intent classifier fine-tuning — DistilBERT (Phase 10)
- Summarization fine-tuning — T5-Small via EMAILSUM (Phase 11)
- Model evaluation — F1, ROUGE (Phase 12)

**Data:** Custom Kaggle dataset (processed Enron, SpamAssassin, intent splits)  
**Summarization data:** Hugging Face `Yale-LILY/emailsum` (loaded at runtime)

## Phase 9 — NLP Pipeline Baseline
Test spaCy NER and Gensim LDA on sample emails before model training.

## Phase 10 — Intent Classifier (DistilBERT)
Fine-tune `distilbert-base-uncased` on your pre-split intent CSVs.

## Phase 11 — Summarization (T5-Small + EMAILSUM)
Summarization CSVs are not in Processed_Data yet.  
We load **Yale-LILY/emailsum** directly from Hugging Face (requires Internet ON).

## Evaluation Results (fill in after running)

| Component | Metric | Score | Target |
|-----------|--------|-------|--------|
| Summarization | ROUGE-1 | ? | > 0.30 |
| Summarization | ROUGE-2 | ? | > 0.10 |
| Summarization | ROUGE-L | ? | > 0.25 |
| Intent Classifier | Accuracy | ? | > 70% |
| Intent Classifier | Macro F1 | ? | — |
| NER (spaCy sm) | Qualitative | Good/Fair/Poor | — |
```

### `setup_environment.sh`
Contains the environment setup and dependency installation commands (extracted from Cells 1, 9, 10) [cite: 1].
```bash
# Install packages not pre-installed on Kaggle (most are already available)
pip install -q transformers datasets accelerate evaluate rouge-score sentencepiece

# Install spaCy model
python -m spacy download en_core_web_sm

# Install gensim
pip install -q gensim
```

### `src/config.py`
Contains configuration, imports, and path definitions (from Cell 2) [cite: 1]. 
*(Note: You will need to update `INPUT_ROOT` to your local offline path).*
```python
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from rouge_score import rouge_scorer

# ── Paths ──────────────────────────────────────────────────────────
# UPDATE this slug after you add your dataset to the notebook
INPUT_ROOT = Path("/kaggle/input/email-intelligence-processed-data/processed")
TRAIN_DIR  = INPUT_ROOT / "train"
VAL_DIR    = INPUT_ROOT / "val"
TEST_DIR   = INPUT_ROOT / "test"
CLEAN_DIR  = INPUT_ROOT / "Cleaned data"

WORK_DIR   = Path("/kaggle/working")
MODEL_DIR  = WORK_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Intent label mapping (matches roadmap Phase 7)
INTENT_LABELS = {
    0: "REQUEST",
    1: "FOLLOW_UP",
    2: "INFORMATION",
    3: "ACKNOWLEDGEMENT",
    4: "COMPLAINT",
    5: "INVITATION",
}
NUM_INTENT_LABELS = len(INTENT_LABELS)

# Training hyperparameters (from roadmap Phases 10 & 11)
INTENT_MODEL_NAME = "distilbert-base-uncased"
SUMMARIZATION_MODEL_NAME = "t5-small"
MAX_SEQ_LENGTH = 512
MAX_SUMMARY_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Input root exists: {INPUT_ROOT.exists()}")
# !ls /kaggle/input/  # Commented out for standard python script execution
```

### `src/01_data_exploration.py`
Data verification and basic exploration (Cells 3-7) [cite: 1].
```python
from config import *

# Cell 3 — Verify dataset files exist
expected_files = [
    CLEAN_DIR / "enron_cleaned.csv",
    CLEAN_DIR / "spamassassin_cleaned.csv",
    CLEAN_DIR / "intent_labeled.csv",
    TRAIN_DIR / "intent_train.csv",
    VAL_DIR   / "intent_val.csv",
    TEST_DIR  / "intent_test.csv",
    TRAIN_DIR / "enron_train.csv",
    TRAIN_DIR / "spamassassin_train.csv",
]

for f in expected_files:
    status = "✓" if f.exists() else "✗ MISSING"
    size_mb = f.stat().st_size / 1e6 if f.exists() else 0
    print(f"{status}  {f.name}  ({size_mb:.1f} MB)")

# Cell 4 — Load all processed CSVs
# Full cleaned datasets
enron_full   = pd.read_csv(CLEAN_DIR / "enron_cleaned.csv")
spam_full    = pd.read_csv(CLEAN_DIR / "spamassassin_cleaned.csv")
intent_full  = pd.read_csv(CLEAN_DIR / "intent_labeled.csv")

# Train / val / test splits
intent_train = pd.read_csv(TRAIN_DIR / "intent_train.csv")
intent_val   = pd.read_csv(VAL_DIR   / "intent_val.csv")
intent_test  = pd.read_csv(TEST_DIR  / "intent_test.csv")

enron_train  = pd.read_csv(TRAIN_DIR / "enron_train.csv")
spam_train   = pd.read_csv(TRAIN_DIR / "spamassassin_train.csv")

print("=== Row counts ===")
for name, df in [
    ("enron_full", enron_full), ("spam_full", spam_full),
    ("intent_full", intent_full),
    ("intent_train", intent_train), ("intent_val", intent_val),
    ("intent_test", intent_test),
]:
    print(f"  {name}: {len(df):,} rows  |  cols: {list(df.columns)}")

# Cell 5 — Explore Enron cleaned data (Phase 5.2)
enron_full["body_word_count"] = enron_full["cleaned_body"].astype(str).str.split().str.len()

print(f"Total Enron emails: {len(enron_full):,}")
print(f"Avg body length (words): {enron_full['body_word_count'].mean():.1f}")
print(f"Min / Max words: {enron_full['body_word_count'].min()} / {enron_full['body_word_count'].max()}")

sample = enron_full.iloc[0]
print(f"\n--- Sample email ---")
print(f"File: {sample['file']}")
print(f"Body (first 500 chars):\n{str(sample['cleaned_body'])[:500]}")

# Cell 6 — Explore SpamAssassin data (Phase 5.3)
print("Spam vs Ham distribution:")
print(spam_full["label"].value_counts().rename({0: "ham", 1: "spam"}))

spam_sample = spam_full[spam_full["label"] == 1].iloc[0]
ham_sample  = spam_full[spam_full["label"] == 0].iloc[0]

print("\n--- SPAM sample ---")
print(str(spam_sample["cleaned_body"])[:400])
print("\n--- HAM sample ---")
print(str(ham_sample["cleaned_body"])[:400])

# Cell 7 — Explore intent-labeled data (Phase 7.4)
print("Intent label distribution (full labeled set):")
dist = intent_full["intent_label"].value_counts().sort_index()
for label_id, count in dist.items():
    print(f"  {label_id} ({INTENT_LABELS[label_id]}): {count}")

split_total = len(intent_train) + len(intent_val) + len(intent_test)
print(f"\nSplit check: train({len(intent_train)}) + val({len(intent_val)}) + test({len(intent_test)}) = {split_total}")
print(f"Matches intent_full ({len(intent_full)}): {split_total == len(intent_full)}")

plt.figure(figsize=(10, 4))
labels_named = [INTENT_LABELS[i] for i in dist.index]
plt.bar(labels_named, dist.values, color="steelblue")
plt.title("Intent Label Distribution")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

### `src/02_nlp_baseline.py`
Baseline models for NER and LDA (Cells 9-10) [cite: 1].
```python
from config import *
from src.01_data_exploration import intent_test, enron_train

# Cell 9 — Install spaCy model & run NER baseline (Phase 9.3)
import spacy
nlp = spacy.load("en_core_web_sm")

ENTITY_TYPES = ["PERSON", "ORG", "DATE", "GPE", "MONEY", "TIME", "EVENT"]

def extract_entities(text):
    doc = nlp(str(text)[:100000])  # spaCy limit safety
    entities = {t: [] for t in ENTITY_TYPES}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return {k: list(set(v)) for k, v in entities.items() if v}

for i, row in intent_test.head(3).iterrows():
    print(f"\n{'='*60}")
    print(f"Subject: {row['subject']}")
    print(f"Intent: {INTENT_LABELS[row['intent_label']]}")
    ents = extract_entities(row["cleaned_body"])
    for etype, names in ents.items():
        print(f"  {etype}: {names}")

# Cell 10 — LDA topic modeling baseline (Phase 9.4)
from gensim import corpora, models
import re

def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]

sample_texts = enron_train["cleaned_body"].astype(str).head(500).tolist()
tokenized = [simple_tokenize(t) for t in sample_texts]

dictionary = corpora.Dictionary(tokenized)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]

lda_model = models.LdaModel(
    corpus, num_topics=3, id2word=dictionary,
    passes=10, random_state=42
)

print("Top 5 words per topic:")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"  Topic {idx}: {topic}")
```

### `src/03_intent_training.py`
Intent classification training script (Cells 12-15) [cite: 1].
```python
from config import *
from src.01_data_exploration import intent_train, intent_val, intent_test

# Cell 12 — Prepare intent datasets for Hugging Face Trainer
intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)

def tokenize_intent(examples):
    return intent_tokenizer(
        examples["cleaned_body"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

from datasets import Dataset as HFDataset

def df_to_hf(df):
    return HFDataset.from_pandas(
        df[["cleaned_body", "intent_label"]].rename(columns={"intent_label": "labels"})
    )

hf_train = df_to_hf(intent_train)
hf_val   = df_to_hf(intent_val)
hf_test  = df_to_hf(intent_test)

hf_train = hf_train.map(tokenize_intent, batched=True)
hf_val   = hf_val.map(tokenize_intent, batched=True)
hf_test  = hf_test.map(tokenize_intent, batched=True)

print(hf_train)
print(hf_train[0]["labels"], "→", INTENT_LABELS[hf_train[0]["labels"]])

# Cell 13 — Define intent metrics & load model
intent_model = AutoModelForSequenceClassification.from_pretrained(
    INTENT_MODEL_NAME,
    num_labels=NUM_INTENT_LABELS,
)

def compute_intent_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

# Cell 14 — Train intent classifier (Phase 10.6)
intent_output_dir = MODEL_DIR / "intent_classifier"

intent_args = TrainingArguments(
    output_dir=str(intent_output_dir),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

intent_trainer = Trainer(
    model=intent_model,
    args=intent_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_intent_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

intent_trainer.train()
intent_trainer.save_model(str(intent_output_dir / "best"))
intent_tokenizer.save_pretrained(str(intent_output_dir / "best"))

print("Intent model saved to:", intent_output_dir / "best")

# Cell 15 — Evaluate intent classifier on test set (Phase 12.3)
from transformers import pipeline

intent_pipe = pipeline(
    "text-classification",
    model=str(intent_output_dir / "best"),
    tokenizer=str(intent_output_dir / "best"),
    device=0 if torch.cuda.is_available() else -1,
)

test_preds = []
for text in intent_test["cleaned_body"].astype(str).tolist():
    result = intent_pipe(text[:MAX_SEQ_LENGTH * 4])[0]  # truncate long emails
    test_preds.append(int(result["label"].split("_")[-1]))

y_true = intent_test["intent_label"].tolist()
y_pred = test_preds

print(classification_report(
    y_true, y_pred,
    target_names=[INTENT_LABELS[i] for i in range(NUM_INTENT_LABELS)]
))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=[INTENT_LABELS[i] for i in range(NUM_INTENT_LABELS)],
            yticklabels=[INTENT_LABELS[i] for i in range(NUM_INTENT_LABELS)],
            cmap="Blues")
plt.title("Intent Classifier — Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()
```

### `src/04_summarization_training.py`
Summarization model training using EMAILSUM (Cells 17-22) [cite: 1].
```python
from config import *

# Cell 17 — Load EMAILSUM from Hugging Face (Phase 8.2)
emailsum = load_dataset("Yale-LILY/emailsum")
print(emailsum)
print("\nFirst training example keys:", emailsum["train"][0].keys())

print("\nSample record:")
sample = emailsum["train"][0]
for k, v in sample.items():
    preview = str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
    print(f"  {k}: {preview}")

# Cell 18 — Prepare summarization data
INPUT_COL  = "email_body"       
TARGET_COL = "short_summary"    

def preprocess_summarization(example):
    inp  = str(example[INPUT_COL]).strip()
    tgt  = str(example[TARGET_COL]).strip()
    example["input_text"]  = "summarize: " + inp
    example["target_text"] = tgt
    return example

sum_train = emailsum["train"].map(preprocess_summarization)
sum_val   = emailsum["validation"].map(preprocess_summarization) \
            if "validation" in emailsum else emailsum["val"].map(preprocess_summarization)
sum_test  = emailsum["test"].map(preprocess_summarization)

def quality_filter(ex):
    inp_words = len(ex["input_text"].split())
    tgt_words = len(ex["target_text"].split())
    return inp_words >= 50 and tgt_words <= 150 and len(ex["target_text"]) > 0

sum_train = sum_train.filter(quality_filter)
sum_val   = sum_val.filter(quality_filter)
sum_test  = sum_test.filter(quality_filter)

print(f"Filtered — train: {len(sum_train)}, val: {len(sum_val)}, test: {len(sum_test)}")

# Cell 19 — Tokenize summarization data
sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)

def tokenize_summarization(examples):
    model_inputs = sum_tokenizer(
        examples["input_text"],
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels = sum_tokenizer(
        examples["target_text"],
        max_length=MAX_SUMMARY_LENGTH,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = [
        [(l if l != sum_tokenizer.pad_token_id else -100) for l in seq]
        for seq in labels["input_ids"]
    ]
    return model_inputs

sum_train_tok = sum_train.map(tokenize_summarization, batched=True, remove_columns=sum_train.column_names)
sum_val_tok   = sum_val.map(tokenize_summarization, batched=True, remove_columns=sum_val.column_names)

# Cell 20 — Train summarization model (Phase 11.5)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
sum_output_dir = MODEL_DIR / "summarization"

sum_args = TrainingArguments(
    output_dir=str(sum_output_dir),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,   # reduce to 4 if OOM
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    predict_with_generate=True,
    generation_max_length=MAX_SUMMARY_LENGTH,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(sum_tokenizer, model=sum_model)

sum_trainer = Trainer(
    model=sum_model,
    args=sum_args,
    train_dataset=sum_train_tok,
    eval_dataset=sum_val_tok,
    data_collator=data_collator,
)

sum_trainer.train()
sum_trainer.save_model(str(sum_output_dir / "best"))
sum_tokenizer.save_pretrained(str(sum_output_dir / "best"))

print("Summarization model saved.")

# Cell 21 — Quick summarization demo (Phase 11.5 visual check)
from transformers import pipeline as hf_pipeline

summarizer = hf_pipeline(
    "summarization",
    model=str(sum_output_dir / "best"),
    tokenizer=str(sum_output_dir / "best"),
    device=0 if torch.cuda.is_available() else -1,
)

for i in range(3):
    email_text = sum_test[i][INPUT_COL]
    reference  = sum_test[i][TARGET_COL]
    generated  = summarizer("summarize: " + email_text,
                           max_length=128, min_length=30,
                           do_sample=False)[0]["summary_text"]
    print(f"\n{'='*60}")
    print(f"REFERENCE:\n{reference}")
    print(f"\nGENERATED:\n{generated}")

# Cell 22 — ROUGE evaluation on test set (Phase 12.2)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

eval_subset = sum_test.select(range(min(100, len(sum_test))))

for ex in eval_subset:
    generated = summarizer(
        "summarize: " + ex[INPUT_COL],
        max_length=128, min_length=20, do_sample=False
    )[0]["summary_text"]
    scores = scorer.score(ex[TARGET_COL], generated)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

print("=== Summarization ROUGE (100 test samples) ===")
print(f"ROUGE-1 F1: {np.mean(rouge1_scores):.4f}  (target > 0.30)")
print(f"ROUGE-2 F1: {np.mean(rouge2_scores):.4f}  (target > 0.10)")
print(f"ROUGE-L F1: {np.mean(rougeL_scores):.4f}  (target > 0.25)")
```

### `src/05_pipeline_demo.py`
End-to-end pipeline demonstration (Cell 23) [cite: 1].
```python
import json
from config import *
from src.02_nlp_baseline import extract_entities
from src.03_intent_training import intent_pipe
from src.04_summarization_training import summarizer
from src.01_data_exploration import intent_test

# Cell 23 — End-to-end pipeline demo (all modules together)
def analyze_email(raw_text, subject=""):
    # 1. Intent
    intent_result = intent_pipe(raw_text[:2000])[0]
    intent_id = int(intent_result["label"].split("_")[-1])
    intent_name = INTENT_LABELS[intent_id]

    # 2. Summary
    summary = summarizer(
        "summarize: " + raw_text,
        max_length=128, min_length=20, do_sample=False
    )[0]["summary_text"]

    # 3. Entities
    entities = extract_entities(raw_text)

    return {
        "subject": subject,
        "intent": intent_name,
        "intent_confidence": round(intent_result["score"], 3),
        "summary": summary,
        "entities": entities,
    }

# Test on one intent test email
test_row = intent_test.iloc[5]
result = analyze_email(test_row["cleaned_body"], test_row["subject"])
print(json.dumps(result, indent=2))
```

### `utils/package_outputs.py`
Utility script for packaging trained models (Cell 24) [cite: 1].
```python
import shutil
from config import WORK_DIR, MODEL_DIR

# Cell 24 — Save & package outputs for download
zip_path = WORK_DIR / "trained_models.zip"
shutil.make_archive(str(WORK_DIR / "trained_models"), "zip", MODEL_DIR)

print(f"Models zipped: {zip_path}")
print(f"Size: {zip_path.stat().st_size / 1e6:.1f} MB")
print("\nModels successfully packaged in the working directory.")
```
