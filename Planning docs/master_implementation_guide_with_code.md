# Master Implementation Guide (Kaggle Edition, with Code)
## Context-Aware Email Intelligence: A Unified NLP Approach to Information Synthesis

**Project Type:** Samsung Collaborative Research Project
**Document Version:** 2.0 — Kaggle Development Track, Merged with Cell-Wise Code
**Scope:** Every step required to complete this project from zero to final deliverable, in exact execution order, using **Kaggle Notebooks as the primary Python/ML development environment** — with the actual code or command for each step inlined directly beneath it.

---

## How to Use This Document

This is the single master file for the project — it merges the phase-by-phase implementation roadmap with the actual notebook code that goes with each step. For any step, everything you need is right there: what to do, why, and (where applicable) the exact code cell or shell command to run.

- Steps marked **[ONE-TIME]** only need to be done once per Kaggle account or notebook.
- Steps marked **[EVERY SESSION]** must be repeated each time you open a fresh Kaggle Notebook session, because Kaggle sessions are ephemeral.
- Steps marked **[VERIFY]** include a quick sanity check you must confirm before moving on.
- Steps marked **[DECISION]** require you to make a choice that affects later steps.
- A **📎 Code for this step** block gives you the exact cell(s) to paste into Kaggle. Where a block corresponds to a specific cell number from the original cell-wise notebook plan, that's noted (e.g. *Cell 14*) so you can also follow the [Appendix C](#appendix-c--full-ordered-cell-list-for-copy-paste) quick-build order if you'd rather assemble one notebook top-to-bottom instead of jumping between phases. Cells labeled with a letter suffix (e.g. *Cell 6a*, *6b*) are preprocessing sub-steps inserted between Cell 6 and Cell 7 — paste them in that order without renumbering the surrounding cells.

> **Scope note:** All Python, NLP, data preparation, model training, and model evaluation work happens on **Kaggle Notebooks** — not on your laptop. The one exception is **Android app development (Phases 14–16)**, which requires Android Studio, an emulator, and/or a physical device, none of which Kaggle can run. That part of the project stays on your laptop. Everywhere else, "your machine" means "your Kaggle Notebook."

---

## Table of Contents

- [Phase 0 — Pre-Work: Accounts and Kaggle Readiness Check](#phase-0--pre-work-accounts-and-kaggle-readiness-check)
- [Phase 1 — Kaggle Notebook Environment Setup](#phase-1--kaggle-notebook-environment-setup)
- [Phase 2 — Android Development Environment Setup (Local Laptop Required)](#phase-2--android-development-environment-setup-local-laptop-required)
- [Phase 3 — Project Structure and Repo Sync Inside Kaggle](#phase-3--project-structure-and-repo-sync-inside-kaggle)
- [Phase 4 — Dataset Acquisition on Kaggle](#phase-4--dataset-acquisition-on-kaggle)
- [Phase 5 — Dataset Exploration and Inventory](#phase-5--dataset-exploration-and-inventory)
- [Phase 6 — Dataset Preprocessing and Curation](#phase-6--dataset-preprocessing-and-curation)
- [Phase 7 — Intent Classification Data Preparation](#phase-7--intent-classification-data-preparation)
- [Phase 8 — Summarization Data Preparation](#phase-8--summarization-data-preparation)
- [Phase 9 — NLP Pipeline Assembly and Baseline Testing](#phase-9--nlp-pipeline-assembly-and-baseline-testing)
- [Phase 10 — Model Training (Intent Classifier)](#phase-10--model-training-intent-classifier)
- [Phase 11 — Model Training (Summarization)](#phase-11--model-training-summarization)
- [Phase 12 — Model Evaluation (Kaggle-Side)](#phase-12--model-evaluation-kaggle-side)
- [Phase 13 — Model Export and Mobile Conversion](#phase-13--model-export-and-mobile-conversion)
- [Phase 14 — Android Application Setup (Local Laptop Required)](#phase-14--android-application-setup-local-laptop-required)
- [Phase 15 — Android App Module Development (Local Laptop Required)](#phase-15--android-app-module-development-local-laptop-required)
- [Phase 16 — Android App Testing and Performance Profiling (Local Laptop Required)](#phase-16--android-app-testing-and-performance-profiling-local-laptop-required)
- [Phase 17 — Usability Testing](#phase-17--usability-testing)
- [Phase 18 — Final Documentation and Deliverables](#phase-18--final-documentation-and-deliverables)
- [Appendix A — Processed Data Folder Reference](#appendix-a--processed-data-folder-reference)
- [Appendix B — Kaggle Troubleshooting](#appendix-b--kaggle-troubleshooting)
- [Appendix C — Full Ordered Cell List (for copy-paste)](#appendix-c--full-ordered-cell-list-for-copy-paste)

---

## Phase 0 — Pre-Work: Accounts and Kaggle Readiness Check

### Step 0.1 — Verify Kaggle Access Instead of Local Hardware [ONE-TIME] [VERIFY]

Because training and preprocessing run on Kaggle's servers, your laptop's RAM/GPU no longer gate the project. You only need a machine that can run a modern browser.

1. Confirm you have a stable internet connection.
2. Confirm your browser is up to date (Chrome or Firefox recommended).
3. **[VERIFY]** Go to `https://www.kaggle.com` and confirm the site loads normally.

---

### Step 0.2 — Create a GitHub Account (if you don't have one) [ONE-TIME]

1. Go to `https://github.com` → **Sign up** → enter email → password → username (e.g., `rishabh-email-nlp`) → solve the puzzle → verify email code.
2. On the welcome screen, click **Skip personalization**.
3. **[VERIFY]** You should see your username in the top-right corner on `github.com`.

---

### Step 0.3 — Create a New GitHub Repository for This Project [ONE-TIME]

1. Click green **New** → Repository name: `context-aware-email-intelligence` → Visibility: **Private** → check **Add a README file** → **Add .gitignore**: **Python** → **Create repository**.
2. Click **Code** → copy the HTTPS URL. Save it for Phase 3.
3. Profile icon → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)** → scope `repo` → **Generate token**. Copy and save it immediately — you'll paste it into a Kaggle Secret in Step 1.6.

---

### Step 0.4 — Create a Kaggle Account [ONE-TIME]

1. Go to `https://www.kaggle.com` → **Register** → **Register with Google** or email/password.
2. Profile icon → **Settings** → **Phone Verification** → verify your number. **Required to enable GPU accelerators.**
3. **[VERIFY]** Settings shows phone-verified.

---

### Step 0.5 — Create a Hugging Face Account [ONE-TIME]

1. Go to `https://huggingface.co` → **Sign Up** → verify email.
2. Profile icon → **Settings** → **Access Tokens** → **New token** → name `email-intelligence-project` → role **Read** → **Generate a token**.
3. Save the token securely — you'll paste it into a Kaggle Secret in Step 1.6.

---

### Step 0.6 — Understand Kaggle's GPU Quota [ONE-TIME]

1. Roughly **30 GPU-hours per week**, resettable weekly (verify current quota in Notebook Settings).
2. Accelerators: **GPU T4 x2** or **GPU P100**.
3. Budget: CPU-only for Phases 5–9; GPU for Phases 10–12.

---

### Step 0.7 — Plan Your Work Split (Team Coordination)

| Task | Owner |
|---|---|
| Dataset download, cleaning, preprocessing | One person |
| NER pipeline + evaluation | Shared |
| Intent classifier training | One person |
| Summarization model training | Other person |
| Model conversion to mobile | One person |
| Android app UI layout (local) | One person |
| Android inference engine (local) | Other person |
| Evaluation report writing | Shared |
| Presentation preparation | Shared |

Agree on a **fork model** for Kaggle collaboration (see Step 3.1). Document the split in a shared notes doc or a Markdown file in the repo.

---

## Phase 1 — Kaggle Notebook Environment Setup

### Step 1.1 — Create Your Project Notebook [ONE-TIME]

1. **+ Create** → **New Notebook** → rename to `email-nlp-project`.
2. Right panel → **Language: Python**.
3. **[VERIFY]** Kaggle-managed Python kernel already running.

📎 **Code for this step** *(Cell 0 — Markdown title cell; paste as the first cell of every notebook you build)*
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
**Summarization data:** Kaggle dataset `prishabhkumar/emailsum` (loaded from `/kaggle/input/datasets/prishabhkumar/emailsum/` at runtime)
```

---

### Step 1.2 — Confirm What's Already Installed [VERIFY]

```python
import numpy, pandas, sklearn, matplotlib, seaborn
print(numpy.__version__, pandas.__version__, sklearn.__version__)
```
**[VERIFY]** No `ModuleNotFoundError` — these ship pre-installed on Kaggle.

---

### Step 1.3 — Install the Remaining Libraries [EVERY SESSION]

Kaggle sessions are ephemeral — put this in the first code cell of every notebook.

📎 **Code for this step** *(Cell 1)*
```python
# Install packages not pre-installed on Kaggle (most are already available)
!pip install -q nltk spacy gensim beautifulsoup4 rouge-score kaggle
!pip install -q transformers datasets accelerate evaluate rouge-score sentencepiece
```

> **Note:** TensorFlow and PyTorch are already pre-installed and GPU-enabled — no `pip install` needed for either.
> **Recommendation:** Plan around the TFLite conversion path (Phase 13) — better documented for Android.

**[VERIFY]** `!pip list 2>/dev/null | grep -E "nltk|spacy|gensim|transformers|rouge"` shows all packages.

---

### Step 1.4 — Enable Internet Access on the Notebook [ONE-TIME PER NOTEBOOK]

1. Right panel → **Session options** → **Internet** toggle **On**.
2. Verify phone number if prompted.
3. **[VERIFY]** `!ping -c 2 google.com` returns replies.

---

### Step 1.5 — Enable the GPU Accelerator [DECISION] [EVERY SESSION]

1. Right panel → **Accelerator** → **GPU T4 x2** or **GPU P100**.
   - **None (CPU)** for Phases 5–9 (save quota).
   - **GPU** for Phases 10–12.

📎 **Code for this step**
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```
**[VERIFY]** Prints `True` and a device name (e.g., `Tesla T4`) when the accelerator is on.

---

### Step 1.6 — Store Your Tokens as Kaggle Secrets [ONE-TIME]

1. **Add-ons** → **Secrets** → **Add a new secret**.
2. Label `GITHUB_TOKEN` → value = token from Step 0.3.
3. Label `HF_TOKEN` → value = token from Step 0.5.
4. Toggle both **On** for this notebook.

📎 **Code for this step**
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
HF_TOKEN = secrets.get_secret("HF_TOKEN")
```
**[VERIFY]** `print(len(GITHUB_TOKEN), len(HF_TOKEN))` prints two nonzero numbers.

---

### Step 1.7 — Authenticate the Hugging Face CLI [EVERY SESSION]

```python
!huggingface-cli login --token $HF_TOKEN
```
**[VERIFY]** `!huggingface-cli whoami` prints your username.

---

### Step 1.8 — Download NLTK and spaCy Resources [EVERY SESSION]

```python
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

!python -m spacy download en_core_web_sm -q
```
**[VERIFY]**
```python
import spacy; nlp = spacy.load("en_core_web_sm"); print("ok")
```
> **Optional time-saver:** package downloaded NLTK/spaCy data as a private Kaggle Dataset and re-attach via **Add Data** in future sessions.

---

### Step 1.9 — Standard Project Configuration Cell [EVERY SESSION]

This is the cell you should adapt and re-run at the top of **every** training/evaluation notebook from Phase 5 onward — it sets up paths, intent labels, hyperparameter constants, and confirms the device. Update `INPUT_ROOT` to match your actual dataset slug (see Phase 4).

📎 **Code for this step** *(Cell 2 — Imports and configuration)*
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
!ls /kaggle/input/
```

---

## Phase 2 — Android Development Environment Setup (Local Laptop Required)

> **This phase cannot be done on Kaggle.** Kaggle Notebooks run in a headless Linux container with no display server, no Android SDK, no emulator support, and no USB passthrough. Android Studio, the emulator, and on-device debugging all require your laptop.

### Step 2.1 — Install Java Development Kit (JDK) [ONE-TIME, LOCAL]

1. `https://www.oracle.com/java/technologies/downloads/` → **JDK 17** tab → download installer for your OS → run with defaults.
2. **[VERIFY]** `java -version` shows `17.x.x`.

---

### Step 2.2 — Download and Install Android Studio [ONE-TIME, LOCAL]

1. `https://developer.android.com/studio` → **Download Android Studio** → accept terms → download (~1 GB) → run installer.
2. Setup wizard: **Standard** → theme → **Finish** → SDK/emulator/build tools download (10–30 min).
3. **[VERIFY]** Welcome screen appears with "new project" option.

---

### Step 2.3 — Install Additional Android SDK Components [ONE-TIME, LOCAL]

1. **Tools** → **SDK Manager** → **SDK Platforms**: check **API 34** and **API 26** → **Apply**.
2. **SDK Tools**: confirm **Build-Tools**, **Emulator**, **Platform-Tools**, **HAXM** checked → **Apply** → **OK**.

---

### Step 2.4 — Create an Android Virtual Device (Emulator) [ONE-TIME, LOCAL]

1. **Tools** → **Device Manager** → **Create Device** → **Phone** → **Pixel 6** → **Next**.
2. **Recommended** tab → **API 34** → **Next** → name `Pixel6_API34`, RAM ≥ 2048 MB → **Finish**.
3. **[VERIFY]** Emulator boots to Android home screen in 1–2 minutes.

---

### Step 2.5 — Enable Developer Mode on a Physical Android Device (Optional, LOCAL)

1. **Settings** → **About phone** → tap **Build number** 7 times.
2. **Developer options** → **On** → **USB debugging** → connect via USB → **Allow**.
3. **[VERIFY]** Device appears under **Device Manager → Physical**.

---

## Phase 3 — Project Structure and Repo Sync Inside Kaggle

### Step 3.1 — Decide Your Kaggle Collaboration Model [DECISION] [ONE-TIME]

- **Shared-notebook model:** both teammates edit the same notebook, taking turns.
- **Fork model (recommended):** each teammate keeps their own copy (**File** → **Copy and Edit**); GitHub is the source of truth.

---

### Step 3.2 — Clone Your GitHub Repository Into the Kaggle Session [EVERY SESSION]

📎 **Code for this step**
```python
import os
os.chdir("/kaggle/working")
!git clone https://{GITHUB_TOKEN}@github.com/YOUR_USERNAME/context-aware-email-intelligence.git
os.chdir("/kaggle/working/context-aware-email-intelligence")
!git config user.email "you@example.com"
!git config user.name "Your Name"
```
**[VERIFY]** `!ls` shows `README.md` and `.gitignore`.

---

### Step 3.3 — Create the Full Directory Structure [ONE-TIME, then re-created each session via clone]

📎 **Code for this step**
```python
import os
dirs = [
    "data/raw/enron", "data/raw/bc3", "data/raw/emailsum", "data/raw/spamassassin",
    "data/processed/train", "data/processed/val", "data/processed/test",
    "models/checkpoints", "models/final", "models/mobile",
    "notebooks", "src", "tests", "docs", "android_app",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
```
**[VERIFY]** `!find . -maxdepth 2 -type d` lists all folders.

> Raw datasets live in `/kaggle/input/` (read-only, via Add Data), not inside `data/raw/` in git — that folder is for small reference files only.

---

### Step 3.4 — Understand Kaggle's Two Storage Areas [ONE-TIME, conceptual]

| Path | Persistence | Purpose |
|---|---|---|
| `/kaggle/input/` | Read-only, persists once attached | Datasets added via **Add Data** |
| `/kaggle/working/` | Read-write, **wiped when session ends** unless **Save Version** clicked | Active repo clone, notebooks, intermediate files |
| Kaggle Datasets (output) | Persists indefinitely | Published trained models/processed data, read back via `/kaggle/input/` |
| GitHub repo | Persists indefinitely | Code, notebooks, docs — the deliverable record |

---

### Step 3.5 — Push the Skeleton Structure to GitHub [EVERY SESSION, as needed]

📎 **Code for this step (run before commit cell so empty folders are trackable in Git)**
```python
from pathlib import Path
import os

REPO_ROOT = Path("/kaggle/working/context-aware-email-intelligence")
os.chdir(REPO_ROOT)

# Git does not track empty folders; place .gitkeep in folders that should exist in GitHub.
gitkeep_dirs = [
    "data/raw/enron", "data/raw/bc3", "data/raw/emailsum", "data/raw/spamassassin",
    "data/processed/train", "data/processed/val", "data/processed/test",
    "models/checkpoints", "models/final", "models/mobile",
    "notebooks", "src", "tests", "docs", "android_app",
]

for d in gitkeep_dirs:
    p = REPO_ROOT / d
    p.mkdir(parents=True, exist_ok=True)
    (p / ".gitkeep").touch(exist_ok=True)

print("Created/ensured .gitkeep files for Git folder tracking.")
```
**[VERIFY]** `!git status --short` shows newly added `.gitkeep` files.

```python
!git add .
!git commit -m "Initial project structure setup"
!git push origin main
```
**[VERIFY]** Folder structure visible on GitHub.

---

### Step 3.6 — Save a Notebook Version to Persist Working Files [EVERY SESSION]

Before closing a session: **Save Version** → **Save & Run All (Commit)**.

---

## Phase 4 — Dataset Acquisition on Kaggle

### Step 4.1 — Attach the Enron Email Dataset via Add Data [ONE-TIME PER NOTEBOOK]

1. **Add Data** → search `enron email dataset` → select `wcukierski/enron-email-dataset` → **Add**.
2. Mounted at `/kaggle/input/datasets/wcukierski/enron-email-dataset/`.
3. **[VERIFY]** `!ls /kaggle/input/datasets/wcukierski/enron-email-dataset/` shows `emails.csv`.

> **This guide uses the `emails.csv` format**, not the `maildir/` folder tree. `emails.csv` has two columns: `file` (an ID string like `allen-p/_sent_mail/1.`) and `message` (the full raw RFC822 email — headers and body together — as a single string). Every raw email is parsed out of that `message` column with Python's `email` library instead of being read off disk file-by-file, so all the Phase 6 preprocessing code below works directly from the CSV.

---

### Step 4.2 — Request and Attach the BC3 Corpus

1. Submit the access request at `https://www.cs.ubc.ca/cs/research/lci/facilities/bc3corpus.html` (purpose: "Academic research project on email thread summarization and NLP, under Samsung Research Collaboration Program.").
2. While waiting, continue with other datasets and Phase 5.
3. On approval: download the archive to your laptop → Kaggle **+ Create** → **New Dataset** → **Upload** → name `bc3-corpus`, **Private** → **Create**.
4. In your notebook: **Add Data** → **My Datasets** → attach `bc3-corpus`.
5. **[VERIFY]** `!ls /kaggle/input/datasets/prishabhkumar/bc3-corpus/` shows the XML file(s).

**Alternative if BC3 takes too long:** use Enron + EMAILSUM only, treat BC3 as a later validation benchmark.

---

### Step 4.3 — Attach and Inspect the EMAILSUM Summary Files from Kaggle [EVERY SESSION]

1. **Add Data** → **My Datasets** (or search) → select `prishabhkumar/emailsum` → **Add**.
2. Mounted at `/kaggle/input/datasets/prishabhkumar/emailsum/`.
3. The two files this project uses live at:
   - `/kaggle/input/datasets/prishabhkumar/emailsum/Avocado/summaries/EmailSum_data.json` — the 2,549 human-written summaries of Avocado email threads.
   - `/kaggle/input/datasets/prishabhkumar/emailsum/Avocado/summaries/one_more_reference.json` — a second reference summary collected for the 500 threads in the test set (useful later for two-reference ROUGE, not required for training).

> **Important — read before Phase 8:** per the [EmailSum release](https://github.com/ZhangShiyue/EmailSum), these two JSON files ship the human **summaries only**. The raw Avocado email/thread text is separately LDC-licensed and is *not* redistributed here. The directory listing below tells you whether this Kaggle mirror bundles thread text anywhere else — if it doesn't, these files alone can't give you `email thread → summary` training pairs, and BC3 (which does include full thread text with its summaries, Step 4.2) becomes your main source of real training pairs for the summarization task. Don't skip the [VERIFY] step below — it decides how much of Phase 8 you can actually do with EMAILSUM.

📎 **Code for this step** *(Cell 17 — also reused at Phase 8.2)*
```python
import json

EMAILSUM_DIR      = Path("/kaggle/input/datasets/prishabhkumar/emailsum")
MAIN_SUMMARY_JSON = EMAILSUM_DIR / "Avocado" / "summaries" / "EmailSum_data.json"
EXTRA_REF_JSON    = EMAILSUM_DIR / "Avocado" / "summaries" / "one_more_reference.json"

# List everything under the dataset root — reveals whether raw thread text
# ships alongside the summaries, or whether this is a summaries-only mirror.
print("Files found in EMAILSUM_DIR:")
for f in sorted(EMAILSUM_DIR.rglob("*")):
    if f.is_file():
        print(" ", f.relative_to(EMAILSUM_DIR))

with open(MAIN_SUMMARY_JSON) as f:
    emailsum_summaries = json.load(f)
with open(EXTRA_REF_JSON) as f:
    emailsum_extra_ref = json.load(f)

def peek(obj, name):
    print(f"\n{name} — top-level type: {type(obj).__name__}, entries: {len(obj)}")
    key = next(iter(obj)) if isinstance(obj, dict) else 0
    example = obj[key] if isinstance(obj, dict) else obj[0]
    print("  example key:", key)
    print("  example value:", example)

peek(emailsum_summaries, "EmailSum_data.json")
peek(emailsum_extra_ref, "one_more_reference.json")
```
**[VERIFY]** The printout tells you (a) whether records are keyed by thread ID (dict) or a flat list, and (b) the exact field names available per thread — especially whether any field holds raw thread/email text (vs. only summary text). Note the field names you see; you'll plug them into `THREAD_TEXT_FIELD` / `SHORT_SUMMARY_FIELD` / `LONG_SUMMARY_FIELD` in Step 8.4.

---

### Step 4.4 — Attach or Download the SpamAssassin Public Corpus

**Option A:** search **Add Data** for `spamassassin` — attach an existing mirror if the structure matches.

**Option B — download directly inside the notebook:**
```python
import os
os.makedirs("/kaggle/working/spamassassin_raw", exist_ok=True)
urls = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
]
for u in urls:
    !wget -q -P /kaggle/working/spamassassin_raw {u}
!cd /kaggle/working/spamassassin_raw && for f in *.tar.bz2; do tar -xjf "$f"; done
```
**[VERIFY]** `!ls /kaggle/working/spamassassin_raw` shows `easy_ham/`, `hard_ham/`, `spam/`, `spam_2/`.

---

### Step 4.5 — Verify All Dataset Attachments

| Dataset | Location | Expected Content | Status |
|---|---|---|---|
| Enron | `/kaggle/input/datasets/wcukierski/enron-email-dataset/` | `emails.csv` | ☐ |
| BC3 | `/kaggle/input/datasets/prishabhkumar/bc3-corpus/` | Annotated XML thread file(s) | ☐ |
| EMAILSUM | `/kaggle/input/datasets/prishabhkumar/emailsum/Avocado/summaries/` | `EmailSum_data.json`, `one_more_reference.json` | ☐ |
| SpamAssassin | `/kaggle/input/...` or `/kaggle/working/spamassassin_raw/` | `easy_ham/`, `hard_ham/`, `spam/` | ☐ |

Once you have **processed** (not raw) data, the check below confirms your final CSVs mounted correctly — reuse it whenever you attach the `email-intelligence-processed-data` dataset (Step 4.7 / Step 6.8) in a new notebook.

📎 **Code for this step** *(Cell 3 — Verify dataset files exist)*
```python
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
```

---

### Step 4.6 — Keep Raw Datasets Out of Git

```
# Large datasets - do not commit to git, even accidentally
data/raw/
data/processed/
models/checkpoints/
models/final/
models/mobile/*.tflite
models/mobile/*.ptl
*.zip
*.tar.bz2
*.tar.gz
```
```python
!git add .gitignore
!git commit -m "Update .gitignore to exclude datasets"
!git push origin main
```

---

### Step 4.7 — Alternative: Upload Already-Processed Data as a Kaggle Dataset

If you preprocessed data **locally** (outside Kaggle) — for example, your `Processed_Data/processed/` folder already contains `Cleaned data/`, `train/`, `val/`, and `test/` subfolders — you don't need to redo Phase 6 on Kaggle from raw files. Upload the finished CSVs directly as their own Kaggle Dataset instead.

**Expected local folder shape** (Kaggle preserves this structure on upload):
```
processed/
├── Cleaned data/
│   ├── enron_cleaned.csv
│   ├── spamassassin_cleaned.csv
│   └── intent_labeled.csv
├── train/
├── val/
└── test/
```

**Step A — Zip it locally:** select the `processed/` folder → right-click → **Send to** → **Compressed (zipped) folder** → name it `email-intelligence-processed.zip`. (Kaggle allows up to 20 GB per dataset — your CSVs will be well under that.)

**Step B — Create the Kaggle Dataset via the UI:**
1. `kaggle.com` → profile → **Your Work** → **Datasets** → **New Dataset**.
2. Drag and drop `email-intelligence-processed.zip`.
3. **Title:** `Email Intelligence Processed Data`. **Subtitle:** `Preprocessed Enron, SpamAssassin, and intent-labeled splits`. **Visibility:** **Private**.
4. **Create** → wait for processing.

**Step C — Attach it to a notebook:**
1. **Code** → **New Notebook** → right panel: **Accelerator: GPU T4 x2** (needed for Phases 10–11), **Internet: ON**.
2. **+ Add Input Data** → search your dataset → **Add**.
3. Data appears at `/kaggle/input/email-intelligence-processed-data/processed/` — exact slug may vary, verify with `!ls /kaggle/input/`.

**Step D — Optional CLI upload from your PC (instead of the drag-and-drop UI):**
1. Kaggle → profile → **Account** → **Create New Token** → downloads `kaggle.json`.
2. Place it at `C:\Users\YOUR_NAME\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (macOS/Linux).
3. `pip install kaggle`
4. Create `dataset-metadata.json` in the folder to upload:
   ```json
   {
     "title": "Email Intelligence Processed Data",
     "id": "your-kaggle-username/email-intelligence-processed",
     "licenses": [{"name": "CC0-1.0"}]
   }
   ```
5. Upload:
   ```bash
   cd "path/to/Processed_Data"
   kaggle datasets create -p . -r zip
   ```

**Step E — Getting trained models back out later** (relevant once you reach Phase 10–11): Kaggle wipes `/kaggle/working/` when a session ends, so after training either (a) **Save Version** → download from the **Output** tab, or (b) publish `/kaggle/working/models/` as a new Kaggle Dataset (same flow as above, using the notebook's own output files as the source instead of a local upload).

---

## Phase 5 — Dataset Exploration and Inventory

### Step 5.1 — Create an Exploration Notebook [ONE-TIME]

1. **+ Create** → **New Notebook** (or **Copy and Edit** your setup notebook) → rename `01-dataset-exploration`.
2. Repeat Steps 1.3–1.9 and 3.2 at the top of this notebook.
3. **Accelerator: None (CPU)** — exploration doesn't need a GPU.

📎 **Code for this step** *(Cell 4 — Load all processed CSVs, once your dataset is attached per Step 4.7)*
```python
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
```

---

### Step 5.2 — Explore the Enron Dataset Structure (Phase 5.2)

If you're exploring **raw** Enron messages fresh from `/kaggle/input/datasets/wcukierski/enron-email-dataset/emails.csv`, load the CSV with `pd.read_csv` and parse each row's `message` column with Python's `email` library (see Step 6.3/6.4 below). If you're working from the **already-cleaned** CSV (Step 4.7), use the code below instead:

📎 **Code for this step** *(Cell 5 — Explore Enron cleaned data)*
```python
# Basic stats on Enron cleaned emails
enron_full["body_word_count"] = enron_full["cleaned_body"].astype(str).str.split().str.len()

print(f"Total Enron emails: {len(enron_full):,}")
print(f"Avg body length (words): {enron_full['body_word_count'].mean():.1f}")
print(f"Min / Max words: {enron_full['body_word_count'].min()} / {enron_full['body_word_count'].max()}")

# Sample one email
sample = enron_full.iloc[0]
print(f"\n--- Sample email ---")
print(f"File: {sample['file']}")
print(f"Body (first 500 chars):\n{str(sample['cleaned_body'])[:500]}")
```

**Record in a markdown cell:** total email count, average length, notable observations.

---

### Step 5.3 — Explore the SpamAssassin Dataset Structure (Phase 5.3)

📎 **Code for this step** *(Cell 6)*
```python
print("Spam vs Ham distribution:")
print(spam_full["label"].value_counts().rename({0: "ham", 1: "spam"}))

spam_sample = spam_full[spam_full["label"] == 1].iloc[0]
ham_sample  = spam_full[spam_full["label"] == 0].iloc[0]

print("\n--- SPAM sample ---")
print(str(spam_sample["cleaned_body"])[:400])
print("\n--- HAM sample ---")
print(str(ham_sample["cleaned_body"])[:400])
```

---

### Step 5.4 — Explore the BC3 Dataset (Once Attached)

Use `xml.etree.ElementTree` to open the BC3 XML, count threads (~40 expected), and print one thread's emails, human-written summary, and speech-act labels. Note the tag names for your Phase 6 parser.

---

### Step 5.5 — Explore the EMAILSUM Dataset

Inspect the `emailsum_summaries` object loaded in Step 4.3 — thread count, the field names printed there, and average summary lengths. Confirm for yourself whether raw thread text is present (see the Step 4.3 note) before you get to Phase 8. (You'll reuse and reformat this in Phase 8/11 — see Cell 17/18 there.)

---

### Step 5.6 — Commit Your Exploration Notebook

```python
!cp /kaggle/working/*.ipynb notebooks/01_Dataset_Exploration.ipynb 2>/dev/null || true
!git add notebooks/01_Dataset_Exploration.ipynb
!git commit -m "Add dataset exploration notebook"
!git push origin main
```
Then click **Save Version** in Kaggle.

---

## Phase 6 — Dataset Preprocessing and Curation

> If you've already got processed CSVs locally and uploaded them per Step 4.7, you can treat this phase as **documentation of the pipeline that produced them** rather than something to re-run inside Kaggle. If you're processing raw data for the first time, run these steps as written.

### Step 6.1 — Create a New Notebook for Preprocessing

Create `02-preprocessing-pipeline`, bootstrap (Steps 1.3–1.9, 3.2), **Accelerator: None**.

---

### Step 6.2 — Plan and Document the Preprocessing Steps

```
Email Preprocessing Pipeline:
1. Parse raw email using Python's `email` library → extract headers and body separately
2. Split email thread into individual messages using delimiter patterns
3. Strip HTML tags from body using BeautifulSoup4
4. Remove email signatures (lines starting with "--", "Best", "Regards", "Thanks", etc.)
5. Remove quoted reply lines (lines starting with ">")
6. Normalize whitespace (collapse multiple blank lines)
7. Lowercase all text
8. Tokenize into sentences using spaCy or NLTK
9. Tokenize sentences into words
10. Remove stopwords using NLTK stopword list + custom email stopwords
11. Save cleaned output to CSV
```

Below is the actual function library that implements steps 1–7 (and the tokenization helpers for steps 8–10) — every later step in this phase imports and reuses these functions instead of re-writing the logic.

📎 **Code for this step** *(Cell 6a — Core preprocessing function library)*
```python
import os
import re
import time
import email
from email import policy
from pathlib import Path
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("english"))
EMAIL_STOPWORDS = {"fwd", "re", "fw", "subject", "from", "to", "cc", "bcc", "sent"}
ALL_STOPWORDS = STOPWORDS | EMAIL_STOPWORDS

SIGNATURE_MARKERS = (
    "--", "best,", "best regards", "regards,", "regards", "thanks,", "thanks",
    "thank you,", "sincerely,", "sincerely", "cheers,", "cheers", "sent from my",
)
QUOTE_PREFIXES = (">", "&gt;")


def parse_raw_email(raw_text):
    """Step 1 — parse a raw RFC822 message string (the `message` column of emails.csv,
    or a raw .eml file's contents) into headers + body via Python's email lib."""
    msg = email.message_from_string(raw_text, policy=policy.default)
    headers = {
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "date": msg.get("Date", ""),
    }
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html") and not part.get("Content-Disposition"):
                try:
                    body += part.get_content()
                except Exception:
                    pass
    else:
        try:
            body = msg.get_content()
        except Exception:
            body = str(msg.get_payload())
    return headers, body


def split_thread_into_messages(body):
    """Step 2 — split a reply chain into individual messages using common delimiter patterns."""
    delimiters = [
        r"-{2,}\s*Original Message\s*-{2,}",
        r"On .{5,80} wrote:",
        r"From:.{0,120}\nSent:.{0,120}\nTo:.{0,120}\nSubject:",
    ]
    parts = re.split("|".join(delimiters), body, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def strip_html(text):
    """Step 3 — strip HTML tags with BeautifulSoup4 (only runs BS4 when tags are present)."""
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return text


def remove_signature(text):
    """Step 4 — cut everything from the first signature marker line onward."""
    lines = text.split("\n")
    cut_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped in SIGNATURE_MARKERS or stripped.startswith("--"):
            cut_idx = i
            break
    return "\n".join(lines[:cut_idx])


def remove_quoted_lines(text):
    """Step 5 — drop quoted reply lines (start with '>')."""
    return "\n".join(l for l in text.split("\n") if not l.strip().startswith(QUOTE_PREFIXES))


def normalize_whitespace(text):
    """Step 6 — collapse repeated spaces/blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_sentences(text):
    """Step 8 — sentence tokenization (NLTK)."""
    return sent_tokenize(text)


def tokenize_words_no_stopwords(text):
    """Steps 9–10 — word tokenize + remove stopwords."""
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in ALL_STOPWORDS]


def clean_email_body(raw_body):
    """Full pipeline (steps 3–7) — returns the cleaned_body text used in the output CSVs."""
    text = strip_html(raw_body)
    text = remove_signature(text)
    text = remove_quoted_lines(text)
    text = normalize_whitespace(text)
    text = text.lower()  # step 7
    return text


print("Preprocessing functions ready: parse_raw_email, split_thread_into_messages,")
print("strip_html, remove_signature, remove_quoted_lines, normalize_whitespace,")
print("tokenize_sentences, tokenize_words_no_stopwords, clean_email_body")
```
**[VERIFY]** The `print` statement lists all nine functions with no `NameError`/`ImportError`.

---

### Step 6.3 — Test Preprocessing on 10 Sample Emails

Load `/kaggle/input/datasets/wcukierski/enron-email-dataset/emails.csv`, take 10 rows spread across different senders, run your cleaning code, inspect input vs. output, adjust logic.

📎 **Code for this step** *(Cell 6b — Test the pipeline on 10 sample emails)*
```python
import random

ENRON_CSV = Path("/kaggle/input/datasets/wcukierski/enron-email-dataset/emails.csv")

# Load the CSV (columns: "file", "message" — "message" holds the full raw RFC822 text)
emails_df = pd.read_csv(ENRON_CSV)
print(f"Total rows in emails.csv: {len(emails_df):,}")
print(f"Columns: {list(emails_df.columns)}")

random.seed(42)
sample_idx = random.sample(range(len(emails_df)), min(10, len(emails_df)))

for idx in sample_idx:
    row = emails_df.iloc[idx]
    raw = row["message"]
    headers, body = parse_raw_email(raw)
    cleaned = clean_email_body(body)

    print("=" * 70)
    print(f"File ID: {row['file']}")
    print(f"Subject: {headers['subject']}")
    print(f"--- RAW (first 300 chars) ---\n{body[:300]}")
    print(f"--- CLEANED (first 300 chars) ---\n{cleaned[:300]}")
```
**[VERIFY]** For each of the 10 samples, the cleaned output has no HTML tags, no `>`-quoted lines, and no trailing signature block. If something slips through, adjust `SIGNATURE_MARKERS` / the regex in `remove_quoted_lines` in Cell 6a and re-run this cell before moving on.

---

### Step 6.4 — Process the Full Enron Dataset

Apply the pipeline to every row of `emails.csv`; save `file, sender, date, subject, cleaned_body` to `data/processed/enron_cleaned.csv`; log processed/skipped/empty counts.

📎 **Code for this step** *(Cell 6c — Process the full Enron dataset)*
```python
def process_enron_dataset(emails_csv_path, out_csv_path, limit=None):
    """Reads emails.csv (columns: file, message) and applies the Step 6.2 cleaning
    pipeline to the raw text in `message`, row by row."""
    records, skipped, empty = [], 0, 0
    start = time.time()

    raw_df = pd.read_csv(emails_csv_path)
    if limit:
        raw_df = raw_df.head(limit)

    for _, row in raw_df.iterrows():
        try:
            headers, body = parse_raw_email(row["message"])
            cleaned = clean_email_body(body)
            if not cleaned or len(cleaned.split()) < 3:
                empty += 1
                continue
            records.append({
                "file": row["file"],
                "sender": headers["from"],
                "date": headers["date"],
                "subject": headers["subject"],
                "cleaned_body": cleaned,
            })
        except Exception:
            skipped += 1

    df = pd.DataFrame(records)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    print(f"Processed: {len(df):,} | Skipped (errors): {skipped:,} | Empty/too short: {empty:,}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} min")
    print(f"Saved to: {out_csv_path}")
    return df

# Start with a small limit (e.g. limit=200) to sanity-check timing/output, then re-run
# with limit=None for the full dataset.
ENRON_CSV = Path("/kaggle/input/datasets/wcukierski/enron-email-dataset/emails.csv")
ENRON_OUT = Path("data/processed/Cleaned data/enron_cleaned.csv")
enron_cleaned_df = process_enron_dataset(ENRON_CSV, ENRON_OUT, limit=None)
enron_cleaned_df.head()
```
**[VERIFY]** `enron_cleaned_df.shape[0]` is a large majority of the total row count in `emails.csv` from Step 4.1, and `enron_cleaned.csv` exists with columns `file, sender, date, subject, cleaned_body`.

---

### Step 6.5 — Process the SpamAssassin Dataset

Same pipeline + `label` column (0 = ham, 1 = spam) → `data/processed/spamassassin_cleaned.csv`.

📎 **Code for this step** *(Cell 6d — Process the SpamAssassin dataset)*
```python
# Points at the folders downloaded in Step 4.4, Option B. If you attached an existing
# mirror instead (Option A), point SPAM_RAW_DIR at that dataset's input path.
SPAM_RAW_DIR = Path("/kaggle/working/spamassassin_raw")
SPAM_FOLDER_LABELS = {"easy_ham": 0, "hard_ham": 0, "spam": 1, "spam_2": 1}

def process_spamassassin_dataset(raw_dir, out_csv_path):
    records, skipped = [], 0

    for folder_name, label in SPAM_FOLDER_LABELS.items():
        folder_path = raw_dir / folder_name
        if not folder_path.exists():
            print(f"  (skipping missing folder: {folder_name})")
            continue
        for fname in os.listdir(folder_path):
            fpath = folder_path / fname
            if not fpath.is_file():
                continue
            try:
                with open(fpath, "r", encoding="latin-1") as fh:
                    raw = fh.read()
                headers, body = parse_raw_email(raw)
                cleaned = clean_email_body(body)
                if not cleaned:
                    continue
                records.append({"file": f"{folder_name}/{fname}", "cleaned_body": cleaned, "label": label})
            except Exception:
                skipped += 1

    df = pd.DataFrame(records)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    print(f"Processed: {len(df):,} | Skipped: {skipped:,}")
    print(df["label"].value_counts().rename({0: "ham", 1: "spam"}))
    print(f"Saved to: {out_csv_path}")
    return df

SPAM_OUT = Path("data/processed/Cleaned data/spamassassin_cleaned.csv")
spam_cleaned_df = process_spamassassin_dataset(SPAM_RAW_DIR, SPAM_OUT)
spam_cleaned_df.head()
```
**[VERIFY]** `spam_cleaned_df["label"].value_counts()` shows a nonzero count for both `0` (ham) and `1` (spam).

---

### Step 6.6 — Process the BC3 Dataset

Parse threads + summaries → `data/processed/bc3_threads.json`.

📎 **Code for this step** *(Cell 6e — Process the BC3 dataset)*
```python
import xml.etree.ElementTree as ET
import json

BC3_DIR = Path("/kaggle/input/datasets/prishabhkumar/bc3-corpus")  # attached in Step 4.2

def find_bc3_xml_file(bc3_dir):
    for root, dirs, files in os.walk(bc3_dir):
        for f in files:
            if f.lower().endswith(".xml"):
                return Path(root) / f
    return None

def process_bc3_dataset(bc3_dir, out_json_path):
    xml_file = find_bc3_xml_file(bc3_dir)
    if xml_file is None:
        print("No BC3 XML file found — skip this step until the corpus is attached (Step 4.2).")
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    threads = []

    # Tag names below match the standard BC3 release; if Step 5.4 showed different tag
    # names for your copy, update the .findall()/.find() calls to match.
    for thread_el in root.findall(".//thread"):
        thread_id = thread_el.get("id", f"thread_{len(threads)}")
        messages = []
        for doc_el in thread_el.findall(".//DOC"):
            sender = doc_el.findtext("From", default="")
            subject = doc_el.findtext("Subject", default="")
            body = " ".join(s.text or "" for s in doc_el.findall(".//Sent"))
            messages.append({"sender": sender, "subject": subject, "body": clean_email_body(body)})

        summaries = [s.text or "" for s in thread_el.findall(".//summary")]
        threads.append({"thread_id": thread_id, "messages": messages, "human_summaries": summaries})

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as fh:
        json.dump(threads, fh, indent=2)

    print(f"Parsed {len(threads)} BC3 threads -> {out_json_path}")
    return threads

BC3_OUT = Path("data/processed/bc3_threads.json")
bc3_threads = process_bc3_dataset(BC3_DIR, BC3_OUT)
```
**[VERIFY]** `len(bc3_threads)` is close to the ~40 threads noted in Step 5.4 (or the cell prints the "not found" message if BC3 access is still pending — that's expected while you wait on Step 4.2).

---

### Step 6.7 — Create the Train / Validation / Test Split

`sklearn.model_selection.train_test_split`: 15% test, ~18% of remainder as val, 70% train. Save under `data/processed/{train,val,test}/`. **[VERIFY]** counts sum to total.

📎 **Code for this step** *(Cell 6f — Train / validation / test split)*
```python
from sklearn.model_selection import train_test_split

def make_splits(df, out_dir, name, test_size=0.15, val_size=0.18, stratify_col=None, seed=42):
    """70% train / ~15% val / 15% test. Val is carved out of the remainder after the
    test split, so val_size=0.18 of the 85% remainder works out to ~15% of the total."""
    stratify = df[stratify_col] if stratify_col else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)

    stratify2 = train_df[stratify_col] if stratify_col else None
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=stratify2)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = out_dir / split_name / f"{name}_{split_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(out_path, index=False)

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"{name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
          f"sum={total}, matches original={total == len(df)}")
    return train_df, val_df, test_df

PROCESSED_DIR = Path("data/processed")

enron_train, enron_val, enron_test = make_splits(enron_cleaned_df, PROCESSED_DIR, "enron")
spam_train, spam_val, spam_test = make_splits(spam_cleaned_df, PROCESSED_DIR, "spamassassin", stratify_col="label")

# The intent split (intent_train/val/test.csv) is created in Phase 7, Step 7.5, once
# intent_labeled.csv exists — it reuses this same make_splits() helper with
# stratify_col="intent_label" so each class is represented in every split.
```
**[VERIFY]** Each printed `sum=` matches the corresponding `len(df)`, and `data/processed/{train,val,test}/` now contain `enron_*.csv` and `spamassassin_*.csv`.

---

### Step 6.8 — Publish Processed Data as a Kaggle Dataset [DECISION]

1. **Save Version** → **Output** tab → **New Dataset** from output files → name `email-nlp-processed-data`, **Private**.
2. Attach via **Add Data** → **My Datasets** in future notebooks instead of regenerating everything.

---

### Step 6.9 — Commit All Preprocessing Work

```python
!cp /kaggle/working/*.ipynb notebooks/02_Preprocessing_Pipeline.ipynb 2>/dev/null || true
!git add notebooks/02_Preprocessing_Pipeline.ipynb
!git commit -m "Add full preprocessing pipeline"
!git push origin main
```

---

## Phase 7 — Intent Classification Data Preparation

### Step 7.1 — Create Notebook `03-intent-data-preparation`

Bootstrap as before. Attach `email-nlp-processed-data` via **Add Data**.

---

### Step 7.2 — Define Intent Labels and Examples

| Label | ID | Example phrases |
|---|---|---|
| REQUEST | 0 | "Could you please send...", "Can you review..." |
| FOLLOW_UP | 1 | "Just following up on...", "Any update on..." |
| INFORMATION | 2 | "Please note that...", "I wanted to inform you..." |
| ACKNOWLEDGEMENT | 3 | "Noted, thanks.", "Confirmed, we'll proceed." |
| COMPLAINT | 4 | "I'm disappointed...", "This is unacceptable..." |
| INVITATION | 5 | "You're invited to...", "Please join us for..." |

---

### Step 7.3 — Select and Label a Subset of Enron Emails

Sample 600 emails (100/class target), export to CSV, label in Google Sheets/Excel (download via notebook Output tab, or commit/pull through GitHub), split labeling 300/300, cross-validate 60 (10%). Save as `data/processed/intent_labeled.csv`.

> **Time estimate:** ~10 hours total at 1 minute/email.

---

### Step 7.4 — Validate Label Distribution (Phase 7.4)

📎 **Code for this step** *(Cell 7)*
```python
print("Intent label distribution (full labeled set):")
dist = intent_full["intent_label"].value_counts().sort_index()
for label_id, count in dist.items():
    print(f"  {label_id} ({INTENT_LABELS[label_id]}): {count}")

# Verify train+val+test sums to total
split_total = len(intent_train) + len(intent_val) + len(intent_test)
print(f"\nSplit check: train({len(intent_train)}) + val({len(intent_val)}) + test({len(intent_test)}) = {split_total}")
print(f"Matches intent_full ({len(intent_full)}): {split_total == len(intent_full)}")

# Plot distribution
plt.figure(figsize=(10, 4))
labels_named = [INTENT_LABELS[i] for i in dist.index]
plt.bar(labels_named, dist.values, color="steelblue")
plt.title("Intent Label Distribution")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```
**[VERIFY]** Each class has ≥60 samples; split counts match the full labeled set.

---

### Step 7.5 — Split Intent Data

70/15/15 split → `data/processed/train/intent_train.csv`, `.../val/intent_val.csv`, `.../test/intent_test.csv`.

---

### Step 7.6 — Commit Intent Data Preparation Work

```python
!git add notebooks/03_Intent_Data_Preparation.ipynb data/processed/intent_labeled.csv
!git commit -m "Add intent classification data preparation"
!git push origin main
```

---

## Phase 8 — Summarization Data Preparation

### Step 8.1 — Create Notebook `04-summarization-data-preparation`

---

### Step 8.2 — Load the EMAILSUM Summary Files from Kaggle (Phase 8.2)

Reuse the Step 4.3 / Cell 17 code (re-paste it here if this is a fresh notebook):
```python
import json

EMAILSUM_DIR      = Path("/kaggle/input/datasets/prishabhkumar/emailsum")
MAIN_SUMMARY_JSON = EMAILSUM_DIR / "Avocado" / "summaries" / "EmailSum_data.json"
EXTRA_REF_JSON    = EMAILSUM_DIR / "Avocado" / "summaries" / "one_more_reference.json"

with open(MAIN_SUMMARY_JSON) as f:
    emailsum_summaries = json.load(f)
with open(EXTRA_REF_JSON) as f:
    emailsum_extra_ref = json.load(f)

print(f"EmailSum_data.json: {len(emailsum_summaries)} threads")
print(f"one_more_reference.json: {len(emailsum_extra_ref)} threads")
```

---

### Step 8.3 — Load BC3 Summarization Data

Load `data/processed/bc3_threads.json`, reformat to `{input, short_summary}` matching EMAILSUM's structure. Treat BC3 (~40 threads) as extra validation data.

---

### Step 8.4 — Prepare the Combined Summarization Dataset (Phase 8.4)

Unlike a Hugging Face `load_dataset()` call, the raw `EmailSum_data.json` doesn't ship pre-made `train`/`val`/`test` splits or a fixed schema — so this step (a) flattens the JSON into a DataFrame using the field names you confirmed in Step 4.3/8.2, (b) splits it with the same `make_splits()` helper used for Enron/SpamAssassin (Step 6.7), and (c) wraps the splits back into HF `Dataset` objects so every later cell that calls `.map()` / `.filter()` / `.select()` on `sum_train` / `sum_val` / `sum_test` keeps working unchanged.

📎 **Code for this step** *(Cell 18 — Prepare summarization data)*
```python
from datasets import Dataset, DatasetDict

# Field names inside each EmailSum_data.json record — UPDATE these to match
# what the Step 4.3 / 8.2 printout actually showed.
THREAD_TEXT_FIELD    = "thread"   # raw email/thread text field, if one exists — set to None if it doesn't
SHORT_SUMMARY_FIELD  = "short"    # short (<30 word) human summary
LONG_SUMMARY_FIELD   = "long"     # long (<100 word) human summary
SUMMARY_TARGET_FIELD = SHORT_SUMMARY_FIELD   # switch to LONG_SUMMARY_FIELD for the long-summary task

items = emailsum_summaries.items() if isinstance(emailsum_summaries, dict) else enumerate(emailsum_summaries)

records = []
skipped = 0
for thread_id, rec in items:
    target = rec.get(SUMMARY_TARGET_FIELD) if isinstance(rec, dict) else None
    thread_text = rec.get(THREAD_TEXT_FIELD) if (THREAD_TEXT_FIELD and isinstance(rec, dict)) else None
    if not target or (THREAD_TEXT_FIELD and not thread_text):
        skipped += 1
        continue
    records.append({
        "thread_id": str(thread_id),
        "email_body": thread_text,
        "short_summary": rec.get(SHORT_SUMMARY_FIELD),
        "long_summary": rec.get(LONG_SUMMARY_FIELD),
    })

emailsum_df = pd.DataFrame(records)
print(f"Parsed {len(emailsum_df)} usable thread records, skipped {skipped}.")
if len(emailsum_df) == 0:
    print("⚠ No records had both a thread-text field and a summary field — see the Step 4.3 note. "
          "You'll need to either locate thread text elsewhere in the Kaggle dataset, join these "
          "summaries against your own licensed copy of Avocado, or rely on BC3 (Step 8.3) as your "
          "training source for this task.")

# No pre-made split ships with the raw JSON — create one with the Step 6.7 helper
emailsum_train_df, emailsum_val_df, emailsum_test_df = make_splits(emailsum_df, PROCESSED_DIR, "emailsum")

emailsum = DatasetDict({
    "train":      Dataset.from_pandas(emailsum_train_df, preserve_index=False),
    "validation": Dataset.from_pandas(emailsum_val_df, preserve_index=False),
    "test":       Dataset.from_pandas(emailsum_test_df, preserve_index=False),
})

# EMAILSUM output column names — standardized above, no more guessing needed
INPUT_COL  = "email_body"
TARGET_COL = "short_summary"    # or "long_summary"

def preprocess_summarization(example):
    inp  = str(example[INPUT_COL]).strip()
    tgt  = str(example[TARGET_COL]).strip()
    # T5 expects a task prefix
    example["input_text"]  = "summarize: " + inp
    example["target_text"] = tgt
    return example

sum_train = emailsum["train"].map(preprocess_summarization)
sum_val   = emailsum["validation"].map(preprocess_summarization)
sum_test  = emailsum["test"].map(preprocess_summarization)

# Filter: min 50 words input, max 150 words summary (roadmap Phase 8.4)
def quality_filter(ex):
    inp_words = len(ex["input_text"].split())
    tgt_words = len(ex["target_text"].split())
    return inp_words >= 50 and tgt_words <= 150 and len(ex["target_text"]) > 0

sum_train = sum_train.filter(quality_filter)
sum_val   = sum_val.filter(quality_filter)
sum_test  = sum_test.filter(quality_filter)

print(f"Filtered — train: {len(sum_train)}, val: {len(sum_val)}, test: {len(sum_test)}")
```
> **What it does:** Flattens the raw EMAILSUM JSON into a DataFrame, splits it 70/15/15 with `make_splits()`, formats it for T5 (`summarize: ...` prefix), and filters by the word-count rules above. The split CSVs also land in `data/processed/{train,val,test}/emailsum_*.csv` alongside the Enron/SpamAssassin splits, so you have durable copies rather than re-deriving from the raw JSON every session. `THREAD_TEXT_FIELD`/`SHORT_SUMMARY_FIELD`/`LONG_SUMMARY_FIELD` above must match the actual field names printed in Step 4.3/8.2 — the Kaggle JSON mirror may not use the same names as the original paper's code.

---

### Step 8.5 — Commit Summarization Data Work

```python
!git add notebooks/04_Summarization_Data_Preparation.ipynb
!git commit -m "Add summarization data preparation"
!git push origin main
```

---

## Phase 9 — NLP Pipeline Assembly and Baseline Testing

### Step 9.1 — Create Notebook `05-nlp-pipeline-baseline`

**Accelerator: None (CPU)**.

📎 **Code for this step** *(Cell 8 — Markdown section header)*
```markdown
## Phase 9 — NLP Pipeline Baseline
Test spaCy NER and Gensim LDA on sample emails before model training.
```

---

### Step 9.2 — Test Preprocessing on 5 New Emails

Pick 5 unseen Enron test emails, run the cleaning pipeline, verify noise removal.

---

### Step 9.3 — Test the NER Module Baseline (Phase 9.3)

📎 **Code for this step** *(Cell 9)*
```python
!python -m spacy download en_core_web_sm

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

# Run on 3 intent test emails
for i, row in intent_test.head(3).iterrows():
    print(f"\n{'='*60}")
    print(f"Subject: {row['subject']}")
    print(f"Intent: {INTENT_LABELS[row['intent_label']]}")
    ents = extract_entities(row["cleaned_body"])
    for etype, names in ents.items():
        print(f"  {etype}: {names}")
```
Manually evaluate: correctness, missed entities, false positives. Record findings — this informs whether `en_core_web_trf` is worth the extra download size later.

---

### Step 9.4 — Test LDA Topic Modeling on a Sample Thread (Phase 9.4)

📎 **Code for this step** *(Cell 10)*
```python
!pip install -q gensim
from gensim import corpora, models
import re

def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]

# Take 500 Enron emails for LDA demo
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

---

### Step 9.5 — Document Baseline Observations

Markdown cell: NER quality, LDA appropriateness, data surprises.

---

### Step 9.6 — Commit Pipeline Baseline Notebook

```python
!git add notebooks/05_NLP_Pipeline_Baseline.ipynb
!git commit -m "Add NLP pipeline baseline testing"
!git push origin main
```

---

## Phase 10 — Model Training (Intent Classifier)

### Step 10.1 — Create Training Notebook `06-intent-classifier-training`

Bootstrap fully. Attach `email-nlp-processed-data`.

📎 **Code for this step** *(Cell 11 — Markdown section header)*
```markdown
## Phase 10 — Intent Classifier (DistilBERT)
Fine-tune `distilbert-base-uncased` on your pre-split intent CSVs.
```

---

### Step 10.2 — Set the Accelerator [DECISION] [EVERY SESSION]

**Accelerator: GPU T4 x2** (or **P100**) — replaces the old "local GPU vs. Colab" decision entirely; Kaggle's free GPU quota covers this comfortably.

**[VERIFY]** `torch.cuda.is_available()` → `True`.

---

### Step 10.3 — Load Training Data (Phase 10.3)

📎 **Code for this step** *(Cell 12 — Prepare intent datasets for Hugging Face Trainer)*
```python
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
```

---

### Step 10.4 — Download the Pre-Trained Base Model & Step 10.5 — Configure Hyperparameters

📎 **Code for this step** *(Cell 13 — Define intent metrics & load model)*
```python
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
```

Hyperparameters used (documented for reference):
```
Model: distilbert-base-uncased
Number of labels: 6
Max sequence length: 512
Batch size: 16
Learning rate: 2e-5
Epochs: 4
Optimizer: AdamW
Evaluation: Per epoch
Checkpointing: Save best model based on validation macro F1
Output directory: models/checkpoints/intent_classifier/
```

---

### Step 10.6 — Run Training (Phase 10.6)

📎 **Code for this step** *(Cell 14 — Train intent classifier)*
```python
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
```
**[VERIFY]** Checkpoint files exist in `models/checkpoints/intent_classifier/`. Watch your GPU-hour quota.

---

### Step 10.7 — Save the Final Intent Classifier

1. Copy the best checkpoint to `models/final/intent_classifier/` (already done by the cell above via `.../best`).
2. Publish as a Kaggle Dataset `email-nlp-intent-classifier` (Step 6.8-style flow) so it survives past this session.

---

### Step 10.8 — Commit Intent Classifier Training Notebook

```python
!git add notebooks/06_Intent_Classifier_Training.ipynb
!git commit -m "Add intent classifier training"
!git push origin main
```

---

## Phase 11 — Model Training (Summarization)

### Step 11.1 — Create Training Notebook `07-summarization-model-training`

Bootstrap fully, attach `email-nlp-processed-data`, **Accelerator: GPU T4 x2** or **P100**.

📎 **Code for this step** *(Cell 16 — Markdown section header)*
```markdown
## Phase 11 — Summarization (T5-Small + EMAILSUM)
Summarization CSVs are not in Processed_Data yet.
We load **EMAILSUM** directly from the Kaggle dataset `prishabhkumar/emailsum` at `/kaggle/input/datasets/prishabhkumar/emailsum/`.
```

---

### Step 11.2 — Choose Your Summarization Base Model [DECISION]

| Model | HuggingFace ID | Size | Recommendation |
|---|---|---|---|
| T5-Small | `t5-small` | ~240 MB | **Best for mobile.** Start here. |
| T5-Base | `t5-base` | ~890 MB | Better quality, harder to compress for mobile |
| DistilBART | `sshleifer/distilbart-cnn-12-6` | ~1.2 GB | Good quality, larger |

---

### Step 11.3 / 11.4 — Load and Tokenize Summarization Training Data

📎 **Code for this step** *(Cell 19 — Tokenize summarization data)*
```python
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
```

---

### Step 11.5 — Run Summarization Fine-Tuning (Phase 11.5)

📎 **Code for this step** *(Cell 20 — Train summarization model)*
```python
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
```
Expect roughly 30–90 minutes on a Kaggle T4.

---

### Step 11.6 — Save the Final Summarization Model & Visual Check

📎 **Code for this step** *(Cell 21 — Quick summarization demo)*
```python
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
```
Publish the model as a Kaggle Dataset `email-nlp-summarization-model` for reuse.

---

### Step 11.7 — Commit Summarization Training Notebook

```python
!git add notebooks/07_Summarization_Model_Training.ipynb
!git commit -m "Add summarization model training"
!git push origin main
```

---

## Phase 12 — Model Evaluation (Kaggle-Side)

### Step 12.1 — Create Evaluation Notebook `08-model-evaluation`

Attach `email-nlp-intent-classifier`, `email-nlp-summarization-model`, and `email-nlp-processed-data`. Accelerator: GPU optional.

---

### Step 12.2 — Evaluate Summarization with ROUGE Scores (Phase 12.2)

📎 **Code for this step** *(Cell 22 — ROUGE evaluation on test set)*
```python
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

# Evaluate on first 100 test samples (full set takes longer)
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

---

### Step 12.3 — Evaluate Intent Classifier (Phase 12.3)

📎 **Code for this step** *(Cell 15 — Evaluate intent classifier on test set)*
```python
from transformers import pipeline

intent_pipe = pipeline(
    "text-classification",
    model=str(intent_output_dir / "best"),
    tokenizer=str(intent_output_dir / "best"),
    device=0 if torch.cuda.is_available() else -1,
)

# Batch predict on test set
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

---

### Step 12.4 — Evaluate NER (Baseline spaCy)

Manually annotate entities in 10 of 50 sampled test emails as ground truth, run spaCy NER on the same 10 (reuse `extract_entities` from Step 9.3), compare, compute precision/recall/F1.

---

### Step 12.5 — Create a Summary Evaluation Table (Phase 12.5)

📎 **Code for this step** *(Cell 25 — Markdown results summary template)*
```markdown
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

**Bonus — end-to-end pipeline demo**, useful for sanity-checking the full flow before moving to Android:

📎 **Code** *(Cell 23 — End-to-end pipeline demo)*
```python
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

---

### Step 12.6 — Commit Evaluation Notebook

```python
!git add notebooks/08_Model_Evaluation.ipynb
!git commit -m "Add model evaluation results"
!git push origin main
```

---

## Phase 13 — Model Export and Mobile Conversion

### Step 13.1 — Create Conversion Notebook `09-model-mobile-conversion`

Attach the two trained-model Kaggle Datasets. Accelerator: CPU usually sufficient.

---

### Step 13.2 — Confirm Your Mobile Framework [DECISION]

**TFLite** if trained with TensorFlow / want the better-documented Android path. **PyTorch Mobile** if minimizing framework changes from Hugging Face/PyTorch. **Recommendation:** TFLite.

---

### Step 13.3 — Export the Summarization Model to Mobile Format

**TFLite:** export to TensorFlow SavedModel → TFLite Converter → float16 (then int8 if needed) quantization → `models/mobile/summarization_model.tflite` (< 50 MB target).
**PyTorch Mobile:** TorchScript → mobile optimizer → `models/mobile/summarization_model.ptl`.

---

### Step 13.4 — Export the Intent Classifier to Mobile Format

Repeat Step 13.3 → `models/mobile/intent_classifier.tflite` (or `.ptl`).

---

### Step 13.5 — Handle the spaCy NER Model

spaCy doesn't convert to TFLite directly — ship it as a separate asset, or implement a regex/rule-based entity extractor in Kotlin. Note `en_core_web_sm`'s size via `python -m spacy info en_core_web_sm`.

---

### Step 13.6 — Benchmark the Mobile Models (Kaggle as PC-equivalent baseline)

Load the `.tflite` file with the TFLite Python interpreter, run inference on 10 test emails, record latency, compare pre/post-quantization accuracy.

---

### Step 13.7 — Record Conversion Results

```
| Model | Original Size | After Quantization | Inference Time (Kaggle CPU) | ROUGE/F1 Retained |
|---|---|---|---|---|
| Summarization | X MB | X MB | X ms | X% |
| Intent Classifier | X MB | X MB | X ms | X% |
```

---

### Step 13.8 — Publish the Mobile Model Files and Commit

Package everything you'll need on your laptop for the Android phases into a single zip before the session ends.

📎 **Code for this step** *(Cell 24 — Save & package outputs for download)*
```python
import shutil

# Zip all trained models for download before Kaggle session ends
zip_path = WORK_DIR / "trained_models.zip"
shutil.make_archive(str(WORK_DIR / "trained_models"), "zip", MODEL_DIR)

print(f"Models zipped: {zip_path}")
print(f"Size: {zip_path.stat().st_size / 1e6:.1f} MB")
print("\nDownload from: Notebook → Output tab (after Save Version)")
```
Then:
```python
!git add notebooks/09_Model_Mobile_Conversion.ipynb
!git commit -m "Add model mobile conversion"
!git push origin main
```
Download `trained_models.zip` (or the individual `.tflite`/`.ptl` files) from the notebook's **Output** tab onto your laptop — you need them locally for Phase 14.

---

## Phase 14 — Android Application Setup (Local Laptop Required)

> Phases 14–16 happen on your **laptop** in Android Studio. Kaggle has no role here. Before starting, download the model files from Step 13.8 onto your laptop.

### Step 14.1 — Create a New Android Project in Android Studio

1. **New Project** → **Empty Views Activity** → **Next**.
2. **Name:** `EmailIntelligence`. **Package:** `com.samsung.emailintelligence`. **Save location:** your locally cloned `context-aware-email-intelligence/android_app/`. **Language:** Kotlin. **Min SDK:** API 26.
3. **Finish**, wait for Gradle sync. **[VERIFY]** "Gradle sync successful."

---

### Step 14.2 — Add Required Dependencies to build.gradle

```
implementation 'org.tensorflow:tensorflow-lite:2.13.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
implementation 'com.google.code.gson:gson:2.10.1'
implementation 'com.google.android.material:material:1.11.0'
```
**Sync Now**. **[VERIFY]** No errors.

---

### Step 14.3 — Add the TFLite Model Files (Downloaded From Kaggle) to App Assets

1. Right-click `app` → **New** → **Directory** → `src/main/assets`.
2. Copy the `.tflite` files (from Step 13.8) into `android_app/app/src/main/assets/`.
3. **[VERIFY]** Both files appear under `app/src/main/assets`.

---

### Step 14.4 — Configure AndroidManifest.xml

Add `<uses-permission android:name="android.permission.INTERNET" />` only if using optional network features. Set `android:allowBackup="false"` (privacy requirement).

---

## Phase 15 — Android App Module Development (Local Laptop Required)

### Step 15.1 — Design the UI Layout

`app/src/main/res/layout/activity_main.xml`: `ScrollView` > `LinearLayout` with title, multiline email-input `EditText`, **Analyze** `Button`, hidden `ProgressBar`, and result sections for Summary, Detected Intent, Key Entities.

---

### Step 15.2 — Create the Kotlin Source Files

Under `com.samsung.emailintelligence`: `EmailPreprocessor.kt`, `InferenceEngine.kt`, `OutputFormatter.kt`, `PrivacyManager.kt` (empty stubs initially).

---

### Step 15.3 — Plan the Threading Architecture

```
Button click (Main Thread)
    └─ Show ProgressBar (Main Thread)
    └─ Launch Coroutine (IO Dispatcher) {
           val cleanedText = EmailPreprocessor.clean(inputText)
           val summary    = InferenceEngine.summarize(cleanedText)
           val intent     = InferenceEngine.classifyIntent(cleanedText)
           val entities   = InferenceEngine.extractEntities(cleanedText)

           withContext(Main) {
               Hide ProgressBar
               Display summary, intent, entities in UI
           }
       }
```

---

### Step 15.4 — Implement and Test UI Progressively

Build/run in the emulator after each increment: (1) input+button+Logcat, (2) ProgressBar show/hide, (3) `EmailPreprocessor` wired in, (4) intent inference, (5) summarization inference, (6) entity extraction.

---

### Step 15.5 — Connect a Physical Device and Do a Device Run

Connect via USB → select device → **Run**. Test: paste email → Analyze → verify output, no crashes.

---

## Phase 16 — Android App Testing and Performance Profiling (Local Laptop Required)

### Step 16.1 — Open the Android Profiler

**View** → **Tool Windows** → **Profiler**, attach to your running app session.

---

### Step 16.2 — Measure Inference Latency

Test 10 emails of varying length; instrument `InferenceEngine.kt` with `System.currentTimeMillis()` + `Log.d("PERF", ...)` around inference calls.

| Email # | Email Length (words) | Inference Time (ms) | Within Target (<3000ms)? |
|---|---|---|---|
| 1 | X | X | ☐ |

---

### Step 16.3 — Measure RAM Usage

Watch the Profiler's memory graph during inference on a long email. **[VERIFY]** Peak RAM < 300 MB.

---

### Step 16.4 — Measure APK Size

**Build** → **Build Bundle(s) / APK(s)** → **Build APK(s)**. Check size in `android_app/app/build/outputs/apk/debug/`. **[VERIFY]** < 100 MB.

---

### Step 16.5 — Document All Performance Results

| Metric | Target | Measured | Pass/Fail |
|---|---|---|---|
| Average inference time (ms) | < 3000 ms | X ms | ☐ |
| Peak RAM usage (MB) | < 300 MB | X MB | ☐ |
| Model file size (MB) | < 50 MB | X MB | ☐ |
| APK size (MB) | < 100 MB | X MB | ☐ |

---

## Phase 17 — Usability Testing

### Step 17.1 — Prepare a Usability Test Plan

`docs/usability_test_plan.md`: 3–5 testers, tasks, observation notes, questionnaire. Tasks: open app → paste sample email → Analyze → describe output → rate summary/intent/entities.

### Step 17.2 — Recruit Testers

3–5 people unfamiliar with the app; neutral briefing; 15-minute slots.

### Step 17.3 — Conduct Usability Tests

Observe silently, then have testers fill out the rating questionnaire.

### Step 17.4 — Document Usability Findings

`docs/usability_test_results.md`: response table, average scores, confusion points, planned UI changes.

---

## Phase 18 — Final Documentation and Deliverables

### Step 18.1 — Write the Performance Evaluation Report

`docs/performance_evaluation_report.md`: intro, ROUGE results (Phase 12), intent metrics, NER results, Android performance (Phase 16), usability results (Phase 17), improvement areas, conclusion.

### Step 18.2 — Write the Privacy and Security Note

`docs/privacy_security_note.md`: on-device processing, permissions, retention policy, no-logging implementation (`PrivacyManager.kt`), future considerations.

### Step 18.3 — Finalize Source Code and Clean Up

Clean notebooks (remove debug cells, confirm top-to-bottom runs), clean Android app (remove content-logging `Log.d` calls), write project-root `README.md` covering Kaggle setup, notebook run order + dataset dependencies, and local Android build instructions.

### Step 18.4 — Package All Final Artifacts

| # | Deliverable | Location | Status |
|---|---|---|---|
| D-1 | Curated preprocessed dataset | Kaggle Dataset `email-nlp-processed-data` | ☐ |
| D-2 | Baseline NLP pipeline notebook | `notebooks/05_NLP_Pipeline_Baseline.ipynb` (GitHub) | ☐ |
| D-3 | Trained summarization model | Kaggle Dataset `email-nlp-summarization-model` | ☐ |
| D-3 | Trained intent classifier | Kaggle Dataset `email-nlp-intent-classifier` | ☐ |
| D-4 | Android app (source + APK) | `android_app/` (local + GitHub) | ☐ |
| D-5 | Performance evaluation report | `docs/performance_evaluation_report.md` | ☐ |
| D-6 | Privacy and security note | `docs/privacy_security_note.md` | ☐ |
| D-7 | All notebooks (source code) | `notebooks/` (GitHub) | ☐ |
| D-7 | Mobile model files | Kaggle Dataset `email-nlp-mobile-models` + `android_app/app/src/main/assets/` | ☐ |
| D-7 | Final presentation | `docs/final_presentation.pptx` | ☐ |
| D-7 | Test cases | `tests/` | ☐ |

### Step 18.5 — Prepare the Final Presentation

Slides: title, problem, solution/architecture (Kaggle training + local Android build), datasets, NLP pipeline, model/training details, evaluation results, app demo screenshots, challenges (incl. Kaggle GPU quota/session ephemerality), future work, Q&A. Save to `docs/final_presentation.pptx`.

### Step 18.6 — Record the Demo Video (Optional but Recommended)

Under 3 minutes: open app → paste email → Analyze → show summary/intent/entities → narrate. Save to `docs/demo_video.mp4`.

### Step 18.7 — Final Git Push — All Deliverables

```
git add .
git commit -m "Final deliverables - all documentation, notebooks, and Android project"
git push origin main
```
**[VERIFY]** Everything visible on GitHub; all three Kaggle Datasets set to at least **Private** and not deleted. Share repo/dataset links with mentors Dr. Naveenkumar J and Dr. Joshva Devadas T.

---

## Appendix A — Processed Data Folder Reference

```
Processed_Data/
└── processed/
    ├── Cleaned data/
    │   ├── enron_cleaned.csv          → columns: file, cleaned_body
    │   ├── spamassassin_cleaned.csv   → columns: file, cleaned_body, label
    │   └── intent_labeled.csv         → columns: file, subject, cleaned_body, intent_label
    ├── train/
    │   ├── enron_train.csv
    │   ├── spamassassin_train.csv
    │   └── intent_train.csv
    ├── val/
    │   ├── enron_val.csv
    │   ├── spamassassin_val.csv
    │   └── intent_val.csv
    └── test/
        ├── enron_test.csv
        ├── spamassassin_test.csv
        └── intent_test.csv
```

**Intent labels** (`intent_label` — numeric 0–5): 0 REQUEST, 1 FOLLOW_UP, 2 INFORMATION, 3 ACKNOWLEDGEMENT, 4 COMPLAINT, 5 INVITATION.
**Spam labels** (`label`): `0` = ham, `1` = spam.

> Summarization CSVs are not part of `Processed_Data` — summarization cells load **EMAILSUM** directly from `/kaggle/input/datasets/prishabhkumar/emailsum/`, per Phase 8.

---

## Appendix B — Kaggle Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError` for CSV paths | Run `!ls /kaggle/input/` and update `INPUT_ROOT` slug |
| CUDA out of memory | Reduce `per_device_train_batch_size` to 4 or 8 |
| EMAILSUM JSON files not found | Run `!ls -R /kaggle/input/datasets/prishabhkumar/emailsum/` and confirm the path matches `Avocado/summaries/EmailSum_data.json` (Step 4.3) |
| EMAILSUM record field names differ | Re-run the `peek()` printout in Step 4.3 and update `THREAD_TEXT_FIELD` / `SHORT_SUMMARY_FIELD` / `LONG_SUMMARY_FIELD` (Step 8.4) |
| EMAILSUM has no thread-text field | The public release is summaries-only (see Step 4.3 note) — rely on BC3 for real training pairs, or join thread IDs against your own licensed Avocado copy |
| Session timeout (9h limit) | Save checkpoints each epoch; re-run from last checkpoint |
| Can't find trained models later | Always run the "package outputs" cell (Phase 13.8) and download `trained_models.zip` |
| spaCy model missing | Re-run `!python -m spacy download en_core_web_sm` |

**Recommended notebook split** (optional, if you'd rather not run one giant notebook):

| Notebook | Covers | Roadmap Phase |
|----------|-------|----------------|
| `01_Data_Exploration` | Setup + load + explore | Phase 5 |
| `02_NLP_Baseline` | NER + LDA | Phase 9 |
| `03_Intent_Training` | Intent classifier train + eval | Phases 10, 12 |
| `04_Summarization_Training` | Summarization train + eval | Phases 11, 12 |
| `05_Pipeline_Demo` | End-to-end demo + results summary | Phase 12 |

---

## Appendix C — Full Ordered Cell List (for copy-paste)

If you'd rather build one linear notebook instead of jumping phase-by-phase, paste these in order (numbers match the 📎 code blocks above):

0. Markdown — Title & overview *(Step 1.1)*
1. Install dependencies *(Step 1.3)*
2. Imports and configuration *(Step 1.9)*
3. Verify dataset files exist *(Step 4.5)*
4. Load all processed CSVs *(Step 5.1)*
5. Explore Enron cleaned data *(Step 5.2)*
6. Explore SpamAssassin data *(Step 5.3)*
6a. Core preprocessing function library *(Step 6.2)*
6b. Test the pipeline on 10 sample emails *(Step 6.3)*
6c. Process the full Enron dataset *(Step 6.4)*
6d. Process the SpamAssassin dataset *(Step 6.5)*
6e. Process the BC3 dataset *(Step 6.6)*
6f. Train / validation / test split *(Step 6.7)*
7. Explore intent-labeled data *(Step 7.4)*
8. Markdown — NLP baseline section *(Step 9.1)*
9. spaCy NER baseline *(Step 9.3)*
10. Gensim LDA baseline *(Step 9.4)*
11. Markdown — Intent classifier training *(Step 10.1)*
12. Prepare intent datasets for Trainer *(Step 10.3)*
13. Define intent metrics & load model *(Step 10.4/10.5)*
14. Train intent classifier *(Step 10.6)*
15. Evaluate intent classifier on test set *(Step 12.3)*
16. Markdown — Summarization training *(Step 11.1)*
17. Load EMAILSUM from Kaggle dataset *(Step 8.2 / 4.3)*
18. Prepare summarization data *(Step 8.4)*
19. Tokenize summarization data *(Step 11.3/11.4)*
20. Train summarization model *(Step 11.5)*
21. Quick summarization demo *(Step 11.6)*
22. ROUGE evaluation on test set *(Step 12.2)*
23. End-to-end pipeline demo *(Step 12.5)*
24. Save & package outputs for download *(Step 13.8)*
25. Markdown — Results summary template *(Step 12.5)*

After Cell 25, everything remaining (Phases 13–18) is either mobile conversion inside Kaggle (Phase 13) or work that moves to your laptop (Phases 14–18) — see those phases above for the full steps.