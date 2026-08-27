# Implementation Roadmap (Kaggle Edition)
## Context-Aware Email Intelligence: A Unified NLP Approach to Information Synthesis

**Project Type:** Samsung Collaborative Research Project
**Document Version:** 1.0 — Kaggle Development Track
**Date:** June 2026
**Scope:** Every non-code step required to complete this project from zero to final deliverable, in exact execution order, using **Kaggle Notebooks as the primary Python/ML development environment**.

---

## How to Use This Document

Every step in this roadmap is numbered and sequenced. Do not skip a step or reorder steps — each step's output is an input to the next. Steps marked **[ONE-TIME]** only need to be done once per Kaggle account or notebook. Steps marked **[EVERY SESSION]** must be repeated each time you open a fresh Kaggle Notebook session, because Kaggle sessions are ephemeral. Steps marked **[VERIFY]** include a quick sanity check you must confirm before moving on. Steps marked **[DECISION]** require you to make a choice that affects later steps.

> **Important scope note:** All Python, NLP, data preparation, model training, and model evaluation work in this roadmap happens on **Kaggle Notebooks** — not on your laptop. The one exception is **Android app development (Phases 14–16)**: Kaggle Notebooks run in a browser-based Linux container and cannot run Android Studio, an emulator, or a physical-device debugger. That part of the project still has to happen on a laptop. Everywhere else, "your machine" in earlier drafts of this roadmap now means "your Kaggle Notebook."

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

---

## Phase 0 — Pre-Work: Accounts and Kaggle Readiness Check

### Step 0.1 — Verify Kaggle Access Instead of Local Hardware [ONE-TIME] [VERIFY]

Because training and preprocessing run on Kaggle's servers, your laptop's RAM/GPU no longer gate the project. You only need a machine that can run a modern browser.

1. Confirm you have a stable internet connection (Kaggle Notebooks stream everything to/from the browser).
2. Confirm your browser is up to date (Chrome or Firefox recommended — Kaggle's notebook editor is JavaScript-heavy).
3. **[VERIFY]** Go to `https://www.kaggle.com` and confirm the site loads normally.

---

### Step 0.2 — Create a GitHub Account (if you don't have one) [ONE-TIME]

You'll still use GitHub as the permanent, version-controlled home for your code, notebooks (as `.ipynb` exports), and documentation — Kaggle sessions are temporary, GitHub is not.

1. Open your browser. Go to `https://github.com`.
2. Click **Sign up** in the top-right corner.
3. Enter your email address → click **Continue**.
4. Create a password → click **Continue**.
5. Choose a username (e.g., `rishabh-email-nlp`) → click **Continue**.
6. Solve the verification puzzle.
7. Check your email for the GitHub verification code → enter it on the page.
8. On the welcome screen, click **Skip personalization** to go directly to your dashboard.
9. **[VERIFY]** You should be on `github.com` and see your username in the top-right corner.

---

### Step 0.3 — Create a New GitHub Repository for This Project [ONE-TIME]

1. On your GitHub dashboard, click the green **New** button.
2. In **Repository name**, type: `context-aware-email-intelligence`.
3. Set visibility to **Private**.
4. Check **Add a README file**.
5. In **Add .gitignore**, select **Python**.
6. Click **Create repository**.
7. Click the green **Code** button → copy the HTTPS URL (`https://github.com/YOUR_USERNAME/context-aware-email-intelligence.git`). Save it — you'll need it in Phase 3.
8. Click your profile icon → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**. Give it `repo` scope, set an expiry, and click **Generate token**. **Copy the token immediately and save it somewhere safe** — you will paste it into a Kaggle Secret in Step 1.6, and GitHub will never show it to you again.

---

### Step 0.4 — Create a Kaggle Account [ONE-TIME]

Kaggle is now your primary development environment, not just a dataset source.

1. Go to `https://www.kaggle.com`.
2. Click **Register** → **Register with Google** (easiest) or fill in the email/password form.
3. Complete email verification if required.
4. Click your profile icon (top-right) → **Settings**.
5. Under **Phone Verification**, verify your phone number. **This step is required to enable GPU accelerators on notebooks** — without it, your notebooks will be CPU-only.
6. **[VERIFY]** Go to **Settings** → confirm your account shows as phone-verified.

---

### Step 0.5 — Create a Hugging Face Account [ONE-TIME]

You will pull pre-trained models (DistilBERT, T5) and the EMAILSUM dataset from Hugging Face directly inside Kaggle Notebooks.

1. Go to `https://huggingface.co`.
2. Click **Sign Up** → fill in name, email, password → click **Next**.
3. Verify your email via the link sent to your inbox.
4. Click your profile icon → **Settings** → **Access Tokens**.
5. Click **New token** → name it `email-intelligence-project` → role **Read** → **Generate a token**.
6. Copy the token and save it somewhere secure — you'll paste it into a Kaggle Secret in Step 1.6.

---

### Step 0.6 — Understand Kaggle's GPU Quota [ONE-TIME]

Kaggle replaces the "do you have a local GPU / use Colab" decision from a laptop-based plan — every notebook can request a free GPU.

1. Kaggle currently grants roughly **30 GPU-hours per week**, resettable weekly (verify the current quota in **Notebook Settings** since Kaggle adjusts this periodically).
2. Available accelerators typically include **GPU T4 x2** and **GPU P100** — you'll pick one per notebook session in Phase 1.
3. Budget your quota deliberately: preprocessing and data exploration should run **CPU-only** (no GPU needed); reserve GPU time for Phases 10–11 (model training) and Phase 12 (evaluation, if needed).
4. **[VERIFY]** No action needed yet — just note this budget before you start burning GPU hours on notebooks that don't need them.

---

### Step 0.7 — Plan Your Work Split (Team Coordination)

Before writing a single line of anything, sit down with your teammate (Ms. Samriddhi) and assign ownership. Suggested division:

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

Since Kaggle Notebooks aren't simultaneously multi-editable the way Google Docs is, agree on a **turn-taking or fork-based workflow** (see Step 3.1) so you don't overwrite each other's work.

Document this split in a shared notes document (Google Docs or a Markdown file in your GitHub repo).

---

## Phase 1 — Kaggle Notebook Environment Setup

### Step 1.1 — Create Your Project Notebook [ONE-TIME]

1. Log into Kaggle. Click **+ Create** (top-left) → **New Notebook**.
2. In the notebook editor, click the notebook's title at the top (default is something like `notebookXXXX`) and rename it to `email-nlp-project`.
3. On the right-hand settings panel, under **Language**, confirm **Python** is selected.
4. **[VERIFY]** The notebook opens with an empty first code cell and a Kaggle-managed Python kernel already running — no local install required.

---

### Step 1.2 — Confirm What's Already Installed [VERIFY]

Kaggle Notebooks ship with a large pre-installed data science stack, so most of "Phase 1" on a laptop is unnecessary here.

1. In the first cell, run:
   ```python
   import numpy, pandas, sklearn, matplotlib, seaborn
   print(numpy.__version__, pandas.__version__, sklearn.__version__)
   ```
2. **[VERIFY]** This should run with no `ModuleNotFoundError`. `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` all come pre-installed on the Kaggle Python image.

---

### Step 1.3 — Install the Remaining Libraries [EVERY SESSION]

Kaggle notebook sessions are **ephemeral compute** — anything you `pip install` disappears when the session restarts. Put this in the first cell of every notebook you create so it self-installs on each fresh session.

```python
!pip install -q nltk spacy gensim beautifulsoup4 rouge-score kaggle
!pip install -q transformers datasets sentencepiece accelerate evaluate
```

> **Note on deep learning frameworks:** Kaggle's base image already includes both **TensorFlow** and **PyTorch** pre-installed and GPU-enabled — you do not need to `pip install` either one. Just `import tensorflow` or `import torch` directly.

> **Recommendation for beginners:** Plan around the TensorFlow Lite conversion path (Phase 13) — it's better documented for Android, and TensorFlow is already available with no setup.

**[VERIFY]** Re-run `!pip list 2>/dev/null | grep -E "nltk|spacy|gensim|transformers|rouge"` and confirm all packages appear.

---

### Step 1.4 — Enable Internet Access on the Notebook [ONE-TIME PER NOTEBOOK]

By default, some Kaggle notebook configurations restrict internet access (required for competition submission notebooks). You need it on for `pip install`, Hugging Face downloads, and GitHub pushes.

1. In the notebook editor, open the right-hand settings panel (click the **>** arrow if collapsed).
2. Under **Session options**, find the **Internet** toggle and switch it **On**.
3. Kaggle may ask you to verify your phone number if you haven't already (see Step 0.4).
4. **[VERIFY]** Run `!ping -c 2 google.com` in a cell — you should see replies, not a network error.

---

### Step 1.5 — Enable the GPU Accelerator [DECISION] [EVERY SESSION]

1. In the right-hand settings panel, find **Accelerator**.
2. Select **GPU T4 x2** (a good default) or **GPU P100** depending on availability and your task:
   - Use **CPU** (no accelerator) for Phases 5–9 (exploration, preprocessing, baseline NLP) — save your GPU quota.
   - Switch to **GPU T4 x2** or **P100** only for Phases 10–12 (training and evaluation).
3. **[VERIFY]** Run:
   ```python
   import torch
   print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
   ```
   This should print `True` and a device name (e.g., `Tesla T4`) when the accelerator is enabled.

---

### Step 1.6 — Store Your Tokens as Kaggle Secrets [ONE-TIME]

Never paste your GitHub token or Hugging Face token directly into notebook cells — they'd be saved in plaintext in your notebook's version history. Kaggle Secrets keep them out of the visible notebook.

1. In the notebook editor, click **Add-ons** (top menu) → **Secrets**.
2. Click **Add a new secret**.
3. Label: `GITHUB_TOKEN` → Value: paste the personal access token from Step 0.3.
4. Click **Add a new secret** again. Label: `HF_TOKEN` → Value: paste the Hugging Face token from Step 0.5.
5. Toggle both secrets **On** for this notebook (Kaggle requires you to explicitly attach secrets per-notebook).
6. In a code cell, load them:
   ```python
   from kaggle_secrets import UserSecretsClient
   secrets = UserSecretsClient()
   GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
   HF_TOKEN = secrets.get_secret("HF_TOKEN")
   ```
7. **[VERIFY]** `print(len(GITHUB_TOKEN), len(HF_TOKEN))` should print two nonzero numbers, not an error.

---

### Step 1.7 — Authenticate the Hugging Face CLI [EVERY SESSION]

```python
!huggingface-cli login --token $HF_TOKEN
```

**[VERIFY]** Run `!huggingface-cli whoami` — it should print your Hugging Face username.

---

### Step 1.8 — Download NLTK and spaCy Resources [EVERY SESSION]

These don't persist between sessions unless you save them to a Kaggle Dataset (optional optimization — see the note at the end of this step). For now, re-download each session:

```python
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

!python -m spacy download en_core_web_sm -q
```

**[VERIFY]** `import spacy; nlp = spacy.load("en_core_web_sm"); print("ok")` should print `ok` with no errors.

> **Optional time-saver:** Once you've confirmed this works, you can package the downloaded NLTK/spaCy data as a private Kaggle Dataset and re-attach it via **Add Data** in future sessions instead of re-downloading every time.

---

## Phase 2 — Android Development Environment Setup (Local Laptop Required)

> **This phase cannot be done on Kaggle.** Kaggle Notebooks run in a headless Linux container with no display server, no Android SDK, no emulator support, and no USB passthrough for a physical device. Android Studio, the emulator, and on-device debugging all require your laptop. Everything else in this roadmap stays on Kaggle; this phase is the deliberate exception.

### Step 2.1 — Install Java Development Kit (JDK) [ONE-TIME, LOCAL]

1. Go to `https://www.oracle.com/java/technologies/downloads/`.
2. Click the tab for **JDK 17** (or the latest LTS version shown).
3. Download the installer for your OS and run it, accepting all defaults.
4. **[VERIFY]** Open a new terminal and type `java -version`. You should see `java version "17.x.x"`.

---

### Step 2.2 — Download and Install Android Studio [ONE-TIME, LOCAL]

1. Go to `https://developer.android.com/studio`.
2. Click **Download Android Studio**, accept the terms, and download (~1 GB).
3. Run the installer (Windows `.exe` with defaults, or macOS `.dmg` dragged to Applications).
4. Follow the first-time setup wizard: **Standard** install → choose theme → **Finish** → let it download the SDK, emulator, and build tools (10–30 minutes).
5. **[VERIFY]** Android Studio opens to the Welcome screen with an option to create a new project.

---

### Step 2.3 — Install Additional Android SDK Components [ONE-TIME, LOCAL]

1. **Tools** → **SDK Manager**.
2. Under **SDK Platforms**: check **Android 14.0 (API 34)** and **Android 8.0 (API 26)** → **Apply**.
3. Under **SDK Tools**: confirm **Android SDK Build-Tools**, **Android Emulator**, **Android SDK Platform-Tools**, and **Intel x86 Emulator Accelerator (HAXM)** are checked → **Apply**.
4. Click **OK**.

---

### Step 2.4 — Create an Android Virtual Device (Emulator) [ONE-TIME, LOCAL]

1. **Tools** → **Device Manager** → **Create Device**.
2. Select **Phone** → **Pixel 6** → **Next**.
3. Under **Recommended** system images, pick **API 34** (download if needed) → **Next**.
4. Name it `Pixel6_API34`, set RAM to at least 2048 MB → **Finish**.
5. **[VERIFY]** Click the green Play button next to `Pixel6_API34` — it should boot to an Android home screen in 1–2 minutes.

---

### Step 2.5 — Enable Developer Mode on a Physical Android Device (Optional, LOCAL)

1. **Settings** → **About phone** → tap **Build number** 7 times.
2. **Settings** → **Developer options** → toggle **On** → enable **USB debugging**.
3. Connect via USB → tap **Allow** on the device prompt.
4. **[VERIFY]** In Android Studio, **Tools** → **Device Manager** → **Physical** tab shows your device.

---

## Phase 3 — Project Structure and Repo Sync Inside Kaggle

### Step 3.1 — Decide Your Kaggle Collaboration Model [DECISION] [ONE-TIME]

Kaggle Notebooks support one primary editor at a time. Choose one:

- **Shared-notebook model:** Both teammates edit the same Kaggle Notebook, taking turns and committing/pushing (Step 3.5) between sessions.
- **Fork model:** Each teammate keeps their own copy of the notebook (**File** → **Copy and Edit**), and you merge finished code back into GitHub, treating GitHub — not Kaggle — as the single source of truth.

Given the team-split table in Step 0.7, the **fork model** is usually easier: it matches "one person owns preprocessing, another owns summarization training."

---

### Step 3.2 — Clone Your GitHub Repository Into the Kaggle Session [EVERY SESSION]

Kaggle's working directory (`/kaggle/working/`) is wiped between sessions, so pull your repo down fresh each time.

```python
import os
os.chdir("/kaggle/working")
!git clone https://{GITHUB_TOKEN}@github.com/YOUR_USERNAME/context-aware-email-intelligence.git
os.chdir("/kaggle/working/context-aware-email-intelligence")
!git config user.email "you@example.com"
!git config user.name "Your Name"
```

**[VERIFY]** `!ls` should show `README.md` and `.gitignore`.

---

### Step 3.3 — Create the Full Directory Structure [ONE-TIME, then re-created each session via Step 3.2's clone]

Run this once locally in the cloned repo (via the notebook) to establish the structure, then commit it so future clones already have it:

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

**[VERIFY]** `!find . -maxdepth 2 -type d` should list all the folders above.

> **Note:** Raw datasets themselves live in **Kaggle's `/kaggle/input/` directory** (read-only, attached via Add Data — see Phase 4), not inside `data/raw/` in your git repo. `data/raw/` in the repo is only for small reference files or symlink notes; the large files never get committed.

---

### Step 3.4 — Understand Kaggle's Two Storage Areas [ONE-TIME, conceptual]

This is the single biggest mental shift from a laptop workflow:

| Path | Persistence | Purpose |
|---|---|---|
| `/kaggle/input/` | Read-only, persists across sessions once attached | Datasets you've added via **Add Data** |
| `/kaggle/working/` | Read-write, **wiped when the session ends** unless you click **Save Version** | Your active repo clone, notebooks, intermediate files |
| Kaggle Datasets (output) | Persists indefinitely | Where you publish trained models/processed data so later notebooks can read them back in via `/kaggle/input/` |
| GitHub repo | Persists indefinitely | Code, notebooks (as files), and documentation — the actual deliverable record |

Anything you want to survive between sessions must end up in **either a Kaggle Dataset (for large binary artifacts like models) or your GitHub repo (for code and docs)** — never left sitting only in `/kaggle/working/`.

---

### Step 3.5 — Push the Skeleton Structure to GitHub [EVERY SESSION, as needed]

```python
!git add .
!git commit -m "Initial project structure setup"
!git push origin main
```

**[VERIFY]** Visit `https://github.com/YOUR_USERNAME/context-aware-email-intelligence` in your browser and confirm the folder structure appears.

---

### Step 3.6 — Save a Notebook Version to Persist Working Files [EVERY SESSION]

Before closing a Kaggle session, click **Save Version** (top-right) → **Save & Run All (Commit)**. This snapshots `/kaggle/working/` at that point in time so you can reopen it later, even though a *fresh* session still starts from a clean clone per Step 3.2.

---

## Phase 4 — Dataset Acquisition on Kaggle

### Step 4.1 — Attach the Enron Email Dataset via Add Data [ONE-TIME PER NOTEBOOK]

Because you're already inside Kaggle, this is now a UI action instead of a CLI download.

1. In the notebook editor, click **Add Data** (right-hand panel).
2. In the search box, type `enron email dataset`.
3. Select the dataset **Enron Email Dataset** (`wcukierski/enron-email-dataset`) → click **Add**.
4. Kaggle mounts it at `/kaggle/input/enron-email-dataset/`.
5. **[VERIFY]** `!ls /kaggle/input/enron-email-dataset/` should show a `maildir/` folder and/or `emails.csv`.

---

### Step 4.2 — Request and Attach the BC3 Corpus

The BC3 Corpus still requires an academic access request — that part doesn't change.

1. Go to `https://www.cs.ubc.ca/cs/research/lci/facilities/bc3corpus.html` and submit the access request form (name, institution, email, purpose: "Academic research project on email thread summarization and NLP, under Samsung Research Collaboration Program.").
2. **While waiting**, continue with the other datasets and Phase 5 using Enron and SpamAssassin.
3. **Once you receive the download link:** download the archive to your laptop first (Kaggle can't fetch it directly since it needs email-gated access).
4. On Kaggle, click **+ Create** → **New Dataset** → **Upload** the archive from your laptop. Name it something like `bc3-corpus`, set visibility to **Private**, and click **Create**.
5. In your project notebook, click **Add Data** → search **My Datasets** → attach `bc3-corpus`.
6. **[VERIFY]** `!ls /kaggle/input/bc3-corpus/` shows the XML file(s), e.g. `bc3corpus.1.0.xml`.

**Alternative if BC3 access takes too long:** use only Enron + EMAILSUM for training and treat BC3 as a validation benchmark once it arrives.

---

### Step 4.3 — Load the EMAILSUM Dataset via Hugging Face [EVERY SESSION]

No manual download needed — pull it directly inside the notebook (Internet must be on, per Step 1.4):

```python
from datasets import load_dataset
emailsum = load_dataset("Yale-LILY/emailsum")
print(emailsum)
```

**[VERIFY]** This should print `DatasetDict` with `train`, `validation`, and `test` splits and nonzero row counts.

---

### Step 4.4 — Attach or Download the SpamAssassin Public Corpus

**Option A — Check Kaggle first (recommended):** search **Add Data** for `spamassassin`. Several public mirrors of this corpus already exist as Kaggle Datasets — attach one if it matches the expected `easy_ham` / `hard_ham` / `spam` structure.

**Option B — Download directly inside the notebook** if no suitable mirror exists:
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

**[VERIFY]** `!ls /kaggle/working/spamassassin_raw` should show `easy_ham/`, `hard_ham/`, `spam/`, `spam_2/` folders. Since this lives in `/kaggle/working/`, package it as a Kaggle Dataset (Step 4.2's upload flow, but via **New Dataset → Upload files from notebook output**) if you don't want to re-download it every session.

---

### Step 4.5 — Verify All Dataset Attachments

| Dataset | Location | Expected Content | Status |
|---|---|---|---|
| Enron | `/kaggle/input/enron-email-dataset/` | `maildir/` OR `emails.csv` | ☐ |
| BC3 | `/kaggle/input/bc3-corpus/` | Annotated XML thread file(s) | ☐ |
| EMAILSUM | Loaded via `datasets` library | train/val/test splits | ☐ |
| SpamAssassin | `/kaggle/input/...` or `/kaggle/working/spamassassin_raw/` | `easy_ham/`, `hard_ham/`, `spam/` | ☐ |

---

### Step 4.6 — Keep Raw Datasets Out of Git

Raw datasets live in `/kaggle/input/` or `/kaggle/working/`, never inside your cloned git repo, so there's no risk of accidentally committing a 1.7 GB file. Still add a safety net to `.gitignore`:

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

## Phase 5 — Dataset Exploration and Inventory

### Step 5.1 — Create an Exploration Notebook [ONE-TIME]

1. From your Kaggle project page, click **+ Create** → **New Notebook** (or duplicate your setup notebook via **Copy and Edit**).
2. Rename it `01-dataset-exploration`.
3. Repeat Steps 1.3–1.8 (installs, secrets, NLTK/spaCy downloads) and Step 3.2 (clone repo) at the top — every notebook needs its own environment bootstrap since Kaggle sessions don't share state across notebooks.
4. Set **Accelerator** to **None (CPU)** for this notebook — exploration doesn't need a GPU; save your quota.

---

### Step 5.2 — Explore the Enron Dataset Structure

Each numbered item below is a separate notebook cell — run with `Shift+Enter`:

**Cell 1 — Check `emails.csv`:** if present at `/kaggle/input/enron-email-dataset/emails.csv`, load with `pandas` and note the row count.

**Cell 2 — Explore `maildir`:** use `os.walk` on `/kaggle/input/enron-email-dataset/maildir/` to count individual email files.

**Cell 3 — Open one sample email:** use Python's `email` library to parse one file (e.g., `maildir/allen-p/inbox/1.`), print From/To/Date/Subject headers and body.

**Cell 4 — Count unique senders.**

**Markdown cell:** record total email count, average length, notable observations.

---

### Step 5.3 — Explore the SpamAssassin Dataset Structure

**Cell 5:** count files in `easy_ham/`, `hard_ham/`, `spam/`.

**Cell 6:** open and compare one spam and one ham email.

---

### Step 5.4 — Explore the BC3 Dataset (Once Attached)

**Cell 7:** parse the BC3 XML with `xml.etree.ElementTree`, count threads (~40 expected), print one thread's emails + summary + speech-act labels.

---

### Step 5.5 — Explore the EMAILSUM Dataset

**Cell 8:** inspect the `emailsum` object loaded in Step 4.3 — print the first record's field names, split sizes, and length statistics.

---

### Step 5.6 — Commit Your Exploration Notebook

1. Kaggle auto-saves your `.ipynb` in the notebook environment; also export a copy into your repo clone: **File** → **Download Notebook** and add it to `notebooks/`, or programmatically copy from the Kaggle notebook's own path.
2. ```python
   !cp /kaggle/working/*.ipynb notebooks/01_Dataset_Exploration.ipynb 2>/dev/null || true
   !git add notebooks/01_Dataset_Exploration.ipynb
   !git commit -m "Add dataset exploration notebook"
   !git push origin main
   ```
3. Click **Save Version** in Kaggle so this notebook's state is preserved too.

---

## Phase 6 — Dataset Preprocessing and Curation

### Step 6.1 — Create a New Notebook for Preprocessing

Create `02-preprocessing-pipeline` on Kaggle, bootstrap it (Steps 1.3–1.8, 3.2), set **Accelerator: None**.

---

### Step 6.2 — Plan and Document the Preprocessing Steps

In a markdown cell:

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

---

### Step 6.3 — Test Preprocessing on 10 Sample Emails

1. Select 10 emails from `/kaggle/input/enron-email-dataset/maildir/` across different users/folders.
2. Run your cleaning code on them in the notebook.
3. Inspect input vs. cleaned output side-by-side; check signature removal, leftover HTML, over-aggressive stopword filtering.
4. Adjust logic as needed.

---

### Step 6.4 — Process the Full Enron Dataset

1. Apply the pipeline to the full dataset (reading from `/kaggle/input/`, never modifying it — it's read-only anyway).
2. Save cleaned output with columns `email_id`, `sender`, `date`, `subject`, `cleaned_body` to `/kaggle/working/context-aware-email-intelligence/data/processed/enron_cleaned.csv`.
3. Log counts: processed, skipped (parse errors), empty-after-cleaning.

---

### Step 6.5 — Process the SpamAssassin Dataset

Same pipeline, add `label` column (0 = ham, 1 = spam), save to `data/processed/spamassassin_cleaned.csv`.

---

### Step 6.6 — Process the BC3 Dataset

Parse threads + summaries, save to `data/processed/bc3_threads.json`.

---

### Step 6.7 — Create the Train / Validation / Test Split

1. Use `sklearn.model_selection.train_test_split`: 15% test, ~18% of remainder as val (≈15% of total), 70% train.
2. Save each split under `data/processed/train/`, `data/processed/val/`, `data/processed/test/`.
3. **[VERIFY]** Split counts sum to the total.

---

### Step 6.8 — Publish Processed Data as a Kaggle Dataset [DECISION]

Since `/kaggle/working/` is wiped between sessions, and re-running the full Enron cleaning pipeline every session is slow, publish your processed CSVs as a private Kaggle Dataset once they're ready:

1. Click **Save Version** → after it finishes, go to the notebook's **Output** tab.
2. Click **New Dataset** from the output files, name it `email-nlp-processed-data`, set to **Private**.
3. In future notebooks, attach it via **Add Data** → **My Datasets** instead of regenerating everything.

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

Bootstrap as before. Attach `email-nlp-processed-data` (Step 6.8) via **Add Data** instead of re-running Phase 6.

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

1. Randomly sample 600 emails from the cleaned Enron CSV (100 per class target).
2. Since Kaggle Notebooks aren't ideal for manual spreadsheet-style labeling, export the sample to CSV, **download it to your laptop** (**File** → **Download** on the output file, or `data/processed/intent_labeling_sample.csv` from the Output tab), and label it in Google Sheets or Excel as before.
3. Split labeling 300/300 between teammates; cross-validate 60 emails (10%) for agreement.
4. Re-upload the completed CSV: click **Add Data** → **Upload** on your Kaggle notebook, or commit it directly to the GitHub repo and `git pull` it back into the notebook.
5. Save as `data/processed/intent_labeled.csv`.

> **Time estimate:** ~10 hours total at 1 minute/email — this labeling step is manual regardless of environment.

---

### Step 7.4 — Validate Label Distribution

Load `intent_labeled.csv`, print per-label counts. **[VERIFY]** each class has ≥60 samples.

---

### Step 7.5 — Split Intent Data

70/15/15 split → `data/processed/train/intent_train.csv`, `.../val/intent_val.csv`, `.../test/intent_test.csv`.

---

### Step 7.6 — Commit Intent Data Preparation Work

```python
!cp /kaggle/working/*.ipynb notebooks/03_Intent_Data_Preparation.ipynb 2>/dev/null || true
!git add notebooks/03_Intent_Data_Preparation.ipynb data/processed/intent_labeled.csv
!git commit -m "Add intent classification data preparation"
!git push origin main
```

---

## Phase 8 — Summarization Data Preparation

### Step 8.1 — Create Notebook `04-summarization-data-preparation`

---

### Step 8.2 — Load the EMAILSUM Dataset via Hugging Face

Reuse the `load_dataset("Yale-LILY/emailsum")` call from Step 4.3. Print field names, split sizes, average lengths.

---

### Step 8.3 — Load BC3 Summarization Data

Load `data/processed/bc3_threads.json` (or from your `email-nlp-processed-data` Kaggle Dataset), reformat to `{input, short_summary}` matching EMAILSUM's structure. Treat BC3 (~40 threads) as extra validation data.

---

### Step 8.4 — Prepare the Combined Summarization Dataset

1. Combine EMAILSUM training data with BC3 records if desired.
2. Filter: non-empty input/summary, input ≥ 50 words, summary ≤ 150 words.
3. Save to `data/processed/train/summarization_train.csv`, `.../val/summarization_val.csv`, `.../test/summarization_test.csv`.

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

Accelerator: **None (CPU)** — this phase is about correctness, not speed.

---

### Step 9.2 — Test Preprocessing on 5 New Emails

Pick 5 unseen Enron test emails, run the cleaning pipeline, verify noise removal.

---

### Step 9.3 — Test the NER Module Baseline

Load `en_core_web_sm`, run NER on the same 5 emails, print entities by type (PERSON, ORG, DATE, GPE, MONEY). Evaluate manually for correctness, missed entities, false positives. Record findings — this informs whether `en_core_web_trf` is worth the extra download size later.

---

### Step 9.4 — Test LDA Topic Modeling on a Sample Thread

Take one 3+ message Enron thread, preprocess each message, run Gensim LDA with 3 topics, print top 5 words per topic, manually assess coherence.

---

### Step 9.5 — Document Baseline Observations

Markdown cell summarizing NER quality, LDA appropriateness, and any data surprises.

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

Bootstrap fully (Steps 1.3–1.8, 3.2). Attach `email-nlp-processed-data`.

---

### Step 10.2 — Set the Accelerator [DECISION] [EVERY SESSION]

Switch **Accelerator** to **GPU T4 x2** (or **P100**) for this notebook — this replaces the old "local GPU vs. Google Colab" decision entirely. Kaggle's free GPU quota (Step 0.6) covers this training run comfortably; no external cloud service is needed.

**[VERIFY]** `torch.cuda.is_available()` returns `True`.

---

### Step 10.3 — Load Training Data

Load `intent_train.csv` / `intent_val.csv`, print first 5 rows of each, confirm label balance.

---

### Step 10.4 — Download the Pre-Trained Base Model

`distilbert-base-uncased` downloads automatically the first time your code references it (via `AutoModel.from_pretrained`), cached inside the Kaggle session (~260 MB). Confirm the model card at `https://huggingface.co/distilbert-base-uncased` if you want details on its pretraining.

---

### Step 10.5 — Configure Training Hyperparameters

```
Model: distilbert-base-uncased
Number of labels: 6
Max sequence length: 512
Batch size: 16
Learning rate: 2e-5
Epochs: 4
Optimizer: AdamW
Evaluation: Per epoch
Checkpointing: Save best model based on validation F1
Output directory: /kaggle/working/context-aware-email-intelligence/models/checkpoints/intent_classifier/
```

---

### Step 10.6 — Run Training

1. Start training; watch per-epoch loss/metrics.
2. Note best validation F1 and its epoch.
3. **[VERIFY]** Checkpoint files exist in `models/checkpoints/intent_classifier/`.
4. Keep an eye on your GPU-hour quota (Step 0.6) — a 6-class DistilBERT classifier over 4 epochs is typically a small fraction of your weekly allowance.

---

### Step 10.7 — Save the Final Intent Classifier

1. Copy the best checkpoint to `models/final/intent_classifier/` (inside `/kaggle/working/`).
2. Save the tokenizer alongside it.
3. Test inference on 5 sample test emails, print predicted labels.
4. **Publish as a Kaggle Dataset** (same flow as Step 6.8) named `email-nlp-intent-classifier` so it survives past this session and can be attached to Phase 12/13 notebooks without retraining.

---

### Step 10.8 — Commit Intent Classifier Training Notebook

```python
!git add notebooks/06_Intent_Classifier_Training.ipynb
!git commit -m "Add intent classifier training"
!git push origin main
```

(The trained model weights themselves stay in the Kaggle Dataset from Step 10.7, not in git — they're too large and git isn't built for binary model storage.)

---

## Phase 11 — Model Training (Summarization)

### Step 11.1 — Create Training Notebook `07-summarization-model-training`

Bootstrap fully, attach `email-nlp-processed-data`, set **Accelerator: GPU T4 x2** or **P100**.

---

### Step 11.2 — Choose Your Summarization Base Model [DECISION]

| Model | HuggingFace ID | Size | Recommendation |
|---|---|---|---|
| T5-Small | `t5-small` | ~240 MB | **Best for mobile.** Start here. |
| T5-Base | `t5-base` | ~890 MB | Better quality, harder to compress for mobile |
| DistilBART | `sshleifer/distilbart-cnn-12-6` | ~1.2 GB | Good quality, larger |

Start with `t5-small`; escalate to `t5-base` only if ROUGE scores are too low.

---

### Step 11.3 — Load Summarization Training Data

Load `summarization_train.csv` / `summarization_val.csv`, print record count and average input/summary token lengths. If input length > 512 tokens, truncate during tokenization (`max_length=512, truncation=True`).

---

### Step 11.4 — Configure Summarization Training Parameters

```
Model: t5-small
Max input length: 512 tokens
Max output length: 128 tokens
Batch size: 8 (reduce to 4 if GPU runs out of memory)
Learning rate: 3e-5
Epochs: 4
Optimizer: AdamW
Beam search: 4 beams during evaluation
No-repeat n-gram size: 2
Output directory: models/checkpoints/summarization/
```

---

### Step 11.5 — Run Summarization Fine-Tuning

1. Start training — expect roughly 30–90 minutes on a Kaggle T4, well within a single GPU-quota session.
2. Monitor per-epoch loss.
3. After training, generate summaries for 3 validation emails and inspect coherence and length.

---

### Step 11.6 — Save and Publish the Final Summarization Model

1. Copy the best checkpoint + tokenizer to `models/final/summarization/`.
2. Test on 5 test-set emails, print generated summaries.
3. Publish as a Kaggle Dataset, `email-nlp-summarization-model`, for reuse in later phases.

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

Attach `email-nlp-intent-classifier`, `email-nlp-summarization-model`, and `email-nlp-processed-data` via **Add Data**. Accelerator: GPU optional (CPU is fine for inference-only evaluation on a modest test set; use GPU if it's slow).

---

### Step 12.2 — Evaluate Summarization with ROUGE Scores

1. Load the summarization model/tokenizer from the attached dataset's `/kaggle/input/email-nlp-summarization-model/` path.
2. Load `summarization_test.csv`.
3. Generate a summary per record, score against the reference with `rouge-score`.
4. Report average ROUGE-1, ROUGE-2, ROUGE-L F1. Target: ROUGE-1 > 0.30, ROUGE-2 > 0.10, ROUGE-L > 0.25.

---

### Step 12.3 — Evaluate Intent Classifier

1. Load the intent classifier from `/kaggle/input/email-nlp-intent-classifier/`.
2. Run inference on `intent_test.csv`.
3. Report accuracy, macro F1, per-class precision/recall/F1 (`classification_report`), and a confusion matrix plot. Target: accuracy > 70%.

---

### Step 12.4 — Evaluate NER (Baseline spaCy)

1. Take 50 test emails; manually annotate entities in 10 as ground truth.
2. Run spaCy NER on those 10, compare, compute precision/recall/F1.

---

### Step 12.5 — Create a Summary Evaluation Table

```
| Component | Metric | Score |
|---|---|---|
| Summarization | ROUGE-1 | X.XX |
| Summarization | ROUGE-2 | X.XX |
| Summarization | ROUGE-L | X.XX |
| Intent Classifier | Accuracy | XX% |
| Intent Classifier | Macro F1 | X.XX |
| NER (spaCy sm) | Precision | X.XX |
| NER (spaCy sm) | Recall | X.XX |
| NER (spaCy sm) | F1 | X.XX |
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

Attach the two trained-model Kaggle Datasets. Accelerator: CPU is usually sufficient for conversion/quantization.

---

### Step 13.2 — Confirm Your Mobile Framework [DECISION]

**Choose TFLite if:** you trained with TensorFlow, or prefer TFLite's better-documented Android path.
**Choose PyTorch Mobile if:** you want to minimize framework changes from Hugging Face/PyTorch.

**Recommendation:** TFLite — most Hugging Face models export to a TensorFlow SavedModel and convert cleanly.

---

### Step 13.3 — Export the Summarization Model to Mobile Format

**TFLite path:** export to TensorFlow SavedModel → TFLite Converter → apply float16 (then int8 if needed) quantization → save to `models/mobile/summarization_model.tflite` (target < 50 MB).

**PyTorch Mobile path:** export to TorchScript → apply mobile optimizer → save as `models/mobile/summarization_model.ptl`.

---

### Step 13.4 — Export the Intent Classifier to Mobile Format

Repeat Step 13.3 for the intent classifier → `models/mobile/intent_classifier.tflite` (or `.ptl`).

---

### Step 13.5 — Handle the spaCy NER Model

spaCy models don't convert to TFLite directly. Either ship the spaCy model as a separate asset, or implement a lightweight regex/rule-based entity extractor in Kotlin for the mobile app. Note `en_core_web_sm`'s size via `python -m spacy info en_core_web_sm` for planning.

---

### Step 13.6 — Benchmark the Mobile Models (on Kaggle, as a PC-equivalent baseline)

1. Load the `.tflite` file with the TFLite Python interpreter inside the Kaggle notebook.
2. Run inference on 10 test emails, record average latency.
3. Compare accuracy (ROUGE/F1) before and after quantization.

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

1. Publish `models/mobile/*.tflite` (or `.ptl`) as a Kaggle Dataset, `email-nlp-mobile-models`, **or** download them directly to your laptop via the notebook's **Output** tab — you'll need the physical files locally for Phase 14.
2. ```python
   !git add notebooks/09_Model_Mobile_Conversion.ipynb
   !git commit -m "Add model mobile conversion"
   !git push origin main
   ```

---

## Phase 14 — Android Application Setup (Local Laptop Required)

> Everything from here through Phase 16 happens on your **laptop**, in Android Studio — Kaggle has no role in mobile app development or on-device testing. Before starting, download the `.tflite`/`.ptl` files you produced in Step 13.8 from your Kaggle Dataset output (or notebook Output tab) onto your laptop.

### Step 14.1 — Create a New Android Project in Android Studio

1. Open Android Studio → **New Project** → **Empty Views Activity** → **Next**.
2. **Name:** `EmailIntelligence`. **Package name:** `com.samsung.emailintelligence`. **Save location:** your locally cloned `context-aware-email-intelligence/android_app/`. **Language:** Kotlin. **Minimum SDK:** API 26.
3. **Finish**, wait for Gradle sync.
4. **[VERIFY]** Status bar shows "Gradle sync successful."

---

### Step 14.2 — Add Required Dependencies to build.gradle

In `build.gradle (Module: app)`, inside `dependencies { }`:

```
implementation 'org.tensorflow:tensorflow-lite:2.13.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
implementation 'com.google.code.gson:gson:2.10.1'
implementation 'com.google.android.material:material:1.11.0'
```

Click **Sync Now**. **[VERIFY]** Sync succeeds with no errors.

---

### Step 14.3 — Add the TFLite Model Files (Downloaded From Kaggle) to App Assets

1. Right-click `app` → **New** → **Directory** → `src/main/assets`.
2. Copy the `.tflite` files you downloaded from Kaggle (Step 13.8) into `android_app/app/src/main/assets/`.
3. **[VERIFY]** Both `.tflite` files appear under `app/src/main/assets` in the project navigator.

---

### Step 14.4 — Configure AndroidManifest.xml

1. Open `app/src/main/AndroidManifest.xml`.
2. Only add `<uses-permission android:name="android.permission.INTERNET" />` if you plan optional network features.
3. Set `android:allowBackup="false"` on `<application>` (privacy requirement — no email content backed up to Google servers).

---

## Phase 15 — Android App Module Development (Local Laptop Required)

### Step 15.1 — Design the UI Layout

In `app/src/main/res/layout/activity_main.xml` (XML/Code view), lay out top-to-bottom: `ScrollView` > `LinearLayout` containing a title, multiline email-input `EditText`, **Analyze** `Button`, hidden `ProgressBar`, and result sections for Summary, Detected Intent, and Key Entities. Apply Material Design spacing.

---

### Step 15.2 — Create the Kotlin Source Files

Under `com.samsung.emailintelligence`, create empty class stubs: `EmailPreprocessor.kt`, `InferenceEngine.kt`, `OutputFormatter.kt`, `PrivacyManager.kt`.

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

Build and run in the emulator after each increment: (1) input + button + Logcat click confirmation, (2) ProgressBar show/hide, (3) `EmailPreprocessor` wired in, (4) intent inference wired in, (5) summarization inference wired in, (6) entity extraction wired in.

---

### Step 15.5 — Connect a Physical Device and Do a Device Run

Connect via USB, select it in the device dropdown, **Run**. Test the full flow: paste email → Analyze → verify output, no crashes.

---

## Phase 16 — Android App Testing and Performance Profiling (Local Laptop Required)

### Step 16.1 — Open the Android Profiler

**View** → **Tool Windows** → **Profiler**, attach to your running app session.

---

### Step 16.2 — Measure Inference Latency

Test 10 emails of varying length; instrument `InferenceEngine.kt` with timestamps around inference calls and `Log.d("PERF", ...)`. Record results:

| Email # | Email Length (words) | Inference Time (ms) | Within Target (<3000ms)? |
|---|---|---|---|
| 1 | X | X | ☐ |

---

### Step 16.3 — Measure RAM Usage

Watch the Profiler's memory graph during inference on a long email. **[VERIFY]** Peak RAM < 300 MB.

---

### Step 16.4 — Measure APK Size

**Build** → **Build Bundle(s) / APK(s)** → **Build APK(s)**. Check the `.apk` size in `android_app/app/build/outputs/apk/debug/`. **[VERIFY]** < 100 MB.

---

### Step 16.5 — Document All Performance Results

| Metric | Target | Measured | Pass/Fail |
|---|---|---|---|
| Average inference time (ms) | < 3000 ms | X ms | ☐ |
| Peak RAM usage (MB) | < 300 MB | X MB | ☐ |
| Model file size (MB) | < 50 MB | X MB | ☐ |
| APK size (MB) | < 100 MB | X MB | ☐ |

If anything fails, note optimization plans (smaller model, more aggressive quantization).

---

## Phase 17 — Usability Testing

### Step 17.1 — Prepare a Usability Test Plan

Write `docs/usability_test_plan.md`: 3–5 testers, tasks, observation notes, post-test questionnaire.

**Tasks:** open app → paste a sample email (non-sensitive, work-relevant, pulled from Enron) → tap Analyze → describe what's shown → rate the summary, intent label, and entities.

---

### Step 17.2 — Recruit Testers

3–5 people unfamiliar with the app. Brief them neutrally. Schedule 15-minute slots.

---

### Step 17.3 — Conduct Usability Tests

Hand them the phone, read the task without helping, observe silently (pauses, confusion, questions, success/fail), then have them fill out the rating questionnaire (ease of use, summary accuracy, intent correctness, entity relevance, comments).

---

### Step 17.4 — Document Usability Findings

In `docs/usability_test_results.md`: response table, average ease-of-use score, common confusion points, summary accuracy rating, planned UI changes.

---

## Phase 18 — Final Documentation and Deliverables

### Step 18.1 — Write the Performance Evaluation Report

`docs/performance_evaluation_report.md`: Introduction; Summarization Evaluation (ROUGE, Phase 12); Intent Classification Evaluation (accuracy/F1/confusion matrix); NER Evaluation; Android App Performance (Phase 16); Usability Results (Phase 17); Observations and Areas for Improvement; Conclusion.

---

### Step 18.2 — Write the Privacy and Security Note

`docs/privacy_security_note.md`: on-device processing overview; Android permissions declared/excluded and why; data retention policy (no email content stored post-session); how the no-logging policy is implemented (`PrivacyManager.kt`); future considerations (OAuth if email integration is added in v2).

---

### Step 18.3 — Finalize Source Code and Clean Up

1. In your Kaggle notebooks: remove noisy debug print cells, add markdown section headers, confirm every notebook runs top-to-bottom (**Save Version → Save & Run All**) without errors.
2. In the Android app: remove debug `Log.d` statements that log email content; confirm clean, correctly-spelled UI labels.
3. Write the project-root `README.md` covering: what the project is; **how to set up the Kaggle Notebook environment** (Phase 1) instead of a local Python environment; how to run each notebook in order (and which Kaggle Datasets each depends on via Add Data); how to open and run the Android app locally; where to find/attach each dataset.

---

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

---

### Step 18.5 — Prepare the Final Presentation

Slides: Title (project, team, Samsung Collaborative Project); Problem statement; Solution overview/architecture (mention the Kaggle-based training pipeline + local Android build); Datasets used; NLP Pipeline explanation; Model details and training approach; Evaluation results; Android app demo screenshots; Challenges faced (including any Kaggle-specific ones — GPU quota, session ephemerality); Future work; Thank you/Q&A. Save to `docs/final_presentation.pptx`.

---

### Step 18.6 — Record the Demo Video (Optional but Recommended)

Use your phone's screen recorder or Android Studio's device-screen-record tool. Record: opening the app, pasting a sample email, tapping Analyze, showing summary/intent/entities, brief narration. Keep under 3 minutes. Save to `docs/demo_video.mp4`.

---

### Step 18.7 — Final Git Push — All Deliverables

```
git add .
git commit -m "Final deliverables - all documentation, notebooks, and Android project"
git push origin main
```

**[VERIFY]** All files/folders visible and current on GitHub. Also do a final check that your three Kaggle Datasets (`email-nlp-processed-data`, `email-nlp-intent-classifier`/`email-nlp-summarization-model`, `email-nlp-mobile-models`) are set to at least **Private** (not deleted, not left as session-only output) so mentors can access them if needed.

Share the repository link — and, if useful, links to your Kaggle Datasets/Notebooks — with your mentors, Dr. Naveenkumar J and Dr. Joshva Devadas T, for review.

---

## Quick Reference: Phase Summary

| Phase | What Gets Done | Where | Output |
|---|---|---|---|
| 0 | Accounts, Kaggle readiness | Browser | GitHub repo, Kaggle (phone-verified), HuggingFace account |
| 1 | Kaggle Notebook environment | **Kaggle** | Bootstrapped notebook with GPU, secrets, packages |
| 2 | Android environment | **Laptop** | Android Studio + emulator ready |
| 3 | Project structure, repo sync | **Kaggle** ↔ GitHub | Cloned repo, folder tree, storage model understood |
| 4 | Dataset attachment | **Kaggle** (Add Data) | All 4 datasets accessible in-notebook |
| 5 | Dataset exploration | **Kaggle** | Understanding of each dataset's format/structure |
| 6 | Preprocessing | **Kaggle** | Cleaned CSVs, train/val/test splits, published as Kaggle Dataset |
| 7 | Intent data prep | **Kaggle** + laptop (manual labeling) | 600 labeled emails, split |
| 8 | Summarization data prep | **Kaggle** | EMAILSUM + BC3 formatted and split |
| 9 | Baseline NLP pipeline | **Kaggle** (CPU) | Working NER and LDA on sample emails |
| 10 | Intent classifier training | **Kaggle** (GPU) | Fine-tuned DistilBERT, published as Kaggle Dataset |
| 11 | Summarization training | **Kaggle** (GPU) | Fine-tuned T5-Small, published as Kaggle Dataset |
| 12 | Model evaluation | **Kaggle** | ROUGE, F1 scores documented |
| 13 | Model conversion | **Kaggle** | `.tflite` files, published/downloaded for Android |
| 14 | Android project setup | **Laptop** | New Android Studio project with dependencies |
| 15 | Android app development | **Laptop** | Working app with all 4 modules implemented |
| 16 | Performance profiling | **Laptop** | Latency, RAM, APK size measured and documented |
| 17 | Usability testing | **Laptop** / physical device | Tester feedback collected and documented |
| 18 | Final documentation | **Kaggle** + GitHub + **Laptop** | All deliverables packaged, repo finalized, presentation ready |
