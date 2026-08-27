# Technical Requirements Document (TRD)
## Context-Aware Email Intelligence: A Unified NLP Approach to Information Synthesis

**Project Type:** Samsung Collaborative Research Project  
**Document Version:** 1.0  
**Date:** June 2026  
**Status:** Active Development

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Dataset Specifications](#3-dataset-specifications)
4. [Hardware Requirements](#4-hardware-requirements)
5. [NLP Pipeline Technical Specification](#5-nlp-pipeline-technical-specification)
6. [Model Training and Fine-Tuning Specification](#6-model-training-and-fine-tuning-specification)
7. [Model Compression and Mobile Deployment](#7-model-compression-and-mobile-deployment)
8. [Android Application Technical Specification](#8-android-application-technical-specification)
9. [Data Pipeline Specification](#9-data-pipeline-specification)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Privacy and Security Technical Requirements](#11-privacy-and-security-technical-requirements)
12. [Final Artifact Specifications](#12-final-artifact-specifications)

---

## 1. System Architecture Overview

The system is composed of three major, sequentially connected layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 1: NLP TRAINING PIPELINE                │
│         (Python environment, executed offline on dev machine)   │
│                                                                 │
│  Dataset Acquisition → Preprocessing → Feature Extraction →    │
│  Model Training / Fine-Tuning → Evaluation                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ Trained Model Files
┌─────────────────────────────────────────────────────────────────┐
│            LAYER 2: MODEL CONVERSION AND OPTIMIZATION           │
│         (Python environment, executed offline on dev machine)   │
│                                                                 │
│  Model Export → Quantization → TFLite / TorchScript Conversion │
│  → Mobile Benchmarking → Final Model File                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ Compressed, Mobile-Ready Model
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 3: ANDROID APPLICATION                    │
│              (Android Studio, deployed on Android device)       │
│                                                                 │
│  Email Text Input → On-Device Preprocessing → On-Device        │
│  Inference → Structured Output Display (Summary, Intent,       │
│  Entities)                                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

### 2.1 Python Development Environment

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python 3.9+ | Core development language for NLP pipeline |
| Basic NLP | NLTK | Tokenization, stopword removal, sentence segmentation |
| Advanced NLP + NER | spaCy | Named entity recognition, dependency parsing, sentence segmentation |
| Topic Modeling | Gensim | Latent Dirichlet Allocation (LDA) for topic flow analysis |
| Transformer Models | Hugging Face Transformers | Fine-tuning BERT, DistilBERT, BART, T5 for summarization and intent classification |
| Mobile Runtime (Option A) | TensorFlow Lite | Model conversion and on-device inference on Android |
| Mobile Runtime (Option B) | PyTorch Mobile | Alternative model conversion and on-device inference on Android |
| Mobile IDE | Android Studio | Android application development |

### 2.2 Supporting Python Libraries (to be installed)

| Library | Purpose |
|---|---|
| `torch` | PyTorch for transformer model training |
| `tensorflow` | TensorFlow for model training and TFLite export |
| `transformers` | Hugging Face model hub and training utilities |
| `datasets` | Hugging Face datasets library for loading training data |
| `sentencepiece` | Tokenization library required by T5 and BART models |
| `rouge-score` | Computing ROUGE metrics for summarization evaluation |
| `scikit-learn` | Classification metrics (accuracy, precision, recall, F1) |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `matplotlib` / `seaborn` | Visualization of evaluation results |
| `tqdm` | Progress bars during training |
| `nltk` | Tokenization, stopwords, sentence tokenizer |
| `spacy` | NER, sentence segmentation, linguistic features |
| `gensim` | Topic modeling with LDA |
| `email` (stdlib) | Parsing raw email files in .eml or mbox format |
| `mailbox` (stdlib) | Reading mbox-format email archives (Enron dataset) |
| `bs4` (BeautifulSoup4) | Stripping HTML from HTML-formatted emails |
| `re` (stdlib) | Regex-based email cleaning |

### 2.3 Android Application Dependencies

| Library | Version/Group ID | Purpose |
|---|---|---|
| TensorFlow Lite | `org.tensorflow:tensorflow-lite` | On-device model inference (Option A) |
| TFLite Support | `org.tensorflow:tensorflow-lite-support` | Text preprocessing utilities for TFLite |
| PyTorch Mobile Lite | `org.pytorch:pytorch_android_lite` | On-device model inference (Option B) |
| Kotlin Coroutines | `org.jetbrains.kotlinx:kotlinx-coroutines-android` | Background threading to prevent UI blocking |
| Gson | `com.google.code.gson:gson` | JSON serialization for structured output |
| AndroidX Core | `androidx.core:core-ktx` | Kotlin extensions for Android framework |
| AndroidX AppCompat | `androidx.appcompat:appcompat` | Backward-compatible UI components |
| Material Components | `com.google.android.material:material` | Material Design UI widgets |

---

## 3. Dataset Specifications

### 3.1 BC3 Corpus

| Attribute | Detail |
|---|---|
| Full Name | British Columbia Conversation Corpus |
| Type | Annotated email thread dataset |
| Use in Project | Primary training and evaluation data for thread summarization |
| Content | Email threads annotated with summaries, speech acts, and extractive segments |
| Format | XML |
| Access | Academic request / research use |

### 3.2 Enron Email Dataset

| Attribute | Detail |
|---|---|
| Full Name | Enron Email Dataset |
| Type | Real-world corporate email archive |
| Use in Project | Large-scale real email training corpus for preprocessing, intent classification, and NER |
| Content | ~500,000 emails from ~150 Enron employees |
| Format | mbox / raw .eml files organized in per-user folders |
| Access | Publicly available for research (CMU mirror and Kaggle) |
| Size | ~1.7 GB compressed |

### 3.3 EMAILSUM

| Attribute | Detail |
|---|---|
| Full Name | EMAILSUM Dataset |
| Type | Human-annotated email thread summary dataset |
| Use in Project | Fine-tuning and evaluating the abstractive summarization model |
| Content | Email threads paired with short and long human-written summaries |
| Format | JSON / CSV |
| Access | Available via research paper authors or HuggingFace Datasets |

### 3.4 SpamAssassin / TREC 2007 Spam Corpus

| Attribute | Detail |
|---|---|
| Full Name | SpamAssassin Public Corpus / TREC 2007 Spam Track Corpus |
| Type | Spam/ham labeled email dataset |
| Use in Project | Noise filtering — training a spam detection layer to discard non-relevant emails before NLP processing |
| Format | mbox / raw text |
| Access | Apache SpamAssassin website (public) / TREC archive |

---

## 4. Hardware Requirements

| Component | Minimum Specification | Recommended |
|---|---|---|
| Development Machine CPU | Intel Core i5 or AMD equivalent | Intel Core i7 / i9 or AMD Ryzen 7/9 |
| Development Machine RAM | 16 GB | 32 GB |
| Development Machine Storage | 50 GB free (for datasets, models, environments) | 100 GB SSD |
| GPU (for model training) | NVIDIA GPU with CUDA support (4 GB VRAM) | NVIDIA RTX 3060 or higher (8+ GB VRAM) |
| Android Test Device | Android 8.0 (API 26), 3 GB RAM | Android 10+ (API 29+), 4–6 GB RAM |
| Android Emulator | Requires sufficient host RAM to run AVD | AVD with Android 11+ (API 30+) |

> **Note:** GPU is strongly recommended for fine-tuning transformer models. If a local GPU is not available, cloud platforms such as Google Colab Pro, Kaggle Kernels, or AWS EC2 (p2/p3 instances) may be used for the training phase only.

---

## 5. NLP Pipeline Technical Specification

### 5.1 Email Parsing and Cleaning Module

**Input:** Raw email text (single email or multi-email thread as a string)  
**Output:** Clean list of individual email body strings, stripped of noise

**Processing Steps:**
1. **Header Removal:** Use `email` standard library to parse and separate headers (To, From, Date, Subject, CC, BCC) from body.
2. **Thread Splitting:** Identify individual messages in a thread using delimiter patterns (e.g., "On [date], [person] wrote:", "-----Original Message-----", "From: ... Sent: ...").
3. **HTML Stripping:** Use `BeautifulSoup4` to strip HTML tags from HTML-formatted email bodies.
4. **Signature Removal:** Detect and remove common signature patterns using regex (e.g., lines starting with "--", "Best regards", "Thanks,", "Sent from my iPhone").
5. **Quoted Reply Removal:** Strip lines beginning with ">" (standard email reply quotation) to avoid processing duplicate content in threads.
6. **Whitespace Normalization:** Collapse multiple blank lines and trailing whitespace.

### 5.2 Text Preprocessing Module

**Input:** Cleaned email body text  
**Output:** Preprocessed tokens and sentences

**Processing Steps:**
1. **Sentence Segmentation:** Use `spaCy`'s sentence boundary detection (`nlp.sentencizer`) or `NLTK`'s `sent_tokenize`.
2. **Word Tokenization:** Use `NLTK`'s `word_tokenize` or spaCy's tokenizer.
3. **Lowercasing:** Convert all text to lowercase for uniformity.
4. **Stopword Removal:** Apply NLTK's English stopword list augmented with custom email-specific stopwords (e.g., "re", "fw", "fwd", "attached", "regards", "sincerely", "dear").
5. **Punctuation Filtering:** Remove or retain punctuation based on downstream task (retain for NER and summarization, remove for topic modeling).

### 5.3 Named Entity Recognition (NER) Module

**Library:** spaCy  
**Model:** `en_core_web_sm` (lightweight) or `en_core_web_trf` (transformer-based, higher accuracy)

**Entity Types to Extract:**

| spaCy Label | Meaning | Email Use Case |
|---|---|---|
| PERSON | People's names | Sender/recipient names, mentioned individuals |
| ORG | Organizations, companies | Company names, departments |
| DATE | Absolute or relative dates | Deadlines, meeting dates |
| TIME | Times of day | Meeting times, scheduled calls |
| GPE | Countries, cities, states | Location references |
| MONEY | Monetary values | Budget figures, financial mentions |
| EVENT | Named events | Conferences, meetings, project names |

**Output Format:**
```json
{
  "entities": {
    "PERSON": ["John Smith", "Priya Mehta"],
    "ORG": ["Samsung", "Acme Corp"],
    "DATE": ["June 15", "next Monday"],
    "GPE": ["Bangalore", "Seoul"],
    "MONEY": ["$5,000"],
    "EVENT": ["Q3 Review Meeting"]
  }
}
```

### 5.4 Intent Classification Module

**Approach:** Supervised multi-class text classification  
**Base Model:** `distilbert-base-uncased` (fine-tuned via Hugging Face Transformers)  
**Training Framework:** Hugging Face `Trainer` API

**Intent Classes:**

| Label | Description | Example Trigger Phrases |
|---|---|---|
| REQUEST | Sender is asking for something | "Could you please...", "Can you send me..." |
| FOLLOW_UP | Checking on a previous communication | "Just following up on...", "Any update on..." |
| INFORMATION | Sharing or reporting information | "Please note that...", "I wanted to let you know..." |
| ACKNOWLEDGEMENT | Confirming receipt or agreement | "Noted, thanks.", "Confirmed, we'll proceed." |
| COMPLAINT | Expressing dissatisfaction or escalating | "I'm disappointed...", "This is unacceptable..." |
| INVITATION | Inviting to an event or meeting | "You're invited to...", "Please join us for..." |

**Training Data Source:** Labeled subset of Enron dataset + BC3 corpus with manually annotated intent labels  
**Loss Function:** Cross-entropy  
**Evaluation Metric:** Accuracy, Macro F1

### 5.5 Summarization Module

**Approach:** Abstractive summarization using pre-trained sequence-to-sequence models

**Model Options and Trade-offs:**

| Model | Size | Quality | Mobile Suitability |
|---|---|---|---|
| `facebook/bart-large-cnn` | Large (~1.6 GB) | Excellent | Needs heavy compression |
| `facebook/bart-base` | Medium (~550 MB) | Good | Needs compression |
| `t5-small` | Small (~242 MB) | Good | Compresses well |
| `t5-base` | Medium (~892 MB) | Very Good | Needs compression |
| `sshleifer/distilbart-cnn-12-6` | Medium-small | Good | Better than BART-large |

**Recommended Starting Model:** `t5-small` for mobile feasibility; `bart-base` if accuracy is prioritized  
**Fine-Tuning Dataset:** EMAILSUM (email-specific summarization with human references)  
**Training Framework:** Hugging Face `Seq2SeqTrainer`

**Thread Summarization Strategy:**
1. Summarize each individual message in the thread independently (micro-summaries).
2. Concatenate micro-summaries in chronological order.
3. Pass concatenated micro-summaries through a second summarization pass to generate the final thread-level summary.

**Evaluation Metrics:**
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)

### 5.6 Topic Flow Analysis Module (Gensim LDA)

**Purpose:** Identify how topics shift across replies in an email thread to provide richer context understanding  
**Library:** Gensim  
**Algorithm:** Latent Dirichlet Allocation (LDA)

**Processing Steps:**
1. Build a dictionary (vocabulary) from all preprocessed email tokens in the thread.
2. Create a bag-of-words corpus from each email in the thread.
3. Train an LDA model with a small number of topics (e.g., 3–5 per thread).
4. Identify the dominant topic for each email in the thread.
5. Output a topic flow summary showing how the dominant subject shifts across the thread.

---

## 6. Model Training and Fine-Tuning Specification

### 6.1 Intent Classifier Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Number of labels | 6 (intent classes) |
| Max sequence length | 512 tokens |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 3–5 |
| Optimizer | AdamW |
| Evaluation strategy | Per epoch |

### 6.2 Summarization Model Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | `t5-small` or `bart-base` |
| Max input length | 512 tokens |
| Max output length (summary) | 128 tokens |
| Batch size | 8 |
| Learning rate | 3e-5 |
| Epochs | 3–5 |
| Optimizer | AdamW |
| Beam search (inference) | 4 beams |
| No-repeat n-gram size | 2 |

---

## 7. Model Compression and Mobile Deployment

### 7.1 TensorFlow Lite Path (Option A)

**Step 1 — Save the trained model** in SavedModel or Keras format.  
**Step 2 — Convert using TFLite Converter:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model("model_path")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enables quantization
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```
**Step 3 — Apply Post-Training Quantization:**
- Float16 quantization: reduces size by ~50%, minimal accuracy loss.
- Int8 quantization: reduces size by ~75%, may require representative dataset calibration.

**Step 4 — Bundle `.tflite` file in Android project** under `app/src/main/assets/`.

### 7.2 PyTorch Mobile Path (Option B)

**Step 1 — Export model to TorchScript:**
```python
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```
**Step 2 — Apply mobile optimization:**
```python
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter("model.ptl")
```
**Step 3 — Bundle `.ptl` file in Android project** under `app/src/main/assets/`.

### 7.3 Selection Criteria

The final deployment format (TFLite vs PyTorch Mobile) shall be selected based on benchmarking results:

| Criterion | Measurement Method |
|---|---|
| Model file size (MB) | File system size comparison |
| Inference time (ms) | Timed inference on target Android device |
| Accuracy retention | ROUGE / F1 score comparison before and after compression |
| Integration complexity | Developer effort to integrate in Android Studio |

---

## 8. Android Application Technical Specification

### 8.1 Development Environment

| Setting | Value |
|---|---|
| IDE | Android Studio (latest stable version) |
| Language | Kotlin (primary) |
| Minimum SDK | Android 8.0 (API level 26) |
| Target SDK | Android 14 (API level 34) |
| Build System | Gradle |

### 8.2 Application Module Architecture

```
app/
├── src/main/
│   ├── assets/
│   │   └── model.tflite  (or model.ptl)
│   ├── java/com/project/emailintelligence/
│   │   ├── MainActivity.kt            ← Entry point, UI controller
│   │   ├── EmailPreprocessor.kt       ← Text cleaning, thread parsing
│   │   ├── InferenceEngine.kt         ← Loads and runs TFLite/PyTorch model
│   │   ├── OutputFormatter.kt         ← Formats JSON output for display
│   │   └── PrivacyManager.kt          ← Handles permission checks, no-log policy
│   └── res/
│       ├── layout/
│       │   └── activity_main.xml      ← UI layout
│       └── values/
│           └── strings.xml
```

### 8.3 UI Screen Layout (activity_main.xml)

| UI Element | Type | Purpose |
|---|---|---|
| Email Input Area | `EditText` (multiline) | User pastes email or thread text |
| Analyze Button | `Button` | Triggers NLP pipeline |
| Progress Indicator | `ProgressBar` | Shown during inference |
| Summary Section | `TextView` | Displays generated summary |
| Intent Section | `TextView` or `Chip` | Displays detected intent label |
| Entities Section | `RecyclerView` or `TextView` | Displays extracted entities grouped by type |

### 8.4 Inference Engine Flow

```
User Input
    │
    ▼
EmailPreprocessor.kt
    │ - Strip headers, signatures, HTML
    │ - Tokenize for model input
    │ - Convert to model input format (input IDs, attention mask)
    ▼
InferenceEngine.kt
    │ - Load model from assets (lazy initialization on first run)
    │ - Run inference (background thread via Coroutine)
    │ - Parse model output logits / token IDs
    ▼
OutputFormatter.kt
    │ - Decode token IDs to text (summary)
    │ - Map logit index to intent label
    │ - Format entity dictionary for display
    ▼
UI Update (Main Thread)
    │ - Populate Summary TextView
    │ - Populate Intent TextView/Chip
    │ - Populate Entities Section
```

### 8.5 Threading Model

- All model inference shall run on a **background coroutine** (`Dispatchers.IO`) to prevent ANR (Application Not Responding) errors.
- UI updates shall always be dispatched back to the **main thread** (`Dispatchers.Main`).
- A `ProgressBar` shall be shown on the main thread while background inference is running.

### 8.6 Performance Targets

| Metric | Target Value |
|---|---|
| Inference time (single email) | < 3,000 ms (3 seconds) on mid-range device |
| App RAM usage during inference | < 300 MB |
| Model file size (on-device) | < 50 MB (after quantization) |
| App APK size | < 100 MB |
| App startup time | < 2 seconds |

---

## 9. Data Pipeline Specification

### 9.1 Dataset Acquisition and Storage

- All datasets shall be downloaded and stored locally on the development machine.
- Real email content from Enron and other datasets shall **not** be uploaded to any cloud storage service.
- A dedicated project directory structure shall be maintained:

```
project_root/
├── data/
│   ├── raw/
│   │   ├── enron/              ← Raw mbox files
│   │   ├── bc3/                ← Raw XML files
│   │   ├── emailsum/           ← Raw JSON/CSV files
│   │   └── spamassassin/       ← Raw email files
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
├── models/
│   ├── checkpoints/            ← Training checkpoints
│   ├── final/                  ← Final trained models
│   └── mobile/                 ← TFLite / TorchScript files
├── notebooks/                  ← Jupyter notebooks for experimentation
├── src/                        ← Python source modules
└── android_app/                ← Android Studio project
```

### 9.2 Train / Validation / Test Split

| Split | Proportion | Purpose |
|---|---|---|
| Train | 70% | Model learning |
| Validation | 15% | Hyperparameter tuning and early stopping |
| Test | 15% | Final unbiased evaluation |

### 9.3 Preprocessing Output Formats

| Stage | Output Format |
|---|---|
| Cleaned email bodies | `.txt` files or `pandas` DataFrame saved as `.csv` |
| Tokenized inputs (for model training) | HuggingFace `Dataset` object (saved as Arrow format) |
| Trained model checkpoints | HuggingFace `Trainer` checkpoint directories |
| Final model (Python) | SavedModel (TF) or `.pt` (PyTorch) |
| Final model (mobile) | `.tflite` or `.ptl` file |

---

## 10. Evaluation Framework

### 10.1 Summarization Evaluation

| Metric | Tool | Description |
|---|---|---|
| ROUGE-1 | `rouge-score` Python library | Unigram overlap between generated and reference summaries |
| ROUGE-2 | `rouge-score` Python library | Bigram overlap |
| ROUGE-L | `rouge-score` Python library | Longest common subsequence overlap |

**Reference summaries:** Human-annotated summaries from EMAILSUM and BC3 datasets.

### 10.2 NER Evaluation

| Metric | Tool | Description |
|---|---|---|
| Precision | `scikit-learn` or spaCy evaluator | Of all predicted entities, what fraction are correct? |
| Recall | `scikit-learn` or spaCy evaluator | Of all true entities, what fraction were found? |
| F1 Score | `scikit-learn` or spaCy evaluator | Harmonic mean of precision and recall |

### 10.3 Intent Classification Evaluation

| Metric | Tool | Description |
|---|---|---|
| Accuracy | `scikit-learn` | Overall correct predictions |
| Macro F1 | `scikit-learn` | F1 averaged equally across all classes |
| Per-class F1 | `scikit-learn classification_report` | F1 score for each intent category |
| Confusion Matrix | `scikit-learn` / `matplotlib` | Visual breakdown of misclassifications |

### 10.4 Android App Performance Evaluation

| Metric | Tool | Description |
|---|---|---|
| Inference latency (ms) | Android Profiler / `System.currentTimeMillis()` | Time from button press to result display |
| RAM usage (MB) | Android Profiler | Peak memory during inference |
| Battery impact | Android Battery Stats | Power draw during processing session |

### 10.5 Usability Evaluation

- Conduct informal usability testing with 3–5 users (peers, lab members).
- Collect feedback using a structured questionnaire (e.g., System Usability Scale — SUS).
- Identify and document any UI issues or confusion points.

---

## 11. Privacy and Security Technical Requirements

| Requirement | Technical Implementation |
|---|---|
| On-device only processing | All model inference runs in-app; no HTTP calls to external APIs during core operation |
| Minimal permissions | `AndroidManifest.xml` shall declare only `INTERNET` (if needed for future email integration) — no `READ_CONTACTS`, `READ_CALL_LOG`, or email account permissions in v1 |
| No persistent storage of user data | Email text entered by the user shall be held only in memory and discarded when the app session ends |
| No logging of content | No logging of user-entered email text to `Logcat`, local files, or analytics services |
| Future email integration (v2) | If direct email account access is added in a future version, OAuth 2.0 shall be used; raw email content shall never be transmitted to third parties |

---

## 12. Final Artifact Specifications

| Artifact | Format | Location |
|---|---|---|
| Python NLP pipeline source code | `.py` modules | `src/` directory |
| Jupyter notebooks for experiments | `.ipynb` | `notebooks/` directory |
| Trained model (Python) | HuggingFace checkpoint | `models/final/` |
| Mobile model file | `.tflite` or `.ptl` | `models/mobile/` and `android_app/assets/` |
| Android app source code | Kotlin + Gradle | `android_app/` (Android Studio project) |
| Evaluation report | `.pdf` or `.md` | Project documentation folder |
| Privacy note | `.pdf` or `.md` | Project documentation folder |
| Test cases | `.py` unit test files | `tests/` directory |
| Final presentation/demo | `.pptx` or recorded demo video | Deliverables folder |