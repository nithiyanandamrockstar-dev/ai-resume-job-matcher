# AI Resume Analyzer & Job Matcher (Next-Gen AI Talent Intelligence Platform)

**Final-Year AI & Data Science Flagship Project | Advanced Glassmorphism Tech UX | NLP**

---

## What It Does

The **AI Resume Analyzer & Job Matcher** is a next-generation talent intelligence and document-evaluation platform. It extracts selectable text from PDF resumes entirely in memory and benchmarks it against any target job description using a **multi-dimensional AI diagnostic engine**.

### 🌟 Unique Architectural Highlights

Unlike basic keyword counters or black-box recruitment bots, this platform introduces:

1. **4 Independent Diagnostic Pillars & Composite Talent Readiness Index (0–100)**:
   - **Pillar 1: TF-IDF Lexical Similarity**: Cosine distance across weighted unigrams and bigrams with stop-word filtering.
   - **Pillar 2: 5-Domain Skill Competency Matrix**: Granular requirement fulfillment across *Languages & Querying*, *AI/ML & NLP*, *Data & Analytics*, *BI & Visualization*, and *Cloud, DevOps & Systems*.
   - **Pillar 3: ATS Readability & Impact Health Audit**: Quantifiable achievement detection (numbers, percentages, multipliers), strong action verb density vs. passive language detection, and contact integrity checks.
   - **Pillar 4: Seniority & Experience Alignment**: Level inference (*Junior*, *Mid*, *Senior*, *Lead*) and requested vs. detected years of experience.
2. **AI Actionable Coaching & STAR Bullet-Point Generator**:
   - For every detected missing skill, the engine automatically synthesizes tailored **STAR (Situation-Task-Action-Result)** project bullet points that candidates can adapt directly into their resumes.
3. **Interactive Keyword Context Inspector**:
   - Select any skill to view exact side-by-side text excerpts from both the resume and the job description.
4. **1-Click Instant Demo Scenarios (Zero Friction UX)**:
   - Instant testing with pre-loaded synthetic personas (*Junior Data Analyst*, *NLP Intern*, and *Pivot Gap Analysis*).
5. **Glassmorphism Tech 2.0 Design System**:
   - Dark cyber aesthetic with ambient multi-point radial glows, SVG circular progress gauges, glowing semantic badges, and live telemetry indicators.
6. **Zero-Persistence Privacy Guarantee**:
   - 100% local, in-memory RAM processing. Uploaded resumes are never saved to disk or transmitted to external third-party cloud APIs.

---

## Multi-Pillar Architecture

```text
PDF Resume ──> In-Memory PDF Extraction (pypdf) ─┐
                                                  ├─> TF-IDF Vectorizer (1,2 n-grams) ──> Text Similarity (%)
Target Job Description ───────────────────────────┤
        │                                         ├─> 5-Domain Skill Taxonomy ─────────> Skill Coverage (%)
        │                                         │
        │                                         ├─> ATS Impact & Verb Audit ─────────> ATS Health Score
        │                                         │
        │                                         ├─> Seniority Level Engine ──────────> Experience Alignment
        │                                         │
        │                                         v
        │                              Composite Readiness Index (0 - 100)
        │                                         │
        v                                         v
Interactive Context Inspector ───> STAR Coaching + JSON Report Export + Glassmorphic UI
```

---

## Folder Structure

```text
resume_job_matcher/
├── app.py                      # Main Streamlit web application & multi-dimensional NLP engine
├── requirements.txt            # Python dependencies (Streamlit, scikit-learn, pypdf, pandas)
├── test_app.py                 # Automated 7-scenario verification test suite
├── README.md                   # Complete architectural guide & project documentation
├── .streamlit/
│   └── config.toml             # Streamlit server & theme configuration
└── data/
    ├── README.md               # Dataset plan, evaluation scenarios, and privacy guidelines
    ├── sample_jobs.csv         # Standardized test job postings with expected skills
    ├── generate_samples.py     # Standalone script to create synthetic test PDFs
    └── samples/                # Pre-generated synthetic PDF resumes
        ├── sample_resume_data_analyst.pdf
        ├── sample_resume_nlp_intern.pdf
        └── sample_resume_software_developer.pdf
```

---

## Setup & Run

Python 3.10 or later is recommended.

### Windows PowerShell

```powershell
cd C:\Users\NIthi\.gemini\antigravity\scratch\resume_job_matcher

# Activate virtual environment (if created) or use Python directly
& "C:\Users\NIthi\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt

# Run the 7-scenario unit test suite
& "C:\Users\NIthi\AppData\Local\Programs\Python\Python311\python.exe" test_app.py

# Launch the Streamlit application
& "C:\Users\NIthi\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
```

### macOS / Linux

```bash
cd /path/to/resume_job_matcher
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_app.py
streamlit run app.py
```

Streamlit is accessible locally at `http://localhost:8501`.

---

## Evaluation & Manual Test Scenarios

The system explicitly passes 7 documented test cases:

1. **High Term Overlap**: Matches high TF-IDF similarity with high skill coverage.
2. **Divergent Phrasing**: Demonstrates why Text Similarity drops while Skill Coverage remains high when vocabulary diverges.
3. **Zero Resume Skills**: Identifies missing skill set with 0% coverage.
4. **No Shared Skills**: Alerts user that 0% coverage does not equal zero competence.
5. **No JD Skills**: Safely handles zero-denominator edge case.
6. **Scanned / Image-Only PDF**: Detects non-selectable PDFs without throwing uncaught exceptions.
7. **Punctuation & Boundary Safety**: Guarantees terms like `C++`, `scikit-learn`, and `Power BI` match accurately without triggering submatches like `Java` in `JavaScript`.

---

## Placement-Ready Resume Description

> **AI Resume Analyzer & Job Matcher — Python, NLP, scikit-learn, Streamlit**  
> Architected a local talent intelligence platform evaluating candidate resumes against job descriptions through a multi-dimensional NLP engine combining TF-IDF cosine similarity, a 5-domain technical skill ontology, and an ATS readability audit. Built an automated STAR bullet-point generator for detected skill gaps, an interactive keyword context inspector, and radial SVG progress gauges in a custom glassmorphism tech UI. Guaranteed complete privacy through zero-disk in-memory buffer processing.
