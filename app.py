"""
AI Resume Analyzer & Job Matcher — Next-Gen AI Talent Intelligence Platform
Final-Year AI & Data Science Flagship Project | Glassmorphism Tech UX | NLP

Architecture:
- In-memory PDF text extraction & ATS layout parsing (pypdf)
- Multi-dimensional evaluation engine:
  1. TF-IDF Cosine Similarity (unigrams & bigrams, stop-words removed)
  2. Domain-Categorized Skill Coverage (5 Competency Domains)
  3. ATS Health & Quantifiable Impact Audit (Metrics, Action Verbs, Formatting)
  4. Seniority & Experience Alignment Analyzer
  5. Composite AI Talent Match Index (0-100)
- Actionable AI Resume Coaching & STAR Bullet-Point Generator
- Context Snippet Inspector (side-by-side keyword occurrences)
- 1-Click Instant Persona Demonstrations (Alex, Samira, Jordan)
- Full Glassmorphic Cyber UI/UX with SVG gauges, interactive pills, and JSON export
"""

import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. COMPREHENSIVE SKILL ONTOLOGY & DOMAIN TAXONOMY
# -----------------------------------------------------------------------------

DEFAULT_SKILLS: Dict[str, List[str]] = {
    # Programming Languages & Querying
    "Python": ["python", "py"],
    "SQL": ["sql", "structured query language"],
    "R": ["r"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "Bash / Shell": ["bash", "shell", "powershell", "sh"],

    # AI, Machine Learning & NLP
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision", "cv"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "Keras": ["keras"],

    # Data Science & Big Data
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "SciPy": ["scipy"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Cleaning": ["data cleaning", "data preprocessing", "data wrangling"],
    "Feature Engineering": ["feature engineering"],
    "Statistics": ["statistics", "statistical analysis", "hypothesis testing"],
    "A/B Testing": ["a/b testing", "ab testing", "split testing"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop"],

    # BI, Visualization & Analytics
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "ms excel", "microsoft excel", "spreadsheets"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Data Visualization": ["data visualization", "visualization", "reporting"],

    # Cloud, DevOps & Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "NoSQL": ["nosql"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "continuous integration", "github actions"],
    "Linux": ["linux", "unix"],
    "Git": ["git", "github", "gitlab"],

    # Web, Architecture & Methodologies
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "Django": ["django"],
    "REST APIs": ["rest api", "restful api", "rest apis", "apis", "api development"],
    "Agile": ["agile", "scrum", "kanban"],
    "Communication": ["communication", "verbal communication", "written communication", "presentation skills"],
}

# 5 Technical Competency Domains
SKILL_DOMAINS: Dict[str, List[str]] = {
    "Languages & Querying": [
        "Python", "SQL", "R", "Java", "C++", "C#", "JavaScript", "TypeScript", "Bash / Shell"
    ],
    "AI, ML & NLP": [
        "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
        "Scikit-learn", "TensorFlow", "PyTorch", "Keras"
    ],
    "Data & Analytics": [
        "Pandas", "NumPy", "SciPy", "Data Analysis", "Data Cleaning",
        "Feature Engineering", "Statistics", "A/B Testing", "Spark", "Hadoop"
    ],
    "BI & Visualization": [
        "Power BI", "Tableau", "Excel", "Matplotlib", "Seaborn", "Data Visualization"
    ],
    "Cloud, DevOps & Systems": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Linux", "Git",
        "PostgreSQL", "MySQL", "MongoDB", "NoSQL", "FastAPI", "Flask", "Django",
        "REST APIs", "Agile", "Communication"
    ]
}

# STAR Bullet-Point Templates for Missing Skills
STAR_TEMPLATES: Dict[str, str] = {
    "Python": "Engineered automated data pipelines and analytical utilities in Python, improving data processing throughput by 40%.",
    "SQL": "Constructed optimized SQL queries and relational database schemas to analyze large datasets and deliver executive metrics.",
    "Power BI": "Designed interactive Power BI business intelligence dashboards with automated refreshes, tracking key retention KPIs.",
    "Excel": "Developed advanced financial and operational models in Microsoft Excel utilizing Pivot Tables, VLOOKUP, and statistical formulas.",
    "Docker": "Containerized multi-service applications using Docker, standardizing reproducible runtime environments and cutting onboarding time by 50%.",
    "PyTorch": "Architected and fine-tuned deep learning neural models in PyTorch, accelerating experiment iterations with GPU acceleration.",
    "Scikit-learn": "Implemented machine learning pipelines with scikit-learn for text classification and clustering, reaching 92% validation F1-score.",
    "Natural Language Processing": "Built NLP preprocessing pipelines featuring tokenization, n-gram extraction, and TF-IDF cosine similarity for text benchmarking.",
    "Git": "Maintained collaborative version control workflows using Git and GitHub, conducting code reviews and managing release branches.",
    "AWS": "Deployed scalable cloud infrastructure and storage buckets on AWS (S3, EC2), reducing server latency and hosting overhead.",
    "FastAPI": "Developed high-throughput asynchronous REST APIs using FastAPI, establishing OpenAPI documentation and unit test coverage.",
    "PostgreSQL": "Architected relational databases in PostgreSQL, creating indexes and views to reduce transaction query times by 35%.",
    "A/B Testing": "Designed randomized A/B experimentation frameworks, evaluating hypothesis metrics with statistical significance testing.",
    "Statistics": "Applied rigorous statistical inference, hypothesis testing, and regression analysis to isolate significant business variables.",
    "Data Analysis": "Conducted exploratory data analysis across multivariate datasets to uncover actionable trends and operational insights.",
    "Data Cleaning": "Engineered automated data cleaning and outlier remediation workflows, elevating raw data fidelity from 78% to 99%.",
    "CI/CD": "Orchestrated automated CI/CD continuous integration pipelines, cutting deployment cycle times from days to under 15 minutes.",
    "Agile": "Collaborated in Agile/Scrum cross-functional sprints, actively participating in daily standups, backlog refinement, and retrospectives.",
}

STRONG_ACTION_VERBS = [
    "spearheaded", "architected", "engineered", "orchestrated", "developed",
    "implemented", "optimized", "automated", "streamlined", "designed",
    "scaled", "deployed", "transformed", "modeled", "analyzed", "delivered",
    "accelerated", "formulated", "constructed", "integrated", "benchmarked"
]

WEAK_PHRASES = [
    "worked on", "responsible for", "assisted with", "helped to",
    "handled", "participated in", "duties included", "familiar with"
]


# -----------------------------------------------------------------------------
# 2. PDF TEXT EXTRACTION & ATS PRE-PROCESSING
# -----------------------------------------------------------------------------

def extract_pdf_text(uploaded_file_or_bytes: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts selectable text from a PDF in-memory using pypdf.
    Does NOT save uploaded files to disk.
    """
    metadata: Dict[str, Any] = {
        "page_count": 0,
        "is_encrypted": False,
        "is_empty": False,
        "error": None,
        "warning": None,
    }

    try:
        if isinstance(uploaded_file_or_bytes, bytes):
            stream = io.BytesIO(uploaded_file_or_bytes)
        elif hasattr(uploaded_file_or_bytes, "read"):
            stream = io.BytesIO(uploaded_file_or_bytes.read())
            if hasattr(uploaded_file_or_bytes, "seek"):
                uploaded_file_or_bytes.seek(0)
        else:
            metadata["error"] = "Unsupported file object provided."
            return "", metadata

        reader = PdfReader(stream)

        if reader.is_encrypted:
            metadata["is_encrypted"] = True
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    metadata["error"] = "The PDF is password-protected or encrypted. Please upload an unencrypted PDF."
                    return "", metadata
            except Exception:
                metadata["error"] = "The PDF is password-protected and could not be decrypted."
                return "", metadata

        num_pages = len(reader.pages)
        metadata["page_count"] = num_pages

        if num_pages == 0:
            metadata["is_empty"] = True
            metadata["error"] = "The PDF file contains 0 pages."
            return "", metadata

        pages_text: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        full_text = "\n\n".join(pages_text).strip()

        if not full_text:
            metadata["is_empty"] = True
            metadata["warning"] = (
                "The PDF was read successfully, but no selectable digital text was found. "
                "The document may be scanned or image-only. "
                "This prototype requires selectable text (OCR is not enabled for privacy/simplicity)."
            )
            return "", metadata

        return full_text, metadata

    except PdfReadError as e:
        metadata["error"] = f"Invalid or corrupted PDF file: {str(e)}"
        return "", metadata
    except Exception as e:
        metadata["error"] = f"Unexpected error while extracting PDF text: {str(e)}"
        return "", metadata


# -----------------------------------------------------------------------------
# 3. ADVANCED NLP & SIMILARITY ENGINES
# -----------------------------------------------------------------------------

def calculate_similarity(
    resume_text: str, job_text: str
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calculates TF-IDF cosine similarity between resume text and job description.
    Uses unigrams and bigrams, with English stop words removed.
    """
    clean_resume = resume_text.strip()
    clean_job = job_text.strip()

    if not clean_resume or not clean_job:
        return 0.0, []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9_+#.-]{2,}\b"
        )
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_job])
        score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

        feature_names = vectorizer.get_feature_names_out()
        v1 = tfidf_matrix[0].toarray()[0]
        v2 = tfidf_matrix[1].toarray()[0]
        product = v1 * v2
        top_indices = product.argsort()[::-1]

        top_terms: List[Tuple[str, float]] = []
        for idx in top_indices:
            if product[idx] > 0:
                top_terms.append((feature_names[idx], float(product[idx])))
            if len(top_terms) >= 12:
                break

        return max(0.0, min(1.0, score)), top_terms

    except ValueError:
        return 0.0, []
    except Exception:
        return 0.0, []


def build_skill_pattern(term: str) -> re.Pattern:
    """Whole-term boundary regex avoiding false positive submatches."""
    escaped = re.escape(term.strip().lower())
    pattern = rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])"
    return re.compile(pattern, re.IGNORECASE)


def find_skills(text: str, skills_vocab: Dict[str, List[str]]) -> Set[str]:
    """Extracts canonical skills from text using whole-term alias matching."""
    if not text or not text.strip():
        return set()

    detected: Set[str] = set()
    for canonical_name, aliases in skills_vocab.items():
        terms_to_check = [canonical_name] + aliases
        for term in terms_to_check:
            if not term.strip():
                continue
            regex = build_skill_pattern(term)
            if regex.search(text):
                detected.add(canonical_name)
                break

    return detected


def find_skill_snippets(text: str, skill_name: str, skills_vocab: Dict[str, List[str]], window: int = 70) -> List[str]:
    """Finds contextual surrounding sentences for a given skill."""
    terms = [skill_name] + skills_vocab.get(skill_name, [])
    snippets: List[str] = []

    for t in terms:
        pattern = rf"(?<![a-zA-Z0-9_]){re.escape(t.lower())}(?![a-zA-Z0-9_])"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            snippet = text[start:end].replace("\n", " ").strip()
            # Clean ellipses
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            snippets.append(f"{prefix}{snippet}{suffix}")
            if len(snippets) >= 2:
                break
        if snippets:
            break

    return snippets


# -----------------------------------------------------------------------------
# 4. ATS HEALTH, SENIORITY & ACTION VERB AUDIT
# -----------------------------------------------------------------------------

def audit_ats_health(resume_text: str) -> Dict[str, Any]:
    """
    Performs an ATS structural, quantifiable metric, and verb strength audit.
    """
    words = resume_text.split()
    word_count = len(words)

    # 1. Quantifiable impact signals (numbers, percentages, currencies, multipliers)
    metric_matches = re.findall(r"\b(?:\d+%(?:\.\d+)?|\$\d+(?:,\d+)*(?:\.\d+)?|\d+x|\d+\+)\b", resume_text, re.IGNORECASE)
    quant_score = min(100, len(metric_matches) * 20)

    # 2. Action verb density
    text_lower = resume_text.lower()
    action_verbs_found = [v for v in STRONG_ACTION_VERBS if re.search(rf"\b{v}\b", text_lower)]
    weak_phrases_found = [w for w in WEAK_PHRASES if re.search(rf"\b{w}\b", text_lower)]

    verb_score = min(100, len(action_verbs_found) * 15)
    if weak_phrases_found:
        verb_score = max(20, verb_score - (len(weak_phrases_found) * 10))

    # 3. Contact information signals
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text))
    has_linkedin_github = bool(re.search(r"\b(linkedin\.com|github\.com|gitlab\.com)\b", text_lower))
    has_phone = bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", resume_text))

    contact_score = 0
    if has_email: contact_score += 40
    if has_linkedin_github: contact_score += 35
    if has_phone: contact_score += 25

    # 4. Length suitability (Optimal 350-900 words for standard tech resumes)
    if 300 <= word_count <= 1000:
        length_score = 100
        length_status = "Optimal Length (1-2 pages)"
    elif word_count < 300:
        length_score = 60
        length_status = "Brief / May lack project depth"
    else:
        length_score = 75
        length_status = "Lengthy / Ensure conciseness"

    composite_ats = int(0.3 * quant_score + 0.3 * verb_score + 0.2 * contact_score + 0.2 * length_score)

    return {
        "overall_ats_score": composite_ats,
        "word_count": word_count,
        "quantifiable_metrics_count": len(metric_matches),
        "metric_examples": metric_matches[:5],
        "strong_action_verbs_count": len(action_verbs_found),
        "action_verbs": action_verbs_found[:6],
        "weak_phrases_count": len(weak_phrases_found),
        "weak_phrases": weak_phrases_found[:4],
        "has_email": has_email,
        "has_portfolio": has_linkedin_github,
        "has_phone": has_phone,
        "length_status": length_status,
    }


def analyze_seniority(resume_text: str, job_text: str) -> Dict[str, Any]:
    """Analyzes seniority level and experience alignment between resume and JD."""
    levels = ["intern", "junior", "entry level", "associate", "mid", "senior", "lead", "principal", "staff"]

    def extract_level(text: str) -> str:
        t = text.lower()
        for lvl in reversed(levels):
            if re.search(rf"\b{lvl}\b", t):
                return lvl.title()
        return "Not Specified"

    def extract_years(text: str) -> Optional[int]:
        m = re.search(r"\b(\d+)\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)\b", text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    jd_level = extract_level(job_text)
    res_level = extract_level(resume_text)
    jd_years = extract_years(job_text)
    res_years = extract_years(resume_text)

    is_aligned = True
    notice = "Seniority level aligns well with role requirements."

    if "Senior" in jd_level or "Lead" in jd_level or "Principal" in jd_level:
        if "Junior" in res_level or "Intern" in res_level:
            is_aligned = False
            notice = f"The job targets a {jd_level} position, whereas the resume indicates an {res_level} profile."

    return {
        "job_level": jd_level,
        "resume_level": res_level,
        "job_years_requested": jd_years,
        "resume_years_detected": res_years,
        "is_aligned": is_aligned,
        "alignment_notice": notice,
    }


# -----------------------------------------------------------------------------
# 5. MULTI-DIMENSIONAL ANALYSIS & TALENT INDEX COMPOSER
# -----------------------------------------------------------------------------

def analyze(
    resume_text: str,
    job_text: str,
    skills_vocab: Dict[str, List[str]],
    pdf_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes complete multi-dimensional AI matching evaluation:
    - Text similarity
    - Skill coverage
    - Domain breakdown
    - ATS health audit
    - Seniority alignment
    - Synthesized Readiness Index
    """
    warnings: List[str] = []

    similarity_score, top_terms = calculate_similarity(resume_text, job_text)

    resume_skills = find_skills(resume_text, skills_vocab)
    job_skills = find_skills(job_text, skills_vocab)

    matched_skills = sorted(list(resume_skills.intersection(job_skills)))
    missing_skills = sorted(list(job_skills - resume_skills))
    additional_skills = sorted(list(resume_skills - job_skills))

    total_job_skills = len(job_skills)
    if total_job_skills > 0:
        skill_coverage = (len(matched_skills) / total_job_skills) * 100.0
    else:
        skill_coverage = 0.0
        warnings.append(
            "The job description did not match any terms from the skill dictionary. "
            "Skill coverage is reported as 0.0% because no denominator exists."
        )

    if total_job_skills > 0 and len(matched_skills) == 0:
        warnings.append(
            "Skill coverage is 0.0%. Note that this does NOT prove the candidate lacks relevant skills; "
            "it indicates that none of the detected job description skills were found in the resume using "
            "the current vocabulary."
        )

    meta = pdf_metadata or {}
    if meta.get("warning"):
        warnings.append(meta["warning"])

    # Domain-by-Domain breakdown
    domain_breakdown: Dict[str, Dict[str, Any]] = {}
    for domain_name, domain_skill_list in SKILL_DOMAINS.items():
        domain_job_skills = [s for s in job_skills if s in domain_skill_list]
        domain_matched = [s for s in matched_skills if s in domain_skill_list]
        domain_missing = [s for s in missing_skills if s in domain_skill_list]

        if domain_job_skills:
            domain_pct = (len(domain_matched) / len(domain_job_skills)) * 100.0
        else:
            domain_pct = 100.0 if not domain_missing else 0.0

        domain_breakdown[domain_name] = {
            "required": domain_job_skills,
            "matched": domain_matched,
            "missing": domain_missing,
            "percentage": round(domain_pct, 1),
        }

    # ATS Health Audit & Seniority Check
    ats_audit = audit_ats_health(resume_text)
    seniority = analyze_seniority(resume_text, job_text)

    # Composite AI Talent Readiness Index (0 - 100)
    # Balanced formulation: 45% Skill Coverage + 25% Text Similarity + 15% ATS Health + 15% Seniority
    seniority_weight = 100.0 if seniority["is_aligned"] else 50.0
    composite_index = (
        0.45 * skill_coverage +
        0.25 * (similarity_score * 100.0) +
        0.15 * ats_audit["overall_ats_score"] +
        0.15 * seniority_weight
    )
    composite_index = max(0.0, min(100.0, composite_index))

    # Actionable Bullet Point Recommendations
    star_recommendations: List[Dict[str, str]] = []
    for skill in missing_skills[:6]:
        template = STAR_TEMPLATES.get(
            skill,
            f"Implemented and managed {skill} workflows to optimize project deliverables and streamline technical documentation."
        )
        star_recommendations.append({"skill": skill, "suggested_bullet": template})

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "project_name": "AI Resume Analyzer & Job Matcher",
            "version": "3.0 (Next-Gen AI Talent Intelligence)",
            "scoring_notice": (
                "Metrics provide transparent decision support. They do not represent automated hiring decisions."
            ),
        },
        "scores": {
            "composite_readiness_index": round(composite_index, 1),
            "text_similarity_percent": round(similarity_score * 100, 2),
            "text_similarity_cosine": round(similarity_score, 4),
            "skill_coverage_percent": round(skill_coverage, 2),
            "ats_health_score": ats_audit["overall_ats_score"],
        },
        "skills_breakdown": {
            "total_job_skills_detected": total_job_skills,
            "total_resume_skills_detected": len(resume_skills),
            "total_matched_skills": len(matched_skills),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "additional_resume_skills": additional_skills,
            "domain_breakdown": domain_breakdown,
        },
        "ats_health": ats_audit,
        "seniority_alignment": seniority,
        "star_recommendations": star_recommendations,
        "text_metrics": {
            "resume_characters": len(resume_text),
            "resume_words": len(resume_text.split()),
            "resume_pages": meta.get("page_count", 0),
            "job_characters": len(job_text),
            "job_words": len(job_text.split()),
        },
        "top_overlapping_ngrams": [term for term, _ in top_terms],
        "warnings": warnings,
    }

    return report


# -----------------------------------------------------------------------------
# 6. GLASSMORHPISM DESIGN SYSTEM CSS
# -----------------------------------------------------------------------------

GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Main Canvas */
.stApp {
    background-color: #070913;
    background-image: 
        radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 90% 15%, rgba(129, 140, 248, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(244, 63, 94, 0.07) 0%, transparent 40%);
    background-attachment: fixed;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #e2e8f0;
}

/* Glass Card */
.glass-card {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    transition: all 0.25s ease;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 15px 35px -10px rgba(99, 102, 241, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.18);
    transform: translateY(-2px);
}

/* Hero Badge */
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #6ee7b7;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
}

.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #34d399;
    animation: pulse-glow 2s infinite;
}

@keyframes pulse-glow {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px #34d399; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 15%, #93c5fd 60%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

/* Preset Persona Button Box */
.persona-btn-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 20px;
}

/* Glass Methodology Banner */
.glass-notice {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-left: 4px solid #6366f1;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* Gauge Container */
.gauge-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 22px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    height: 100%;
}

.gauge-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.7), transparent);
}

/* Skill Badges */
.badge-matched {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 4px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    color: #6ee7b7;
    border: 1px solid rgba(52, 211, 153, 0.35);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    transition: all 0.2s ease;
}
.badge-matched:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(16, 185, 129, 0.3);
}

.badge-missing {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 4px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(244, 63, 94, 0.12);
    color: #fda4af;
    border: 1px solid rgba(251, 113, 133, 0.35);
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.15);
    transition: all 0.2s ease;
}
.badge-missing:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(244, 63, 94, 0.3);
}

.badge-additional {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 4px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(14, 165, 233, 0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0 0 12px rgba(14, 165, 233, 0.15);
    transition: all 0.2s ease;
}

/* STAR Recommendation Card */
.star-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #38bdf8;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* N-Gram Capsule */
.ngram-pill {
    display: inline-block;
    margin: 3px 4px;
    padding: 4px 11px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: rgba(30, 41, 59, 0.6);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Streamlit Widget Styling */
div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
    background: rgba(15, 23, 42, 0.7) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
}

div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 20px rgba(129, 140, 248, 0.3) !important;
}

div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.45);
    border: 1px dashed rgba(129, 140, 248, 0.4);
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8;
    background: rgba(15, 23, 42, 0.6);
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 25px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.5) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

section[data-testid="stSidebar"] {
    background-color: rgba(9, 13, 24, 0.88) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    text-shadow: 0 0 12px rgba(56, 189, 248, 0.5) !important;
}

details[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 12px !important;
}
</style>
"""


def render_radial_gauge(
    percentage: float,
    label: str,
    color_start: str = "#38bdf8",
    color_end: str = "#6366f1",
    subtitle: str = "",
    detail: str = "",
) -> str:
    radius = 54
    circumference = 2 * 3.14159265 * radius
    clamped_pct = max(0.0, min(100.0, percentage))
    stroke_offset = circumference * (1.0 - clamped_pct / 100.0)

    grad_id = f"grad_{re.sub(r'[^a-zA-Z0-9]', '', label)}"

    return f"""
    <div class="gauge-card">
        <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#94a3b8; margin-bottom:12px;">
            {label}
        </div>
        <div style="position:relative; width:135px; height:135px; margin:0 auto;">
            <svg viewBox="0 0 130 130" style="width:100%; height:100%; transform:rotate(-90deg);">
                <defs>
                    <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="{color_start}" />
                        <stop offset="100%" stop-color="{color_end}" />
                    </linearGradient>
                </defs>
                <circle cx="65" cy="65" r="{radius}" fill="none" stroke="rgba(255, 255, 255, 0.06)" stroke-width="10" />
                <circle cx="65" cy="65" r="{radius}" fill="none" stroke="url(#{grad_id})" stroke-width="10"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}"
                    stroke-linecap="round"
                    style="transition: stroke-dashoffset 1s ease-in-out; filter: drop-shadow(0 0 8px {color_start}66);" />
            </svg>
            <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <span style="font-size:1.85rem; font-weight:800; color:#ffffff; letter-spacing:-0.03em;">
                    {percentage:.1f}<span style="font-size:1rem; color:#94a3b8;">%</span>
                </span>
                <span style="font-size:0.75rem; color:#cbd5e1; font-weight:500;">{subtitle}</span>
            </div>
        </div>
        <div style="font-size:0.78rem; color:#94a3b8; margin-top:12px; line-height:1.4;">
            {detail}
        </div>
    </div>
    """


# -----------------------------------------------------------------------------
# 7. STREAMLIT APPLICATION CONTROLLER
# -----------------------------------------------------------------------------

def load_sample_jobs() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "data" / "sample_jobs.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    return pd.DataFrame([
        {
            "role_id": 1,
            "title": "Junior Data Analyst",
            "job_description": "Use SQL, Python, Excel and Power BI to clean datasets, build interactive dashboards, and deliver statistical summaries. Candidates should possess solid communication skills and familiarity with data analysis and A/B testing methodologies.",
            "expected_skills": "sql|python|excel|power bi|data analysis|communication|statistics|a/b testing"
        },
        {
            "role_id": 2,
            "title": "Junior NLP Intern",
            "job_description": "Build NLP projects with Python and scikit-learn. Develop text classification, tokenization, and similarity models using TF-IDF, PyTorch, or Hugging Face. Familiarity with Git, pandas, and data preprocessing required.",
            "expected_skills": "python|natural language processing|scikit-learn|pytorch|git|pandas|data cleaning"
        },
        {
            "role_id": 3,
            "title": "Full-Stack Python Developer",
            "job_description": "Develop and maintain scalable web backends using Python, FastAPI, PostgreSQL, and Docker. Experience with Git version control, CI/CD pipelines, and writing unit tests. Solid understanding of REST APIs and Agile collaboration.",
            "expected_skills": "python|fastapi|postgresql|docker|git|ci/cd|rest apis|agile"
        }
    ])


def get_sample_resumes() -> Dict[str, Path]:
    sample_dir = Path(__file__).parent / "data" / "samples"
    if not sample_dir.exists():
        return {}
    return {f.stem.replace("_", " ").title(): f for f in sample_dir.glob("*.pdf")}


def init_session_state():
    if "skills_vocab" not in st.session_state:
        st.session_state["skills_vocab"] = {k: list(v) for k, v in DEFAULT_SKILLS.items()}
    if "job_description_input" not in st.session_state:
        st.session_state["job_description_input"] = ""
    if "preset_loaded" not in st.session_state:
        st.session_state["preset_loaded"] = None
    if "trigger_analysis" not in st.session_state:
        st.session_state["trigger_analysis"] = False


def main():
    st.set_page_config(
        page_title="AI Resume Analyzer & Job Matcher | Glassmorphism Tech",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_session_state()

    # Inject Glassmorphism Design System CSS
    st.markdown(GLASS_CSS, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------------------
    header_col1, header_col2 = st.columns([0.7, 0.3])
    with header_col1:
        st.markdown(
            '<div class="live-pill"><span class="live-dot"></span>NEXT-GEN AI ENGINE | IN-MEMORY TALENT INTELLIGENCE</div>',
            unsafe_allow_html=True
        )
        st.markdown('<h1 class="hero-title">AI Resume Analyzer & Job Matcher</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#94a3b8; font-size:1.05rem; margin-bottom:14px;">'
            'Multi-dimensional NLP architecture: TF-IDF vector similarity, domain skill coverage, ATS health audit & STAR coaching'
            '</div>',
            unsafe_allow_html=True
        )
    with header_col2:
        st.markdown(
            """
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:16px 20px; text-align:right;">
                <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Analysis Pillars</div>
                <div style="font-size:0.95rem; font-weight:700; color:#38bdf8;">TF-IDF + Domain Ontology + ATS</div>
                <div style="font-size:0.75rem; color:#6ee7b7; margin-top:4px;">🔒 100% In-Memory RAM Sandbox</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # ZERO-FRICTION 1-CLICK INSTANT DEMO PERSONAS
    # -------------------------------------------------------------------------
    sample_df = load_sample_jobs()
    sample_resumes = get_sample_resumes()

    st.markdown(
        """
        <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; margin-bottom:8px;">
            ⚡ 1-Click Instant Demo Scenarios (Experience the app instantly)
        </div>
        """,
        unsafe_allow_html=True
    )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("📊 Demo A: Junior Data Analyst (High Match)", use_container_width=True):
            st.session_state["preset_loaded"] = "sample_resume_data_analyst.pdf"
            da_job = sample_df[sample_df["title"].str.contains("Data Analyst", case=False, na=False)]
            if not da_job.empty:
                st.session_state["job_description_input"] = da_job.iloc[0]["job_description"]
            st.session_state["trigger_analysis"] = True
            st.rerun()

    with p_col2:
        if st.button("🤖 Demo B: Junior NLP Intern (AI Research)", use_container_width=True):
            st.session_state["preset_loaded"] = "sample_resume_nlp_intern.pdf"
            nlp_job = sample_df[sample_df["title"].str.contains("NLP", case=False, na=False)]
            if not nlp_job.empty:
                st.session_state["job_description_input"] = nlp_job.iloc[0]["job_description"]
            st.session_state["trigger_analysis"] = True
            st.rerun()

    with p_col3:
        if st.button("🔄 Demo C: Developer to Analyst (Pivot Gap)", use_container_width=True):
            st.session_state["preset_loaded"] = "sample_resume_software_developer.pdf"
            da_job = sample_df[sample_df["title"].str.contains("Data Analyst", case=False, na=False)]
            if not da_job.empty:
                st.session_state["job_description_input"] = da_job.iloc[0]["job_description"]
            st.session_state["trigger_analysis"] = True
            st.rerun()

    # Methodology Callout
    st.markdown(
        """
        <div class="glass-notice">
            <div style="font-size:0.95rem; font-weight:700; color:#ffffff; margin-bottom:4px;">
                💡 Deliberate Multi-Pillar Scoring Methodology
            </div>
            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.5;">
                Unlike black-box screening bots, this platform reports <b>four transparent, independent indicators</b>:
                <b>(1) Overall Talent Readiness Index</b>, <b>(2) TF-IDF Text Similarity</b>, <b>(3) Skill Coverage %</b>, and <b>(4) ATS Health Score</b>.
                <br>
                <span style="color:#93c5fd;">All analyses run locally in-memory. Designed for candidate decision support rather than automated filtering.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: Editable Vocabulary & Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom:16px;">
                <div style="font-size:1.2rem; font-weight:800; color:#ffffff; display:flex; align-items:center; gap:8px;">
                    <span>⚙️</span> Skill Dictionary
                </div>
                <div style="font-size:0.8rem; color:#94a3b8;">Whole-term regex matching with alias normalization</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        vocab = st.session_state["skills_vocab"]
        st.metric("Active Recognized Skills", len(vocab))

        with st.expander("➕ Add Custom Skill", expanded=False):
            new_skill_name = st.text_input("Canonical Name (e.g. Rust, PySpark)", key="new_skill_name").strip()
            new_aliases_input = st.text_input("Aliases (comma-separated)", key="new_aliases").strip()

            if st.button("Add to Vocabulary", use_container_width=True):
                if new_skill_name:
                    aliases = [a.strip().lower() for a in new_aliases_input.split(",") if a.strip()]
                    st.session_state["skills_vocab"][new_skill_name] = aliases
                    st.success(f"Added '{new_skill_name}' with {len(aliases)} alias(es)!")
                    st.rerun()
                else:
                    st.warning("Please specify a skill name.")

        with st.expander("📖 Browse Active Vocabulary", expanded=False):
            search_query = st.text_input("Filter skills:", "").lower()
            filtered_vocab = {
                k: v for k, v in sorted(vocab.items())
                if search_query in k.lower() or any(search_query in a.lower() for a in v)
            }
            vocab_list = [f"**{k}**: {', '.join(v) if v else '*(none)*'}" for k, v in filtered_vocab.items()]
            st.markdown("\n\n".join(vocab_list) if vocab_list else "*(No matches found)*")

        if st.button("Reset Vocabulary to Default", use_container_width=True):
            st.session_state["skills_vocab"] = {k: list(v) for k, v in DEFAULT_SKILLS.items()}
            st.success("Vocabulary reset to default.")
            st.rerun()

        vocab_json = json.dumps(st.session_state["skills_vocab"], indent=2)
        st.download_button(
            label="Download Vocabulary (JSON)",
            data=vocab_json,
            file_name="skills_vocabulary.json",
            mime="application/json",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="background:rgba(15,23,42,0.4); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px;">
                <div style="font-size:0.8rem; font-weight:700; color:#38bdf8; margin-bottom:4px;">🛡️ Privacy Protocol</div>
                <div style="font-size:0.75rem; color:#94a3b8; line-height:1.4;">
                    Uploaded PDFs are parsed strictly into memory. No resume data is stored on disk or sent over the internet.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # MAIN INPUT SECTION (2 COLUMNS)
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    resume_text = ""
    pdf_meta: Dict[str, Any] = {}

    with col_left:
        st.markdown(
            """
            <div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin-bottom:12px; display:flex; align-items:center; gap:8px;">
                <span>📄</span> 1. Candidate Resume (PDF)
            </div>
            """,
            unsafe_allow_html=True
        )

        input_mode = st.radio(
            "Select Resume Source:",
            ["Upload PDF Document", "Use Preloaded Synthetic Resume"],
            index=1 if st.session_state.get("preset_loaded") else 0,
            horizontal=True
        )

        if input_mode == "Upload PDF Document":
            uploaded_pdf = st.file_uploader(
                "Upload a text-based PDF resume (Max 10MB)",
                type=["pdf"],
                help="Requires digital selectable text. Scanned images will be flagged."
            )
            if uploaded_pdf is not None:
                if uploaded_pdf.size > 10 * 1024 * 1024:
                    st.error("File size exceeds 10MB limit. Please upload a smaller file.")
                else:
                    with st.spinner("Extracting text in-memory..."):
                        resume_text, pdf_meta = extract_pdf_text(uploaded_pdf)
                        if pdf_meta.get("error"):
                            st.error(f"❌ {pdf_meta['error']}")
                        elif pdf_meta.get("warning"):
                            st.warning(f"⚠️ {pdf_meta['warning']}")
                        else:
                            st.success(f"✓ Extracted {len(resume_text.split())} words across {pdf_meta.get('page_count')} page(s).")
        else:
            if sample_resumes:
                sample_keys = list(sample_resumes.keys())
                default_idx = 0
                if st.session_state.get("preset_loaded"):
                    for i, k in enumerate(sample_keys):
                        if sample_resumes[k].name == st.session_state["preset_loaded"]:
                            default_idx = i
                            break

                selected_sample_name = st.selectbox("Choose a synthetic sample resume:", sample_keys, index=default_idx)
                if selected_sample_name:
                    sample_path = sample_resumes[selected_sample_name]
                    with open(sample_path, "rb") as f:
                        resume_text, pdf_meta = extract_pdf_text(f.read())
                    st.success(f"✓ Loaded '{selected_sample_name}' ({len(resume_text.split())} words).")
            else:
                st.info("No sample PDFs found in data/samples/.")

        if resume_text:
            with st.expander("👁️ Inspect Extracted Resume Text", expanded=False):
                st.text_area("Extracted Text", resume_text, height=180, disabled=True)

    with col_right:
        st.markdown(
            """
            <div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin-bottom:12px; display:flex; align-items:center; gap:8px;">
                <span>🎯</span> 2. Target Job Description
            </div>
            """,
            unsafe_allow_html=True
        )

        sample_options = ["(Paste custom job description)"] + [
            f"{row['title']} (ID: {row['role_id']})" for _, row in sample_df.iterrows()
        ]

        selected_sample_job = st.selectbox(
            "Quick-load a sample job role:",
            sample_options
        )

        default_text = st.session_state["job_description_input"]
        if selected_sample_job != "(Paste custom job description)":
            role_title = selected_sample_job.split(" (ID:")[0]
            matched_row = sample_df[sample_df["title"] == role_title]
            if not matched_row.empty:
                default_text = matched_row.iloc[0]["job_description"]

        job_text = st.text_area(
            "Paste target job description text:",
            value=default_text,
            height=250,
            placeholder="e.g. Seeking a Junior Data Analyst proficient in SQL, Python, Excel and Power BI..."
        )

    # -------------------------------------------------------------------------
    # ANALYSIS TRIGGER
    # -------------------------------------------------------------------------
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ RUN DEEP AI TALENT MATCH ANALYSIS", type="primary", use_container_width=True)

    # Check auto-trigger from preset
    do_run = analyze_btn or st.session_state.get("trigger_analysis", False)
    st.session_state["trigger_analysis"] = False

    if do_run:
        if not resume_text.strip():
            st.error("Please provide a resume with selectable text before running analysis.")
            return

        if not job_text.strip():
            st.error("Please provide a target job description.")
            return

        with st.spinner("Processing TF-IDF matrix, domain taxonomy & ATS health audit..."):
            report = analyze(
                resume_text=resume_text,
                job_text=job_text,
                skills_vocab=st.session_state["skills_vocab"],
                pdf_metadata=pdf_meta,
            )

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        scores = report["scores"]
        skills_info = report["skills_breakdown"]
        ats_info = report["ats_health"]
        seniority = report["seniority_alignment"]

        # ---------------------------------------------------------------------
        # TOP SUMMARY HERO CARD WITH EXECUTIVE STATUS
        # ---------------------------------------------------------------------
        c_index = scores["composite_readiness_index"]
        if c_index >= 80:
            status_badge = '<span style="background:rgba(16,185,129,0.2); color:#6ee7b7; border:1px solid #10b981; padding:4px 12px; border-radius:9999px; font-weight:700; font-size:0.85rem;">🏆 EXCELLENT MATCH</span>'
            status_text = "The resume demonstrates strong keyword coverage, high vocabulary alignment, and solid ATS formatting."
        elif c_index >= 55:
            status_badge = '<span style="background:rgba(56,189,248,0.2); color:#7dd3fc; border:1px solid #38bdf8; padding:4px 12px; border-radius:9999px; font-weight:700; font-size:0.85rem;">⚡ COMPETITIVE CANDIDATE</span>'
            status_text = "Good core qualification alignment. Adding missing technologies will notably enhance candidate positioning."
        else:
            status_badge = '<span style="background:rgba(244,63,94,0.2); color:#fda4af; border:1px solid #f43f5e; padding:4px 12px; border-radius:9999px; font-weight:700; font-size:0.85rem;">⚠️ SIGNIFICANT SKILL GAP</span>'
            status_text = "Several key role requirements are missing. Review the STAR tailoring recommendations below."

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; color:#94a3b8; letter-spacing:0.05em;">AI Executive Summary</div>
                        <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:4px 0;">{status_badge}</div>
                        <div style="font-size:0.95rem; color:#cbd5e1;">{status_text}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:#94a3b8; font-weight:600;">Overall Match Index</div>
                        <div style="font-size:2.6rem; font-weight:800; color:#38bdf8; line-height:1;">{c_index}<span style="font-size:1.2rem; color:#94a3b8;">/100</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------------------------------------------------------
        # FOUR RADIAL PROGRESS GAUGES
        # ---------------------------------------------------------------------
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)

        with g_col1:
            st.markdown(
                render_radial_gauge(
                    percentage=scores["composite_readiness_index"],
                    label="Talent Readiness Index",
                    color_start="#38bdf8",
                    color_end="#818cf8",
                    subtitle="Weighted AI Index",
                    detail="Synthesizes skills, text similarity, ATS health & seniority.",
                ),
                unsafe_allow_html=True
            )

        with g_col2:
            st.markdown(
                render_radial_gauge(
                    percentage=scores["skill_coverage_percent"],
                    label="Skill Coverage",
                    color_start="#10b981",
                    color_end="#06b6d4",
                    subtitle=f"{skills_info['total_matched_skills']} / {skills_info['total_job_skills_detected']}",
                    detail="Percentage of detected job requirements satisfied in resume.",
                ),
                unsafe_allow_html=True
            )

        with g_col3:
            st.markdown(
                render_radial_gauge(
                    percentage=scores["text_similarity_percent"],
                    label="TF-IDF Text Similarity",
                    color_start="#c084fc",
                    color_end="#ec4899",
                    subtitle=f"{scores['text_similarity_cosine']:.3f} cosine",
                    detail="Measures overall unigram/bigram document vector angle.",
                ),
                unsafe_allow_html=True
            )

        with g_col4:
            st.markdown(
                render_radial_gauge(
                    percentage=float(ats_info["overall_ats_score"]),
                    label="ATS Health Score",
                    color_start="#f59e0b",
                    color_end="#10b981",
                    subtitle=ats_info["length_status"].split("/")[0].strip(),
                    detail=f"{ats_info['quantifiable_metrics_count']} metrics, {ats_info['strong_action_verbs_count']} action verbs.",
                ),
                unsafe_allow_html=True
            )

        if report["warnings"]:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            for warn in report["warnings"]:
                st.warning(f"⚠️ {warn}")

        # ---------------------------------------------------------------------
        # DOMAIN-BY-DOMAIN COMPETENCY RADAR / BARS
        # ---------------------------------------------------------------------
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:1.3rem; font-weight:700; color:#ffffff; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
                <span>📊</span> 5-Domain Competency Matrix
            </div>
            """,
            unsafe_allow_html=True
        )

        dom_cols = st.columns(5)
        domains = skills_info["domain_breakdown"]
        for idx, (dom_name, dom_data) in enumerate(domains.items()):
            with dom_cols[idx]:
                req_cnt = len(dom_data["required"])
                match_cnt = len(dom_data["matched"])
                pct = dom_data["percentage"]
                bar_color = "#10b981" if pct >= 75 else ("#38bdf8" if pct >= 50 else "#f43f5e")

                st.markdown(
                    f"""
                    <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:16px 14px; text-align:center;">
                        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#94a3b8; height:32px; display:flex; align-items:center; justify-content:center;">
                            {dom_name}
                        </div>
                        <div style="font-size:1.5rem; font-weight:800; color:#ffffff; margin:8px 0 4px 0;">{pct:.0f}%</div>
                        <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:10px;">{match_cnt}/{req_cnt} matched</div>
                        <div style="background:rgba(255,255,255,0.08); border-radius:9999px; height:6px; overflow:hidden;">
                            <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:9999px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ---------------------------------------------------------------------
        # SKILL DETECTION BREAKDOWN (GLOWING PILLS)
        # ---------------------------------------------------------------------
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:1.3rem; font-weight:700; color:#ffffff; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
                <span>🎯</span> Skill Gap & Requirement Alignment
            </div>
            """,
            unsafe_allow_html=True
        )

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.markdown(f"**✅ Matched Skills ({len(skills_info['matched_skills'])})**")
            st.caption("Shared by resume and job description:")
            if skills_info["matched_skills"]:
                badges = "".join([f'<span class="badge-matched">✓ {s}</span>' for s in skills_info["matched_skills"]])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748b; font-size:0.85rem;'>*(No matched skills detected)*</span>", unsafe_allow_html=True)

        with b_col2:
            st.markdown(f"**❌ Missing Skills ({len(skills_info['missing_skills'])})**")
            st.caption("Required by job but not found in resume:")
            if skills_info["missing_skills"]:
                badges = "".join([f'<span class="badge-missing">✗ {s}</span>' for s in skills_info["missing_skills"]])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748b; font-size:0.85rem;'>*(Zero missing skills! Perfect requirement coverage)*</span>", unsafe_allow_html=True)

        with b_col3:
            st.markdown(f"**➕ Additional Resume Skills ({len(skills_info['additional_resume_skills'])})**")
            st.caption("Bonus competencies in resume not requested in JD:")
            if skills_info["additional_resume_skills"]:
                badges = "".join([f'<span class="badge-additional">+ {s}</span>' for s in skills_info["additional_resume_skills"]])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748b; font-size:0.85rem;'>*(None detected)*</span>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # ACTIONABLE AI RESUME COACHING & STAR BULLET GENERATOR
        # ---------------------------------------------------------------------
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:1.3rem; font-weight:700; color:#ffffff; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
                <span>✨</span> AI Resume Tailoring Coach & STAR Bullet Generator
            </div>
            """,
            unsafe_allow_html=True
        )

        if report["star_recommendations"]:
            st.caption("To bridge the identified skill gaps, integrate these STAR (Situation-Task-Action-Result) bullet point templates into your projects:")
            for rec in report["star_recommendations"]:
                st.markdown(
                    f"""
                    <div class="star-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-size:0.85rem; font-weight:700; color:#38bdf8;">Missing Skill: {rec['skill']}</span>
                            <span style="font-size:0.75rem; color:#94a3b8; background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:6px;">STAR Template</span>
                        </div>
                        <div style="font-size:0.92rem; color:#f1f5f9; line-height:1.5;">
                            "{rec['suggested_bullet']}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("🎉 Outstanding! Your resume already encompasses all key technical competencies required by this role.")

        # ---------------------------------------------------------------------
        # INTERACTIVE CONTEXT SNIPPET INSPECTOR
        # ---------------------------------------------------------------------
        with st.expander("🔎 Interactive Keyword Context Inspector (See where terms appear)", expanded=False):
            all_detected = sorted(list(set(skills_info["matched_skills"] + skills_info["missing_skills"])))
            if all_detected:
                selected_term = st.selectbox("Select a detected skill to inspect context:", all_detected)
                if selected_term:
                    snip_col1, snip_col2 = st.columns(2)
                    with snip_col1:
                        st.markdown(f"**Occurrence in Resume:**")
                        r_snippets = find_skill_snippets(resume_text, selected_term, st.session_state["skills_vocab"])
                        if r_snippets:
                            for s in r_snippets:
                                st.info(f"📄 {s}")
                        else:
                            st.markdown("<span style='color:#f87171;'>✗ Not found in resume</span>", unsafe_allow_html=True)
                    with snip_col2:
                        st.markdown(f"**Occurrence in Job Description:**")
                        j_snippets = find_skill_snippets(job_text, selected_term, st.session_state["skills_vocab"])
                        if j_snippets:
                            for s in j_snippets:
                                st.success(f"🎯 {s}")
                        else:
                            st.markdown("<span style='color:#94a3b8;'>- Not specified in job description</span>", unsafe_allow_html=True)
            else:
                st.write("No vocabulary skills detected.")

        # ---------------------------------------------------------------------
        # DEEP INSPECTION TABS (ATS HEALTH, TOP N-GRAMS, RAW TEXT, JSON)
        # ---------------------------------------------------------------------
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 ATS Health & Audit",
            "🔍 Score Explanation & Insights",
            "🔤 Top TF-IDF N-Grams",
            "📄 Raw Document Texts",
            "💾 Export JSON Report"
        ])

        with tab1:
            st.markdown("### ATS Parsing & Readability Diagnostic")
            ats_c1, ats_c2 = st.columns(2)
            with ats_c1:
                st.markdown(f"**Length Status:** {ats_info['length_status']} ({ats_info['word_count']} words)")
                st.markdown(f"**Quantifiable Metrics Detected:** {ats_info['quantifiable_metrics_count']}")
                if ats_info["metric_examples"]:
                    st.caption(f"Examples found: {', '.join(ats_info['metric_examples'])}")

                st.markdown(f"**Contact Integrity:**")
                st.markdown(f"- Email: {'✅ Yes' if ats_info['has_email'] else '❌ Missing'}")
                st.markdown(f"- Portfolio / LinkedIn / GitHub: {'✅ Yes' if ats_info['has_portfolio'] else '⚠️ Recommended'}")
                st.markdown(f"- Phone: {'✅ Yes' if ats_info['has_phone'] else '❌ Missing'}")

            with ats_c2:
                st.markdown(f"**Action Verb Strength ({ats_info['strong_action_verbs_count']} detected):**")
                if ats_info["action_verbs"]:
                    st.write(", ".join([f"`{v}`" for v in ats_info["action_verbs"]]))
                else:
                    st.caption("No strong action verbs identified. Consider starting project bullet points with verbs like 'Spearheaded', 'Engineered', 'Orchestrated'.")

                if ats_info["weak_phrases_count"] > 0:
                    st.warning(f"Found {ats_info['weak_phrases_count']} passive or weak phrases ({', '.join(ats_info['weak_phrases'])}). Upgrade these to decisive verbs.")

        with tab2:
            st.markdown("### Why Do These Independent Metrics Differ?")
            st.markdown(
                f"""
                - **Composite Talent Readiness Index ({scores['composite_readiness_index']}/100)**: A holistic weighted metric uniting requirements match (45%), semantic phrasing similarity (25%), ATS formatting hygiene (15%), and seniority level alignment (15%).
                - **Text Similarity ({scores['text_similarity_percent']}%)**: Evaluates sentence structures, formatting, project narratives, and general vocabulary using TF-IDF cosine vector angles. If a resume describes SQL in an academic lab setting while the JD describes SQL in an e-commerce setting, similarity drops even if technical keywords match.
                - **Skill Coverage ({scores['skill_coverage_percent']}%)**: Evaluates explicit keyword intersection against detected job requirements (`{skills_info['total_matched_skills']} ÷ {skills_info['total_job_skills_detected']}`).
                - **Decision Support Only**: This platform equips candidates to diagnose gaps and tailor resumes objectively. It must never be used as an automated rejection filter.
                """
            )

        with tab3:
            st.markdown("### Top Overlapping N-Grams Influencing Cosine Similarity")
            top_ngrams = report["top_overlapping_ngrams"]
            if top_ngrams:
                st.write("The following unigrams and bigrams had the highest TF-IDF vector weight product across both documents:")
                pills = "".join([f'<span class="ngram-pill">{ngram}</span>' for ngram in top_ngrams])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.write("*(No significant overlapping n-grams after English stop-word filtering)*")

        with tab4:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Resume Extracted Text**")
                st.text_area("Resume Text", resume_text, height=280, disabled=True, key="res_tab_txt")
            with col_t2:
                st.markdown("**Target Job Description**")
                st.text_area("Job Description Text", job_text, height=280, disabled=True, key="job_tab_txt")

        with tab5:
            st.markdown("### Evaluation Report (JSON)")
            json_str = json.dumps(report, indent=2)
            st.code(json_str, language="json")

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Comprehensive JSON Report",
                data=json_str,
                file_name=f"talent_match_report_{timestamp_str}.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
