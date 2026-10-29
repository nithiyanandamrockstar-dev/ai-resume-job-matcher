"""
test_app.py
Unit and edge-case verification suite for AI Resume Analyzer & Job Matcher.
Tests all 7 scenarios specified in the project evaluation plan.
"""

import unittest
from pathlib import Path
from app import (
    DEFAULT_SKILLS,
    analyze,
    build_skill_pattern,
    calculate_similarity,
    extract_pdf_text,
    find_skills,
)


class TestResumeJobMatcher(unittest.TestCase):

    def setUp(self):
        self.vocab = DEFAULT_SKILLS.copy()

    # -------------------------------------------------------------------------
    # 1. Skill Extraction & Boundary Matching Tests
    # -------------------------------------------------------------------------
    def test_word_boundary_isolation(self):
        """Verify that substring matches like 'java' in 'javascript' do not falsely trigger."""
        text = "I develop front-end interfaces using JavaScript and TypeScript."
        skills = find_skills(text, self.vocab)
        self.assertIn("JavaScript", skills)
        self.assertIn("TypeScript", skills)
        self.assertNotIn("Java", skills, "Java should NOT match inside JavaScript")

    def test_single_letter_boundary(self):
        """Verify that single-letter skills like 'R' do not match inside words like 'linear'."""
        text = "Conducted linear regression modeling and correlation studies."
        skills = find_skills(text, self.vocab)
        self.assertNotIn("R", skills, "'R' should not match inside 'linear'")

        text_with_r = "Experienced in R programming and Python for statistical analysis."
        skills_with_r = find_skills(text_with_r, self.vocab)
        self.assertIn("R", skills_with_r, "Standalone 'R' should be detected")
        self.assertIn("Python", skills_with_r)

    def test_punctuation_and_special_characters(self):
        """Verify skills with symbols like C++, A/B Testing, and scikit-learn match properly."""
        text = "Built backend services in C++ and performed A/B testing using scikit-learn."
        skills = find_skills(text, self.vocab)
        self.assertIn("C++", skills)
        self.assertIn("A/B Testing", skills)
        self.assertIn("Scikit-learn", skills)

    def test_alias_mapping(self):
        """Verify aliases map to canonical names (e.g. 'postgres' -> 'PostgreSQL', 'tf' -> 'TensorFlow')."""
        text = "Deployed models with sklearn and tf connected to a postgres database."
        skills = find_skills(text, self.vocab)
        self.assertIn("Scikit-learn", skills)
        self.assertIn("TensorFlow", skills)
        self.assertIn("PostgreSQL", skills)

    # -------------------------------------------------------------------------
    # 2. Text Similarity Tests
    # -------------------------------------------------------------------------
    def test_similarity_identical_texts(self):
        """Identical texts should yield high cosine similarity (~1.0)."""
        text = "Experienced Python and SQL data analyst building dashboards with Power BI."
        score, top_terms = calculate_similarity(text, text)
        self.assertGreaterEqual(score, 0.95)
        self.assertTrue(len(top_terms) > 0)

    def test_similarity_disjoint_texts(self):
        """Completely unrelated texts should yield near 0 similarity."""
        text1 = "Baroque classical music composition and violin performance."
        text2 = "PostgreSQL database indexing and query optimization with Docker."
        score, _ = calculate_similarity(text1, text2)
        self.assertLess(score, 0.15)

    def test_similarity_empty_or_stop_words(self):
        """Empty texts or stop words only should return 0.0 without throwing errors."""
        score, terms = calculate_similarity("", "")
        self.assertEqual(score, 0.0)
        self.assertEqual(terms, [])

        score2, _ = calculate_similarity("the and or in on at", "the a is by")
        self.assertEqual(score2, 0.0)

    # -------------------------------------------------------------------------
    # 3. Seven Evaluation Scenarios (from Project Specification)
    # -------------------------------------------------------------------------
    def test_scenario_1_high_overlap(self):
        """Scenario 1: Resume and JD share many terms -> similarity rises; coverage reflects shared skills."""
        resume = "Junior Data Analyst with SQL, Python, Excel and Power BI experience. Strong in data analysis."
        jd = "Seeking Junior Data Analyst proficient in SQL, Python, Excel and Power BI for data analysis."
        report = analyze(resume, jd, self.vocab)

        self.assertGreater(report["scores"]["text_similarity_percent"], 40.0)
        self.assertEqual(report["scores"]["skill_coverage_percent"], 100.0)
        self.assertEqual(len(report["skills_breakdown"]["missing_skills"]), 0)

    def test_scenario_2_same_skills_divergent_wording(self):
        """Scenario 2: Same detected skills, but unrelated surrounding wording -> coverage stays high, similarity varies."""
        skills_core = "Proficient in Python, SQL, and Docker."
        jd = f"Corporate enterprise position. Duties involve {skills_core} Agile sprints, client presentations."
        academic_resume = f"University laboratory thesis project. Developed research scripts using {skills_core} Experimental results."

        report = analyze(academic_resume, jd, self.vocab)
        # Skill coverage should be 100% because all JD skills are in the resume
        self.assertEqual(report["scores"]["skill_coverage_percent"], 100.0)
        # Similarity should be moderate rather than near 100% due to phrasing divergence
        self.assertLess(report["scores"]["text_similarity_percent"], 80.0)

    def test_scenario_3_job_has_skills_resume_has_none(self):
        """Scenario 3: Job has detected skills, resume has none -> matched empty, missing all, coverage 0%."""
        resume = "Fictional artist portfolio focusing on oil painting and sculpture history."
        jd = "Hiring software engineer with Python, Docker, and PostgreSQL expertise."

        report = analyze(resume, jd, self.vocab)
        self.assertEqual(report["scores"]["skill_coverage_percent"], 0.0)
        self.assertEqual(len(report["skills_breakdown"]["matched_skills"]), 0)
        self.assertEqual(len(report["skills_breakdown"]["missing_skills"]), 3)

    def test_scenario_4_no_shared_vocabulary_skills(self):
        """Scenario 4: Resume & Job share no listed vocabulary skills -> coverage 0% with warning."""
        resume = "Skilled in Java and C++ microservices."
        jd = "Looking for a Data Analyst with Power BI and Excel reporting skills."

        report = analyze(resume, jd, self.vocab)
        self.assertEqual(report["scores"]["skill_coverage_percent"], 0.0)
        self.assertTrue(any("does NOT prove" in w for w in report["warnings"]))

    def test_scenario_5_job_contains_no_vocabulary_terms(self):
        """Scenario 5: Job contains no vocabulary terms -> warns and reports coverage 0% (zero denominator)."""
        resume = "Experienced in Python, SQL, Docker, and Machine Learning."
        vague_jd = "Looking for a dynamic problem solver who collaborates well and is passionate."

        report = analyze(resume, vague_jd, self.vocab)
        self.assertEqual(report["scores"]["skill_coverage_percent"], 0.0)
        self.assertEqual(report["skills_breakdown"]["total_job_skills_detected"], 0)
        self.assertTrue(any("no denominator exists" in w for w in report["warnings"]))

    def test_scenario_6_pdf_extraction(self):
        """Scenario 6: Test selectable-text PDF extraction using synthetic PDF sample."""
        sample_path = Path(__file__).parent / "data" / "samples" / "sample_resume_data_analyst.pdf"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                text, meta = extract_pdf_text(f.read())
            self.assertGreater(meta["page_count"], 0)
            self.assertFalse(meta["is_empty"])
            self.assertIn("Python", text)
            self.assertIn("SQL", text)

    def test_scenario_7_edge_case_report_export(self):
        """Scenario 7: Full report dictionary contains all required fields for export."""
        report = analyze("Python and SQL", "Python and Excel", self.vocab)
        self.assertIn("metadata", report)
        self.assertIn("scores", report)
        self.assertIn("skills_breakdown", report)
        self.assertIn("text_metrics", report)
        self.assertIn("matched_skills", report["skills_breakdown"])
        self.assertIn("missing_skills", report["skills_breakdown"])
        self.assertIn("additional_resume_skills", report["skills_breakdown"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
