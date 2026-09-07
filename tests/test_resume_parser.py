"""
Tests for Resume Parser Tool.
"""

from pathlib import Path
import pytest
from tools.resume_parser import extract_text_from_file, parse_resume, parse_resume_heuristics


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_extract_text_from_pdf():
    pdf_path = SAMPLE_DIR / "alex_morgan_resume.pdf"
    assert pdf_path.exists(), "Sample PDF must exist"
    text = extract_text_from_file(pdf_path)
    assert len(text) > 50
    assert "Alex Morgan" in text or "ALEX MORGAN" in text
    assert "Python" in text


def test_extract_text_from_docx():
    docx_path = SAMPLE_DIR / "priya_sharma_resume.docx"
    assert docx_path.exists(), "Sample DOCX must exist"
    text = extract_text_from_file(docx_path)
    assert len(text) > 50
    assert "Priya Sharma" in text
    assert "Django" in text


def test_parse_resume_heuristics():
    sample_text = """
    John Doe
    john.doe@example.com | (555) 123-4567
    Senior Python Engineer with 6 years experience in FastAPI and Docker.
    Skills: Python, FastAPI, Docker, PostgreSQL
    """
    result = parse_resume_heuristics(sample_text)
    assert result["name"] != ""
    assert result["email"] == "john.doe@example.com"
    assert result["years_of_experience"] == 6.0
    assert "Python" in result["skills"]


def test_parse_resume_direct():
    sample_text = """
    Jane Smith
    jane.smith@techcorp.io | +1 555-987-6543
    Lead AI Engineer with 8 years experience building LangGraph and RAG workflows.
    """
    result = parse_resume(sample_text, is_raw_text=True)
    assert "name" in result
    assert "skills" in result
    assert "years_of_experience" in result
