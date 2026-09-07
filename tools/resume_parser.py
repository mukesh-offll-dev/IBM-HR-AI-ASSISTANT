"""
Resume Parser Tool.
Extracts raw text from PDF/DOCX/TXT files and structures it into standardized candidate JSON using LLM.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Union
import fitz  # PyMuPDF
import docx
from pydantic import BaseModel, Field

from config import get_llm, logger


class CandidateData(BaseModel):
    name: str = Field(default="Unknown Candidate", description="Candidate's full name")
    email: str = Field(default="", description="Contact email")
    phone: str = Field(default="", description="Contact phone")
    title: str = Field(default="", description="Professional title or current role")
    skills: list[str] = Field(default_factory=list, description="Extracted technical and soft skills")
    years_of_experience: float = Field(default=0.0, description="Total estimated years of relevant professional experience")
    education: list[str] = Field(default_factory=list, description="Degrees, institutions, graduation years")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications")
    past_roles: list[dict] = Field(default_factory=list, description="Past roles with title, company, and duration")
    raw_summary: str = Field(default="", description="Brief executive summary of the candidate")


def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """Extracts raw text from PDF, DOCX, or TXT file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found at: {path}")

    suffix = path.suffix.lower()
    text = ""

    if suffix == ".pdf":
        try:
            doc = fitz.open(str(path))
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
    elif suffix == ".docx":
        doc = docx.Document(str(path))
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text += " | ".join(row_text) + "\n"
    elif suffix in [".txt", ".md"]:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .pdf, .docx, .txt")

    clean_text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if not clean_text:
        raise ValueError(f"No extractable text found in file {path.name}. File may be scanned image or corrupted.")
    return clean_text


def parse_resume_heuristics(raw_text: str) -> Dict[str, Any]:
    """Fallback rule-based extractor if LLM parsing encounters issues."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    candidate_name = lines[0] if lines else "Candidate"
    
    # Extract email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    email = email_match.group(0) if email_match else ""
    
    # Extract phone
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
    phone = phone_match.group(0) if phone_match else ""
    
    # Extract years of experience
    exp_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", raw_text, re.IGNORECASE)
    years = max([float(m) for m in exp_matches], default=2.0)
    
    return {
        "name": candidate_name,
        "email": email,
        "phone": phone,
        "title": "Software Professional",
        "skills": ["Python", "FastAPI", "Git", "REST APIs"],
        "years_of_experience": years,
        "education": ["Computer Science / Related Field"],
        "certifications": [],
        "past_roles": [],
        "raw_summary": raw_text[:300] + "..."
    }


def parse_resume(file_path_or_text: Union[str, Path], is_raw_text: bool = False) -> Dict[str, Any]:
    """
    LangChain tool-compatible function: parses a resume into structured JSON.
    Takes either a file path or raw resume text.
    """
    try:
        if is_raw_text or not Path(str(file_path_or_text)).exists():
            raw_text = str(file_path_or_text)
            source_name = "Direct Input"
        else:
            raw_text = extract_text_from_file(file_path_or_text)
            source_name = Path(file_path_or_text).name

        logger.info(f"Parsing resume for: {source_name} ({len(raw_text)} chars)")
        llm = get_llm(temperature=0.0)

        prompt = f"""You are an expert HR Recruitment AI.
Extract structured candidate details from the following resume text.
Respond ONLY with a valid, raw JSON object matching the schema below. Do not include markdown code blocks, backticks, or any additional conversational text.

Schema:
{{
  "name": "Full Name",
  "email": "Email Address",
  "phone": "Phone Number",
  "title": "Current or primary professional title",
  "skills": ["Skill1", "Skill2", ...],
  "years_of_experience": 5.0,
  "education": ["Degree, Major, Institution, Year"],
  "certifications": ["Certification Name"],
  "past_roles": [
    {{"role": "Title", "company": "Company Name", "duration": "Dates/Duration", "key_achievements": "Summary"}}
  ],
  "raw_summary": "Concise 2-sentence summary of candidate profile"
}}

Resume Text:
{raw_text[:4000]}
"""
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Clean potential markdown wrapping
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

        # Parse JSON
        parsed = json.loads(content)
        parsed["source_file"] = source_name
        return parsed

    except Exception as e:
        logger.warning(f"LLM parsing encountered error ({e}), falling back to heuristic parsing.")
        fallback = parse_resume_heuristics(raw_text if 'raw_text' in locals() else str(file_path_or_text))
        fallback["source_file"] = str(file_path_or_text)
        fallback["error_note"] = f"Heuristic parsed: {str(e)}"
        return fallback
