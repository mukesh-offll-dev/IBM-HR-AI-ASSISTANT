"""
Candidate Scoring Tool.
Evaluates structured candidate data against a Job Description using LLM + RAG grounding.
"""

import json
import re
from typing import Dict, Any, Optional
from config import get_llm, logger
from rag.retriever import get_jd_context


def score_candidate(
    candidate_json: Dict[str, Any],
    jd_text: Optional[str] = None,
    retrieved_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    LangChain tool: Compares a candidate against a Job Description.
    Returns a match score (0-100), fit level, strengths, gaps breakdown, and hiring recommendation.
    """
    candidate_name = candidate_json.get("name", "Candidate")
    skills = candidate_json.get("skills", [])
    experience_yrs = candidate_json.get("years_of_experience", 0)
    title = candidate_json.get("title", "")
    summary = candidate_json.get("raw_summary", "")

    # Retrieve context via RAG if not passed
    if not retrieved_context:
        query_terms = f"{title} {' '.join(skills[:8])}"
        retrieved_context = get_jd_context(query_terms, k=3)

    if not jd_text:
        from rag.ingestor import rag_manager
        jd_text = rag_manager.active_jd_full_text or retrieved_context

    logger.info(f"Scoring candidate '{candidate_name}' against JD.")
    llm = get_llm(temperature=0.1)

    prompt = f"""You are an unbiased, highly analytical Senior Technical Recruiter.
Evaluate this candidate against the provided Job Description and requirements.

GROUNDING REQUIREMENTS:
You must strictly base your evaluation on the provided Job Description requirements and the candidate's actual qualifications.

Candidate Data:
- Name: {candidate_name}
- Current/Recent Title: {title}
- Total Years Experience: {experience_yrs}
- Skills: {', '.join(skills)}
- Education: {', '.join(candidate_json.get('education', []))}
- Certifications: {', '.join(candidate_json.get('certifications', []))}
- Summary: {summary}

Job Description & Retrieved Criteria:
{jd_text[:3000]}

Respond ONLY with a valid JSON object matching the following structure (no markdown fences, no extra text):
{{
  "candidate_name": "{candidate_name}",
  "match_score": 85,
  "fit_level": "Strong Match",
  "score_breakdown": {{
    "skills_alignment": 35,
    "experience_depth": 25,
    "education_and_certs": 12,
    "domain_relevance": 13
  }},
  "strengths": [
    "Specific matching skill or experience item 1",
    "Specific matching skill or experience item 2",
    "Specific matching skill or experience item 3"
  ],
  "gaps": [
    "Specific missing requirement or shortfall 1",
    "Specific missing requirement or shortfall 2"
  ],
  "reasoning": "A concise 2-3 sentence grounded justification explaining the score.",
  "recommendation": "Advance to Interview"
}}

Rules for "fit_level":
- 80-100: "Strong Match"
- 60-79: "Moderate Match"
- 40-59: "Low Match"
- 0-39: "Not Qualified"

Rules for "recommendation":
- "Advance to Technical Interview", "Consider for Phone Screening", or "Do Not Advance"
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

        result = json.loads(content)
        # Ensure candidate_name is preserved
        result["candidate_name"] = candidate_name
        result["raw_candidate"] = candidate_json
        return result

    except Exception as e:
        logger.warning(f"Error during LLM candidate scoring ({e}), falling back to heuristic scoring.")
        # Fallback heuristic calculation
        skills_lower = [s.lower() for s in skills]
        jd_lower = (jd_text or "").lower()
        matched = [s for s in skills if s.lower() in jd_lower]
        missing = [s for s in ["python", "fastapi", "langchain", "docker", "rag", "mcp"] if s not in skills_lower and s in jd_lower]
        
        ratio = len(matched) / max(1, len(matched) + len(missing))
        score = int(min(95, max(25, ratio * 70 + (20 if experience_yrs >= 3 else 10))))
        
        fit = "Strong Match" if score >= 80 else ("Moderate Match" if score >= 60 else "Low Match")
        rec = "Advance to Technical Interview" if score >= 75 else ("Consider for Phone Screening" if score >= 55 else "Do Not Advance")

        return {
            "candidate_name": candidate_name,
            "match_score": score,
            "fit_level": fit,
            "score_breakdown": {
                "skills_alignment": int(score * 0.4),
                "experience_depth": int(score * 0.3),
                "education_and_certs": int(score * 0.15),
                "domain_relevance": int(score * 0.15)
            },
            "strengths": [f"Demonstrated proficiency in {s}" for s in matched[:4]] or ["Relevant background in software development"],
            "gaps": [f"Missing explicit experience in {g.upper()}" for g in missing[:3]] or ["Could benefit from deeper domain alignment"],
            "reasoning": f"Candidate demonstrates {score}% alignment based on verified skill overlap and {experience_yrs} years of experience.",
            "recommendation": rec,
            "raw_candidate": candidate_json
        }
