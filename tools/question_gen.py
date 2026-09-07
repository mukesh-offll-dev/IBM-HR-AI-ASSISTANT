"""
Interview Question Generator Tool.
Generates tailored interview questions (role-fit + gap-probing) per candidate based on JD criteria.
"""

import json
import re
from typing import Dict, Any, List, Optional
from config import get_llm, logger


def generate_interview_questions(
    candidate_name: str,
    role: str = "Backend AI Engineer",
    candidate_skills: Optional[List[str]] = None,
    skill_gaps: Optional[List[str]] = None,
    jd_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    LangChain tool: Generates customized interview questions.
    Produces a mix of role-fit questions and deep-dive gap-probing questions with interviewer rubrics.
    """
    candidate_skills = candidate_skills or []
    skill_gaps = skill_gaps or []
    
    logger.info(f"Generating interview questions for candidate: {candidate_name} (Role: {role})")
    llm = get_llm(temperature=0.3)

    prompt = f"""You are a Principal Hiring Manager designing an interview guide for:
Candidate Name: {candidate_name}
Target Role: {role}

Candidate Strengths/Known Skills:
{', '.join(candidate_skills) if candidate_skills else 'Standard technical background'}

Identified Skill Gaps / Weaknesses / Missing Qualifications:
{', '.join(skill_gaps) if skill_gaps else 'None identified'}

Job Context & Requirements:
{(jd_text or '')[:2000]}

Generate 5 high-impact, professional interview questions:
1. 2 Role-Fit & Architecture Questions (evaluating primary engineering competencies)
2. 2 Gap-Probing Questions (tactfully probing the specific identified gaps/missing areas to see if they can ramp up or have transferable experience)
3. 1 Behavioral / Problem-Solving Scenario Question

Respond ONLY with a valid JSON object matching the schema below:
{{
  "candidate_name": "{candidate_name}",
  "role": "{role}",
  "questions": [
    {{
      "category": "Role-Fit / Core Technical",
      "question": "Question text here?",
      "objective": "What competency this tests",
      "what_to_listen_for": "Key signs of a strong answer vs. weak answer"
    }},
    {{
      "category": "Role-Fit / Architecture",
      "question": "Question text here?",
      "objective": "What competency this tests",
      "what_to_listen_for": "Key signs of a strong answer vs. weak answer"
    }},
    {{
      "category": "Gap-Probing (Targeted)",
      "question": "Question specifically targeting one of the identified gaps?",
      "objective": "Verifying depth or potential in the identified gap area",
      "what_to_listen_for": "Honest assessment of limitations and how they bridge knowledge gaps"
    }},
    {{
      "category": "Gap-Probing (Targeted)",
      "question": "Second question probing another gap or missing skill?",
      "objective": "Verifying practical understanding despite lack of formal resume mention",
      "what_to_listen_for": "Evidence of curiosity and rapid learning capability"
    }},
    {{
      "category": "Behavioral & Conflict/Trade-off",
      "question": "Scenario-based question?",
      "objective": "Teamwork, communication, and engineering trade-offs",
      "what_to_listen_for": "Clear communication and structured decision-making"
    }}
  ]
}}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

        result = json.loads(content)
        result["candidate_name"] = candidate_name
        return result

    except Exception as e:
        logger.warning(f"LLM question generation notice ({e}), using standard template.")
        # Fallback question set
        return {
            "candidate_name": candidate_name,
            "role": role,
            "questions": [
                {
                    "category": "Role-Fit / Core Technical",
                    "question": f"Can you walk us through how you design high-performance backend systems in Python for {role}?",
                    "objective": "Assess core technical depth and architectural habits.",
                    "what_to_listen_for": "Mentions of asynchronous programming, data validation, and clean API design."
                },
                {
                    "category": "Role-Fit / Architecture",
                    "question": "How have you implemented RAG pipelines or vector database lookups to minimize latency and hallucinations?",
                    "objective": "Examine practical RAG & retrieval knowledge.",
                    "what_to_listen_for": "Chunking strategies, embedding trade-offs, and evaluation of retrieval accuracy."
                },
                {
                    "category": "Gap-Probing (Targeted)",
                    "question": f"Our role emphasizes {skill_gaps[0] if skill_gaps else 'agentic workflows'}. How would you approach learning or implementing this in production?",
                    "objective": "Evaluate adaptability and bridge strategy for missing background.",
                    "what_to_listen_for": "Self-learning methodology, understanding fundamental concepts."
                },
                {
                    "category": "Gap-Probing (Targeted)",
                    "question": f"Have you had exposure to {skill_gaps[1] if len(skill_gaps) > 1 else 'Model Context Protocol (MCP)'} or similar protocols?",
                    "objective": "Probe secondary gap.",
                    "what_to_listen_for": "Curiosity and eagerness to work with emerging standards."
                },
                {
                    "category": "Behavioral & Trade-offs",
                    "question": "Tell us about a time you had to balance shipping an AI feature quickly versus ensuring strict output accuracy.",
                    "objective": "Engineering maturity and risk management.",
                    "what_to_listen_for": "Pragmatic balance between speed, safety guardrails, and user experience."
                }
            ]
        }
