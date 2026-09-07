"""
LangGraph Node Implementations for AI HR Recruitment Assistant.
Each node logs transitions, interacts with tools/RAG, and updates state.
"""

from typing import Dict, Any, List
from config import logger, get_llm
from agent.state import HRState
from tools.resume_parser import parse_resume
from tools.jd_retriever import retrieve_job_requirements
from tools.scorer import score_candidate
from tools.ranker import rank_candidates
from tools.question_gen import generate_interview_questions
from memory.session import session_manager


def resume_parse_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Parses candidate resume file or text into structured JSON.
    """
    logger.info("[Node: resume_parse_node] Starting resume parsing.")
    file_path = state.get("target_resume_path")
    user_input = state.get("user_input", "")
    target = file_path or user_input

    if not target:
        return {
            "error": "No resume file path or content was provided for screening.",
            "final_output": "Please upload a resume (PDF/DOCX) or provide candidate text to parse."
        }

    try:
        parsed = parse_resume(target, is_raw_text=not bool(file_path))
        candidate_name = parsed.get("name", "Candidate")
        
        parsed_dict = dict(state.get("parsed_candidates", {}))
        parsed_dict[candidate_name] = parsed

        output_msg = (
            f"Successfully parsed resume for **{candidate_name}** ({parsed.get('title', 'Professional')}). "
            f"Extracted {len(parsed.get('skills', []))} skills and {parsed.get('years_of_experience', 0)} years experience."
        )

        return {
            "current_candidate_parsed": parsed,
            "target_candidate_name": candidate_name,
            "parsed_candidates": parsed_dict,
            "final_output": output_msg
        }
    except Exception as e:
        logger.error(f"[Node: resume_parse_node] Error: {e}")
        return {"error": f"Failed to parse resume: {str(e)}"}


def jd_retrieve_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Retrieves relevant JD context and criteria via RAG.
    """
    logger.info("[Node: jd_retrieve_node] Retrieving JD grounding context.")
    cand = state.get("current_candidate_parsed", {})
    skills = cand.get("skills", [])
    query = f"{cand.get('title', '')} {' '.join(skills[:5])}" if skills else state.get("user_input", "core requirements")

    try:
        retrieved_text = retrieve_job_requirements(query, k=4)
        return {
            "retrieved_jd_context": retrieved_text
        }
    except Exception as e:
        logger.warning(f"[Node: jd_retrieve_node] Retrieval warning: {e}")
        return {
            "retrieved_jd_context": state.get("active_jd_text", "")
        }


def match_score_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Evaluates candidate match score, strengths, and gaps against JD.
    """
    logger.info("[Node: match_score_node] Computing candidate match score.")
    cand = state.get("current_candidate_parsed")
    cand_name = state.get("target_candidate_name")

    if not cand:
        parsed_candidates = state.get("parsed_candidates", {})
        if cand_name and cand_name in parsed_candidates:
            cand = parsed_candidates[cand_name]
        elif parsed_candidates:
            # Score first available or most recent
            cand = list(parsed_candidates.values())[-1]
            cand_name = cand.get("name")
        else:
            return {
                "error": "No parsed candidate found to match against JD. Please screen a resume first.",
                "final_output": "Please screen at least one resume before requesting JD matching."
            }

    try:
        jd_text = state.get("active_jd_text") or state.get("retrieved_jd_context", "")
        scored_result = score_candidate(
            candidate_json=cand,
            jd_text=jd_text,
            retrieved_context=state.get("retrieved_jd_context")
        )

        scored_dict = dict(state.get("scored_candidates", {}))
        scored_dict[scored_result["candidate_name"]] = scored_result

        score = scored_result.get("match_score", 0)
        fit = scored_result.get("fit_level", "Evaluated")
        output_msg = (
            f"**Match Assessment for {scored_result['candidate_name']}:**\n"
            f"- Score: **{score}/100** ({fit})\n"
            f"- Recommendation: **{scored_result.get('recommendation', 'N/A')}**\n"
            f"- Key Strengths: {', '.join(scored_result.get('strengths', [])[:3])}\n"
            f"- Gaps Identified: {', '.join(scored_result.get('gaps', [])[:3])}"
        )

        return {
            "scored_candidates": scored_dict,
            "final_output": output_msg
        }
    except Exception as e:
        logger.error(f"[Node: match_score_node] Error: {e}")
        return {"error": f"Failed to score candidate: {str(e)}"}


def rank_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Ranks all scored candidates for the active JD.
    """
    logger.info("[Node: rank_node] Ranking candidates shortlist.")
    scored_candidates = list(state.get("scored_candidates", {}).values())

    if not scored_candidates:
        return {
            "final_output": "No scored candidates are currently available to rank. Please screen and score candidates first."
        }

    try:
        ranked_res = rank_candidates(scored_candidates)
        ranked_list = ranked_res.get("ranked_list", [])

        lines = [f"**Ranked Candidate Shortlist ({len(ranked_list)} Total):**"]
        for c in ranked_list:
            star = "⭐ " if c.get("is_shortlisted") else ""
            lines.append(f"{c['rank']}. {star}**{c['candidate_name']}** — {c.get('match_score', 0)}/100 ({c.get('fit_level')}) | Rec: {c.get('recommendation')}")

        return {
            "ranked_candidates": ranked_list,
            "final_output": "\n".join(lines)
        }
    except Exception as e:
        logger.error(f"[Node: rank_node] Error: {e}")
        return {"error": f"Failed to rank candidates: {str(e)}"}


def question_gen_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Generates customized role-fit and gap-probing interview questions.
    """
    logger.info("[Node: question_gen_node] Generating candidate interview questions.")
    cand_name = state.get("target_candidate_name")
    scored_cands = state.get("scored_candidates", {})
    parsed_cands = state.get("parsed_candidates", {})

    target_cand_scored = None
    target_cand_parsed = None

    if cand_name and cand_name in scored_cands:
        target_cand_scored = scored_cands[cand_name]
    elif scored_cands:
        target_cand_scored = list(scored_cands.values())[-1]
        cand_name = target_cand_scored.get("candidate_name")

    if cand_name and cand_name in parsed_cands:
        target_cand_parsed = parsed_cands[cand_name]

    if not target_cand_scored and not target_cand_parsed:
        return {
            "error": "No candidate specified or available to generate questions for.",
            "final_output": "Please select or screen a candidate before generating interview questions."
        }

    candidate_skills = target_cand_parsed.get("skills", []) if target_cand_parsed else []
    skill_gaps = target_cand_scored.get("gaps", []) if target_cand_scored else []
    role = state.get("active_jd_title", "Software Engineer")

    try:
        q_result = generate_interview_questions(
            candidate_name=cand_name or "Candidate",
            role=role,
            candidate_skills=candidate_skills,
            skill_gaps=skill_gaps,
            jd_text=state.get("active_jd_text")
        )

        q_dict = dict(state.get("interview_questions", {}))
        q_dict[cand_name or "Candidate"] = q_result

        # Format readable markdown summary
        output_lines = [f"### Tailored Interview Guide for **{cand_name}** ({role}):\n"]
        for i, q in enumerate(q_result.get("questions", []), start=1):
            output_lines.append(f"**Q{i} [{q.get('category')}]:** {q.get('question')}")
            output_lines.append(f"- *Objective:* {q.get('objective')}")
            output_lines.append(f"- *Interviewer Rubric:* {q.get('what_to_listen_for')}\n")

        return {
            "interview_questions": q_dict,
            "final_output": "\n".join(output_lines)
        }
    except Exception as e:
        logger.error(f"[Node: question_gen_node] Error: {e}")
        return {"error": f"Failed to generate questions: {str(e)}"}


def general_chat_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Handles general inquiries, recruitment dialogue, or JD questions.
    """
    logger.info("[Node: general_chat_node] Handling general user query.")
    user_input = state.get("user_input", "")
    active_jd_title = state.get("active_jd_title", "Active Job")
    active_jd_text = state.get("active_jd_text", "")
    cand_count = len(state.get("parsed_candidates", {}))
    scored_count = len(state.get("scored_candidates", {}))

    prompt = f"""You are the AI HR Recruitment Assistant.
Current Hiring Context:
- Active Role: {active_jd_title}
- Total Candidates Screened: {cand_count}
- Total Candidates Scored: {scored_count}
- Job Description Summary: {active_jd_text[:800] if active_jd_text else 'No JD provided yet'}

User Question: {user_input}

Provide a helpful, professional, recruitment-focused response. Be concise and actionable.
"""
    try:
        llm = get_llm(temperature=0.3)
        res = llm.invoke(prompt)
        return {"final_output": res.content.strip()}
    except Exception as e:
        return {"final_output": f"Hello! I am your AI HR Recruitment Assistant for {active_jd_title}. How can I assist your hiring pipeline today?"}


def memory_update_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Commits current turn state updates into persistent session memory.
    """
    logger.info("[Node: memory_update_node] Updating session memory.")
    for name, cand in state.get("parsed_candidates", {}).items():
        session_manager.add_candidate(cand)
    for name, scored in state.get("scored_candidates", {}).items():
        session_manager.add_scored_candidate(scored)
    if state.get("ranked_candidates"):
        session_manager.update_ranked(state.get("ranked_candidates", []))
    for name, q in state.get("interview_questions", {}).items():
        session_manager.add_questions(name, q)

    if state.get("user_input"):
        session_manager.append_message("user", state.get("user_input"))
    if state.get("final_output"):
        session_manager.append_message("assistant", state.get("final_output"))

    return {}


def error_handler_node(state: HRState) -> Dict[str, Any]:
    """
    Node: Catches failures gracefully and provides recovery instructions.
    """
    err = state.get("error", "An unknown error occurred during workflow execution.")
    logger.warning(f"[Node: error_handler_node] Handling error: {err}")
    
    fallback_message = (
        f"⚠️ **Notice:** The assistant encountered an issue while processing your request:\n\n"
        f"> *{err}*\n\n"
        f"**Suggested next steps:**\n"
        f"1. Check that uploaded resumes are standard PDF or DOCX formats with readable text.\n"
        f"2. Ensure the active Job Description is populated before scoring.\n"
        f"3. Try repeating your request or selecting the action directly."
    )
    return {
        "final_output": fallback_message,
        "error": None  # clear error state
    }
