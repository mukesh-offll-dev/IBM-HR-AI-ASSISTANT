"""
Router Node for LangGraph.
Classifies user intent into HR recruitment workflows.
"""

import re
from typing import Dict, Any
from config import get_llm, logger
from agent.state import HRState


INTENTS = ["screen_resume", "match_jd", "rank_candidates", "generate_questions", "general_chat"]


def route_intent(state: HRState) -> Dict[str, Any]:
    """
    Router node: inspects user query or parameters and routes to the appropriate branch.
    """
    user_input = (state.get("user_input") or "").strip()
    explicit_intent = (state.get("intent") or "").strip()

    if explicit_intent in INTENTS:
        logger.info(f"[Router Node] Using explicit intent: '{explicit_intent}'")
        return {"intent": explicit_intent}

    # Fast pattern matching before calling LLM
    lower = user_input.lower()
    if any(w in lower for w in ["rank", "shortlist", "top candidate", "who is the best", "sort candidate"]):
        intent = "rank_candidates"
    elif any(w in lower for w in ["question", "interview", "prep question", "probe gap"]):
        intent = "generate_questions"
    elif any(w in lower for w in ["score", "match", "compare", "evaluate fit", "qualification"]):
        intent = "match_jd"
    elif any(w in lower for w in ["screen", "parse resume", "extract cv", "upload resume"]):
        intent = "screen_resume"
    else:
        # LLM based intent classifier
        try:
            llm = get_llm(temperature=0.0)
            prompt = f"""You are the routing controller for an AI HR Recruitment Assistant.
Given the user input, classify it into EXACTLY ONE of the following intent labels:
- screen_resume (user wants to upload, parse, or extract info from resumes)
- match_jd (user wants to match or score a candidate against the job description)
- rank_candidates (user wants to rank, sort, or see top candidates/shortlist)
- generate_questions (user wants interview questions for a candidate)
- general_chat (general hiring conversation, greetings, help, or queries about the JD)

User Input: "{user_input}"

Respond with ONLY the exact intent label from the list above. No quotes, no markdown, no other words.
"""
            resp = llm.invoke(prompt)
            classified = resp.content.strip().lower()
            # Clean punctuation
            classified = re.sub(r"[^\w_]", "", classified)
            intent = classified if classified in INTENTS else "general_chat"
        except Exception as e:
            logger.warning(f"[Router Node] Classification error ({e}), defaulting to general_chat.")
            intent = "general_chat"

    logger.info(f"[Router Node] Routed to: '{intent}' for input: '{user_input[:60]}'")
    return {"intent": intent}
