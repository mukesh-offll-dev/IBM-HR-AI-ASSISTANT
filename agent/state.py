"""
LangGraph State Schema for the AI HR Recruitment Assistant.
Carries the conversation context, active JD, candidates list, and routing metadata.
"""

from typing import TypedDict, List, Dict, Any, Optional


class HRState(TypedDict, total=False):
    # Intent / Routing
    user_input: str
    intent: str  # "screen_resume" | "match_jd" | "rank_candidates" | "generate_questions" | "general_chat"
    
    # Active Job Description
    active_jd_title: str
    active_jd_text: str
    retrieved_jd_context: str
    
    # Candidate payloads
    target_resume_path: Optional[str]
    target_candidate_name: Optional[str]
    current_candidate_parsed: Optional[Dict[str, Any]]
    
    # Accumulated State across turns
    parsed_candidates: Dict[str, Dict[str, Any]]
    scored_candidates: Dict[str, Dict[str, Any]]
    ranked_candidates: List[Dict[str, Any]]
    interview_questions: Dict[str, Dict[str, Any]]
    
    # Conversation & Responses
    messages: List[Dict[str, str]]
    final_output: str
    error: Optional[str]
