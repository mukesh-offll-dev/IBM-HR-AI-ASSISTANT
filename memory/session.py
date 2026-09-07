"""
Session Memory Management for AI HR Recruitment Assistant.
Maintains session state for current hiring rounds across multiple turns and candidate uploads.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import BASE_DIR, IS_VERCEL, logger

if IS_VERCEL:
    DB_PATH = Path("/tmp/sessions.db")
else:
    DB_PATH = BASE_DIR / "sessions.db"


class RecruitmentSession:
    """In-memory and SQLite-backed session store for the active recruitment round."""
    def __init__(self, session_id: str = "default_round"):
        self.session_id = session_id
        self.active_jd_title: str = "Senior Backend AI Engineer"
        self.active_jd_text: str = ""
        self.parsed_candidates: Dict[str, Dict[str, Any]] = {}
        self.scored_candidates: Dict[str, Dict[str, Any]] = {}
        self.ranked_candidates: List[Dict[str, Any]] = []
        self.interview_questions: Dict[str, Dict[str, Any]] = {}
        self.chat_history: List[Dict[str, str]] = []
        self._init_db()
        self.load_session()

    def _init_db(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    data_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"SQLite init notice: {e}")

    def save_session(self):
        """Persists session state to SQLite."""
        data = {
            "active_jd_title": self.active_jd_title,
            "active_jd_text": self.active_jd_text,
            "parsed_candidates": self.parsed_candidates,
            "scored_candidates": self.scored_candidates,
            "ranked_candidates": self.ranked_candidates,
            "interview_questions": self.interview_questions,
            "chat_history": self.chat_history
        }
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO session_state (session_id, data_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    data_json = excluded.data_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (self.session_id, json.dumps(data)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Session save notice: {e}")

    def load_session(self):
        """Loads existing session state from SQLite if available."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT data_json FROM session_state WHERE session_id = ?", (self.session_id,))
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                data = json.loads(row[0])
                self.active_jd_title = data.get("active_jd_title", self.active_jd_title)
                self.active_jd_text = data.get("active_jd_text", "")
                self.parsed_candidates = data.get("parsed_candidates", {})
                self.scored_candidates = data.get("scored_candidates", {})
                self.ranked_candidates = data.get("ranked_candidates", [])
                self.interview_questions = data.get("interview_questions", {})
                self.chat_history = data.get("chat_history", [])
        except Exception as e:
            logger.warning(f"Session load notice: {e}")

    def set_job_description(self, title: str, text: str):
        self.active_jd_title = title
        self.active_jd_text = text
        self.save_session()

    def add_candidate(self, candidate_data: Dict[str, Any]):
        name = candidate_data.get("name", f"Candidate_{len(self.parsed_candidates)+1}")
        self.parsed_candidates[name] = candidate_data
        self.save_session()

    def add_scored_candidate(self, scored_data: Dict[str, Any]):
        name = scored_data.get("candidate_name", f"Candidate_{len(self.scored_candidates)+1}")
        self.scored_candidates[name] = scored_data
        self.save_session()

    def update_ranked(self, ranked_list: List[Dict[str, Any]]):
        self.ranked_candidates = ranked_list
        self.save_session()

    def add_questions(self, candidate_name: str, questions_data: Dict[str, Any]):
        self.interview_questions[candidate_name] = questions_data
        self.save_session()

    def append_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})
        self.save_session()

    def reset_round(self):
        self.parsed_candidates.clear()
        self.scored_candidates.clear()
        self.ranked_candidates.clear()
        self.interview_questions.clear()
        self.chat_history.clear()
        self.save_session()


# Global active session
session_manager = RecruitmentSession()
