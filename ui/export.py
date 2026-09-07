"""
Export utilities for the AI HR Recruitment Assistant.
Generates CSV and formatted reports of candidate evaluations and shortlists.
"""

import csv
import io
from typing import List, Dict, Any


def generate_shortlist_csv(ranked_candidates: List[Dict[str, Any]]) -> str:
    """
    Generates a structured CSV of the ranked candidate shortlist.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Rank",
        "Candidate Name",
        "Current Title",
        "Match Score (0-100)",
        "Fit Level",
        "Recommendation",
        "Years of Experience",
        "Top Strengths",
        "Identified Gaps",
        "Reasoning"
    ])

    for c in ranked_candidates:
        raw = c.get("raw_candidate", {})
        strengths = "; ".join(c.get("strengths", []))
        gaps = "; ".join(c.get("gaps", []))
        writer.writerow([
            c.get("rank", ""),
            c.get("candidate_name", ""),
            raw.get("title", ""),
            c.get("match_score", 0),
            c.get("fit_level", ""),
            c.get("recommendation", ""),
            raw.get("years_of_experience", ""),
            strengths,
            gaps,
            c.get("reasoning", "")
        ])

    return output.getvalue()
