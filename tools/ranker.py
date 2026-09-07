"""
Candidate Ranking Tool.
Sorts scored candidates for a given JD into a prioritized shortlist with statistical insights.
"""

from typing import List, Dict, Any


def rank_candidates(scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    LangChain tool: Sorts and ranks candidates by match score.
    Returns ranked shortlist with rankings, percentiles, and summary stats.
    """
    if not scored_candidates:
        return {
            "ranked_list": [],
            "total_candidates": 0,
            "average_score": 0.0,
            "top_candidate": None,
            "shortlist_count": 0
        }

    # Sort descending by match_score
    sorted_list = sorted(
        scored_candidates,
        key=lambda c: c.get("match_score", 0),
        reverse=True
    )

    total = len(sorted_list)
    ranked_candidates = []
    shortlist = []

    for rank_idx, cand in enumerate(sorted_list, start=1):
        score = cand.get("match_score", 0)
        entry = dict(cand)
        entry["rank"] = rank_idx
        # Percentile rank
        percentile = round(((total - rank_idx) / total) * 100, 1) if total > 1 else 100.0
        entry["percentile"] = percentile
        
        # Qualified for shortlist if score >= 60
        is_shortlisted = score >= 60
        entry["is_shortlisted"] = is_shortlisted
        if is_shortlisted:
            shortlist.append(entry)

        ranked_candidates.append(entry)

    avg_score = round(sum(c.get("match_score", 0) for c in sorted_list) / total, 1)

    return {
        "ranked_list": ranked_candidates,
        "total_candidates": total,
        "average_score": avg_score,
        "top_candidate": ranked_candidates[0]["candidate_name"] if ranked_candidates else None,
        "shortlist_count": len(shortlist),
        "shortlisted_candidates": shortlist
    }
