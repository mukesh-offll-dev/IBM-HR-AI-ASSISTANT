"""
Tests for Candidate Ranker Tool.
"""

from tools.ranker import rank_candidates


def test_rank_candidates_sorting():
    candidates = [
        {"candidate_name": "Low Match Cand", "match_score": 45, "fit_level": "Low Match"},
        {"candidate_name": "High Match Cand", "match_score": 92, "fit_level": "Strong Match"},
        {"candidate_name": "Mid Match Cand", "match_score": 70, "fit_level": "Moderate Match"}
    ]

    ranked = rank_candidates(candidates)

    assert ranked["total_candidates"] == 3
    assert ranked["top_candidate"] == "High Match Cand"
    ranked_list = ranked["ranked_list"]
    assert ranked_list[0]["candidate_name"] == "High Match Cand"
    assert ranked_list[0]["rank"] == 1
    assert ranked_list[1]["candidate_name"] == "Mid Match Cand"
    assert ranked_list[1]["rank"] == 2
    assert ranked_list[2]["candidate_name"] == "Low Match Cand"
    assert ranked_list[2]["rank"] == 3
    assert ranked["average_score"] == 69.0


def test_rank_candidates_empty():
    ranked = rank_candidates([])
    assert ranked["total_candidates"] == 0
    assert ranked["ranked_list"] == []
