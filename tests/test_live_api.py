"""
Live API End-to-End Test for AI HR Recruitment Assistant.
"""

import urllib.request
import urllib.parse
import json
import sys

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def post_json(url, data=None):
    body = json.dumps(data).encode("utf-8") if data is not None else b"{}"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def get_text(url):
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def main():
    print("--- 1. Testing Sample Candidates Load ---")
    samples = post_json("http://127.0.0.1:8000/api/resumes/load-samples")
    print(f"Loaded: {samples.get('loaded')} candidates: {samples.get('candidates')}")

    print("\n--- 2. Testing Evaluation & Ranking ---")
    eval_res = post_json("http://127.0.0.1:8000/api/evaluate/all")
    print(f"Total Scored: {eval_res.get('total_scored')}")
    shortlist = eval_res.get("ranked_shortlist", [])
    for c in shortlist:
        print(f"  #{c.get('rank')}: {c.get('candidate_name')} | Score: {c.get('match_score')}/100 ({c.get('fit_level')}) | Rec: {c.get('recommendation')}")

    print("\n--- 3. Testing Interview Question Generation for Top Candidate ---")
    if shortlist:
        top_name = shortlist[0].get("candidate_name", "Alex Morgan")
        url = f"http://127.0.0.1:8000/api/candidates/{urllib.parse.quote(top_name)}/interview-questions"
        q_res = post_json(url)
        print(f"Generated {len(q_res.get('questions', []))} interview questions for {top_name}:")
        for i, q in enumerate(q_res.get("questions", [])[:2], 1):
            print(f"  Q{i} [{q.get('category')}]: {q.get('question')}")

    print("\n--- 4. Testing CSV Export ---")
    csv_text = get_text("http://127.0.0.1:8000/api/export/csv")
    lines = csv_text.strip().splitlines()
    print(f"CSV exported successfully with {len(lines)} lines (Header + {len(lines)-1} candidates).")

    print("\n--- 5. Testing Agent Chat ---")
    chat_res = post_json("http://127.0.0.1:8000/api/agent/chat", {"message": "Who is our highest ranked candidate and why?"})
    print("Agent Chat Response:")
    print(chat_res.get("response", "")[:300])

    print("\n ALL LIVE API TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
