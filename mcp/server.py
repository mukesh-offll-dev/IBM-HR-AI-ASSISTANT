"""
Model Context Protocol (MCP) Server for AI HR Recruitment Assistant.
Exposes recruitment tools (score_candidate, rank_candidates, generate_interview_questions)
to any MCP-compatible client (Claude Desktop, Cursor, Antigravity, or custom MCP clients).
"""

import json
from typing import List
from config import logger
from tools.scorer import score_candidate
from tools.ranker import rank_candidates
from tools.question_gen import generate_interview_questions

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        MCPServer = None

# Initialize MCP Server
if MCPServer is not None:
    mcp_server = MCPServer("AI_HR_Recruitment_Assistant")
else:
    mcp_server = None


if mcp_server is not None:
    @mcp_server.tool()
    def mcp_score_candidate(
        candidate_name: str,
        skills_csv: str,
        years_of_experience: float,
        jd_text: str
    ) -> str:
        """
        Evaluates and scores a candidate against a job description.
        Returns match score (0-100), fit level, strengths, and gap analysis.
        """
        skills = [s.strip() for s in skills_csv.split(",") if s.strip()]
        candidate_json = {
            "name": candidate_name,
            "skills": skills,
            "years_of_experience": years_of_experience,
            "title": "Applicant",
            "education": [],
            "certifications": [],
            "raw_summary": f"{candidate_name} with {years_of_experience} yrs experience in {', '.join(skills[:4])}"
        }
        result = score_candidate(candidate_json=candidate_json, jd_text=jd_text)
        return json.dumps(result, indent=2)

    @mcp_server.tool()
    def mcp_rank_candidates(scored_candidates_json: str) -> str:
        """
        Ranks a list of scored candidates into a prioritized shortlist for a given job.
        Input should be a JSON array of scored candidate objects.
        """
        try:
            candidates = json.loads(scored_candidates_json)
            if isinstance(candidates, dict) and "candidates" in candidates:
                candidates = candidates["candidates"]
            result = rank_candidates(candidates)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Invalid JSON or ranking error: {str(e)}"})

    @mcp_server.tool()
    def mcp_generate_interview_questions(
        candidate_name: str,
        role: str,
        skill_gaps_csv: str
    ) -> str:
        """
        Generates role-specific and targeted gap-probing interview questions for a candidate.
        """
        gaps = [g.strip() for g in skill_gaps_csv.split(",") if g.strip()]
        result = generate_interview_questions(
            candidate_name=candidate_name,
            role=role,
            skill_gaps=gaps
        )
        return json.dumps(result, indent=2)


def run_mcp_server():
    """Starts the MCP server via standard I/O."""
    if mcp_server is None:
        raise RuntimeError("MCP library is not installed.")
    logger.info("Starting AI HR Recruitment Assistant MCP Server...")
    mcp_server.run()


if __name__ == "__main__":
    run_mcp_server()
