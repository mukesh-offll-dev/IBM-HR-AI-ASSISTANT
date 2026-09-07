"""
Main FastAPI Application for AI HR Recruitment Assistant.
Combines LangGraph Agent, RAG Pipeline, Document Ingestion, and Modern Web UI.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    HOST,
    PORT,
    TEMP_UPLOAD_DIR,
    BASE_DIR,
    logger
)
from memory.session import session_manager
from rag.ingestor import rag_manager
from tools.resume_parser import parse_resume, extract_text_from_file
from tools.scorer import score_candidate
from tools.ranker import rank_candidates
from tools.question_gen import generate_interview_questions
from agent.graph import execute_agent_workflow
from ui.export import generate_shortlist_csv

app = FastAPI(
    title="AI HR Recruitment Assistant",
    description="Agentic AI Recruitment Platform with LangChain, LangGraph, RAG, and MCP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class JDUploadRequest(BaseModel):
    title: str
    text: str


class ChatMessageRequest(BaseModel):
    message: str


class QuestionGenRequest(BaseModel):
    candidate_name: str
    role: Optional[str] = None


# Embedded Modern HTML/CSS/JS UI
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI HR Recruitment Assistant — Agentic AI Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-secondary: #111827;
      --bg-card: rgba(22, 30, 49, 0.75);
      --bg-card-hover: rgba(30, 41, 67, 0.85);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(99, 102, 241, 0.5);
      
      --accent-cyan: #00d2ff;
      --accent-indigo: #6366f1;
      --accent-purple: #a855f7;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-primary);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(0, 210, 255, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(168, 85, 247, 0.08) 0px, transparent 60%);
      background-attachment: fixed;
      color: var(--text-main);
      font-family: var(--font-body);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Navbar */
    header {
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent-indigo), var(--accent-cyan));
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }

    .brand-title h1 {
      font-family: var(--font-display);
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #ffffff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-title span {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: var(--accent-cyan);
      font-weight: 600;
    }

    .header-badges {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .pill-badge {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-emerald);
      box-shadow: 0 0 10px var(--accent-emerald);
    }

    /* Main Container */
    main {
      flex: 1;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 32px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* Tabs Navigation */
    .tabs-nav {
      display: flex;
      gap: 8px;
      background: rgba(17, 24, 39, 0.6);
      padding: 6px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
      width: fit-content;
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 600;
      padding: 10px 20px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .tab-btn:hover {
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.04);
    }

    .tab-btn.active {
      color: #ffffff;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(0, 210, 255, 0.6));
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    /* Tab Contents */
    .tab-pane {
      display: none;
      animation: fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .tab-pane.active {
      display: block;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Glass Cards */
    .glass-card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-glass);
      position: relative;
    }

    .card-title {
      font-family: var(--font-display);
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .card-subtitle {
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 20px;
    }

    /* Stat Cards Row */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }

    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      position: relative;
      overflow: hidden;
    }

    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: linear-gradient(180deg, var(--accent-indigo), var(--accent-cyan));
    }

    .stat-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-dim);
      font-weight: 600;
    }

    .stat-val {
      font-family: var(--font-display);
      font-size: 32px;
      font-weight: 800;
      color: #ffffff;
    }

    .stat-sub {
      font-size: 12px;
      color: var(--accent-cyan);
    }

    /* Forms, Buttons, Inputs */
    .form-group {
      margin-bottom: 18px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    input[type="text"], textarea {
      width: 100%;
      background: rgba(10, 14, 23, 0.7);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      color: var(--text-main);
      padding: 12px 16px;
      font-family: var(--font-body);
      font-size: 14px;
      transition: all 0.2s;
    }

    input[type="text"]:focus, textarea:focus {
      outline: none;
      border-color: var(--accent-indigo);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }

    .btn {
      background: linear-gradient(135deg, var(--accent-indigo), #4f46e5);
      color: #ffffff;
      border: none;
      padding: 12px 24px;
      border-radius: var(--radius-sm);
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      box-shadow: none;
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
      box-shadow: none;
    }

    .btn-emerald {
      background: linear-gradient(135deg, var(--accent-emerald), #059669);
      box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }

    .btn-emerald:hover {
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
    }

    .btn-group {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }

    /* Drag and Drop Zone */
    .drop-zone {
      border: 2px dashed rgba(99, 102, 241, 0.4);
      border-radius: var(--radius-md);
      padding: 36px 24px;
      text-align: center;
      background: rgba(99, 102, 241, 0.03);
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--accent-cyan);
      background: rgba(0, 210, 255, 0.06);
    }

    .drop-zone-icon {
      font-size: 40px;
      margin-bottom: 10px;
      color: var(--accent-cyan);
    }

    /* Candidate Table */
    .table-container {
      overflow-x: auto;
      margin-top: 16px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }

    th {
      background: rgba(17, 24, 39, 0.8);
      color: var(--text-muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-color);
    }

    td {
      padding: 16px 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-size: 14px;
      vertical-align: middle;
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .score-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-family: var(--font-mono);
      font-size: 13px;
    }

    .score-high {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .score-mid {
      background: rgba(245, 158, 11, 0.15);
      color: var(--accent-amber);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .score-low {
      background: rgba(244, 63, 94, 0.15);
      color: var(--accent-rose);
      border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .tag-chip {
      display: inline-block;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 2px 8px;
      font-size: 11px;
      margin-right: 4px;
      margin-bottom: 4px;
      color: var(--text-muted);
    }

    .tag-strength {
      background: rgba(16, 185, 129, 0.1);
      border-color: rgba(16, 185, 129, 0.2);
      color: #6ee7b7;
    }

    .tag-gap {
      background: rgba(244, 63, 94, 0.1);
      border-color: rgba(244, 63, 94, 0.2);
      color: #fda4af;
    }

    /* Modal / Drawer */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      display: none;
      justify-content: center;
      align-items: center;
      z-index: 200;
      padding: 20px;
    }

    .modal-overlay.active {
      display: flex;
    }

    .modal-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      max-width: 840px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 32px;
      position: relative;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .modal-close {
      position: absolute;
      top: 24px;
      right: 24px;
      background: rgba(255, 255, 255, 0.08);
      border: none;
      color: var(--text-muted);
      width: 36px;
      height: 36px;
      border-radius: 50%;
      font-size: 18px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .modal-close:hover {
      color: #ffffff;
      background: rgba(255, 255, 255, 0.15);
    }

    /* Chat Window */
    .chat-box {
      height: 400px;
      overflow-y: auto;
      padding: 16px;
      background: rgba(10, 14, 23, 0.6);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-bottom: 16px;
    }

    .chat-bubble {
      max-width: 80%;
      padding: 12px 18px;
      border-radius: 14px;
      font-size: 14px;
      line-height: 1.5;
    }

    .chat-user {
      align-self: flex-end;
      background: linear-gradient(135deg, var(--accent-indigo), #4f46e5);
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }

    .chat-assistant {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      border-bottom-left-radius: 2px;
    }

    /* Loading Spinner */
    .spinner {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #ffffff;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    footer {
      text-align: center;
      padding: 24px;
      color: var(--text-dim);
      font-size: 12px;
      border-top: 1px solid var(--border-color);
      margin-top: auto;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">⚡</div>
      <div class="brand-title">
        <h1>AI HR Recruitment Assistant</h1>
        <span>LangChain • LangGraph • RAG • MCP</span>
      </div>
    </div>
    <div class="header-badges">
      <div class="pill-badge" id="activeJdBadge">
        <span class="dot"></span>
        <span id="headerJdTitle">Loading JD...</span>
      </div>
      <button class="btn btn-secondary" style="padding: 6px 14px; font-size: 12px;" onclick="resetRound()">
        🔄 Reset Session
      </button>
    </div>
  </header>

  <main>
    <!-- Navigation Tabs -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('dashboard')">📊 Ranking & Shortlist</button>
      <button class="tab-btn" onclick="switchTab('resumes')">📁 Resumes & Screening</button>
      <button class="tab-btn" onclick="switchTab('jd')">📝 Job Description (RAG)</button>
      <button class="tab-btn" onclick="switchTab('chat')">💬 Agent Dialogue</button>
    </div>

    <!-- TAB 1: DASHBOARD & RANKING -->
    <div id="tab-dashboard" class="tab-pane active">
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-label">Candidates Screened</span>
          <span class="stat-val" id="statScreened">0</span>
          <span class="stat-sub">Parsed & Normalized</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Scored Candidates</span>
          <span class="stat-val" id="statScored">0</span>
          <span class="stat-sub">Evaluated vs. JD Criteria</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Shortlisted Candidates</span>
          <span class="stat-val" id="statShortlist">0</span>
          <span class="stat-sub">Score ≥ 60/100</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Top Match</span>
          <span class="stat-val" style="font-size: 22px; margin-top: 4px;" id="statTopCandidate">—</span>
          <span class="stat-sub" id="statAvgScore">Avg: 0 pts</span>
        </div>
      </div>

      <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
          <div>
            <h2 class="card-title">🎯 Prioritized Candidate Leaderboard</h2>
            <p class="card-subtitle">Candidates sorted by LangGraph RAG matching score with strengths & gaps breakdown.</p>
          </div>
          <div class="btn-group">
            <button class="btn btn-emerald" id="btnRunScoring" onclick="runEvaluation()">
              🚀 Run Evaluation & Ranking
            </button>
            <button class="btn btn-secondary" onclick="exportCsv()">
              📥 Export Shortlist CSV
            </button>
          </div>
        </div>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Candidate</th>
                <th>Match Score</th>
                <th>Recommendation</th>
                <th>Key Strengths</th>
                <th>Identified Gaps</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="rankedTableBody">
              <tr>
                <td colspan="7" style="text-align: center; color: var(--text-dim); padding: 36px;">
                  No candidates scored yet. Click "Run Evaluation & Ranking" or upload resumes in the Resumes tab.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 2: RESUMES & SCREENING -->
    <div id="tab-resumes" class="tab-pane">
      <div class="glass-card">
        <h2 class="card-title">📂 Bulk Resume Screening</h2>
        <p class="card-subtitle">Upload resumes in PDF, DOCX, or TXT format for automatic structured extraction.</p>

        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
          <div class="drop-zone-icon">📄</div>
          <h3 style="font-size: 16px; margin-bottom: 6px;">Drag and drop resumes here or click to browse</h3>
          <p style="font-size: 13px; color: var(--text-muted);">Supports multiple PDF and DOCX files simultaneously</p>
          <input type="file" id="fileInput" multiple accept=".pdf,.docx,.txt" style="display: none;" onchange="handleFilesSelected(this.files)">
        </div>

        <div style="margin-top: 18px;" class="btn-group">
          <button class="btn" onclick="loadSampleCandidates()">
            ⚡ Load 3 Built-in Sample Candidates (Alex, Priya, Marcus)
          </button>
          <span id="uploadStatusText" style="font-size: 13px; color: var(--accent-cyan);"></span>
        </div>

        <div style="margin-top: 28px;">
          <h3 style="font-size: 15px; margin-bottom: 12px; color: var(--text-muted);">Screened Candidates Pool (<span id="poolCount">0</span>)</h3>
          <div id="candidatesPoolList" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px;">
            <!-- Rendered cards -->
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: JOB DESCRIPTION (RAG) -->
    <div id="tab-jd" class="tab-pane">
      <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <div>
            <h2 class="card-title">📝 Active Job Description & RAG Knowledge Base</h2>
            <p class="card-subtitle">Chunked and embedded into Chroma vector storage for candidate grounding.</p>
          </div>
          <button class="btn btn-secondary" onclick="loadSampleJd()">
            ⚡ Pre-fill Sample Senior AI Engineer JD
          </button>
        </div>

        <div class="form-group">
          <label for="jdTitleInput">Target Job Title</label>
          <input type="text" id="jdTitleInput" placeholder="e.g. Senior Backend AI Engineer" value="Senior Backend AI Engineer" />
        </div>

        <div class="form-group">
          <label for="jdTextInput">Full Job Description & Rubric Criteria</label>
          <textarea id="jdTextInput" rows="12" placeholder="Paste full job description, required skills, and qualification standards here..."></textarea>
        </div>

        <button class="btn btn-emerald" onclick="saveJobDescription()">
          💾 Save & Ingest into RAG Vector Store
        </button>
        <span id="jdSaveStatus" style="margin-left: 12px; font-size: 13px; color: var(--accent-emerald);"></span>
      </div>
    </div>

    <!-- TAB 4: AGENT DIALOGUE -->
    <div id="tab-chat" class="tab-pane">
      <div class="glass-card">
        <h2 class="card-title">🤖 Conversational Recruitment Agent</h2>
        <p class="card-subtitle">Powered by LangGraph multi-node state machine with RAG retrieval and tool-calling.</p>

        <div class="chat-box" id="chatBox">
          <div class="chat-bubble chat-assistant">
            Hello! I am your AI HR Recruitment Assistant. I can screen resumes, match candidates against the active Job Description, rank them into a shortlist, or generate targeted interview guides. How can I help with your hiring pipeline?
          </div>
        </div>

        <div style="display: flex; gap: 10px;">
          <input type="text" id="chatInput" placeholder="e.g. 'Who is the best match for RAG pipelines?' or 'Rank all candidates'..." onkeydown="if(event.key==='Enter') sendChatMessage()" />
          <button class="btn" onclick="sendChatMessage()">Send</button>
        </div>
      </div>
    </div>
  </main>

  <!-- Candidate Detail & Interview Modal -->
  <div class="modal-overlay" id="detailModal">
    <div class="modal-card">
      <button class="modal-close" onclick="closeModal()">✕</button>
      <div id="modalContent">
        <!-- Injected via JS -->
      </div>
    </div>
  </div>

  <footer>
    <p>AI HR Recruitment Assistant • TNSDC Virtual Internship Program (IBM Agentic AI Track) • Powered by LangGraph, Chroma RAG & MCP</p>
  </footer>

  <script>
    // State cache
    let currentCandidates = {};
    let currentScored = {};
    let currentRanked = [];

    // Tab Switching
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
      
      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (targetBtn) targetBtn.classList.add('active');
      
      const targetPane = document.getElementById('tab-' + tabId);
      if (targetPane) targetPane.classList.add('active');
    }

    // Modal helpers
    function closeModal() {
      document.getElementById('detailModal').classList.remove('active');
    }

    // Refresh Overview stats and lists
    async function refreshState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        
        currentCandidates = data.parsed_candidates || {};
        currentScored = data.scored_candidates || {};
        currentRanked = data.ranked_candidates || [];

        // Update badges & stats
        document.getElementById('headerJdTitle').innerText = data.active_jd_title || 'No Active JD';
        document.getElementById('statScreened').innerText = Object.keys(currentCandidates).length;
        document.getElementById('statScored').innerText = Object.keys(currentScored).length;
        
        const shortlisted = currentRanked.filter(c => c.is_shortlisted || c.match_score >= 60);
        document.getElementById('statShortlist').innerText = shortlisted.length;

        if (currentRanked.length > 0) {
          document.getElementById('statTopCandidate').innerText = currentRanked[0].candidate_name;
          const avg = Math.round(currentRanked.reduce((acc, c) => acc + (c.match_score || 0), 0) / currentRanked.length);
          document.getElementById('statAvgScore').innerText = `Avg Score: ${avg} pts`;
        } else {
          document.getElementById('statTopCandidate').innerText = '—';
          document.getElementById('statAvgScore').innerText = 'Avg: 0 pts';
        }

        renderCandidatePool();
        renderRankedTable();
      } catch (err) {
        console.error("State refresh error:", err);
      }
    }

    // Render candidate pool cards
    function renderCandidatePool() {
      const container = document.getElementById('candidatesPoolList');
      const names = Object.keys(currentCandidates);
      document.getElementById('poolCount').innerText = names.length;

      if (names.length === 0) {
        container.innerHTML = '<p style="color: var(--text-dim); grid-column: 1/-1;">No resumes screened yet. Upload documents above or load built-in sample candidates.</p>';
        return;
      }

      container.innerHTML = names.map(name => {
        const c = currentCandidates[name];
        const skills = (c.skills || []).slice(0, 5).map(s => `<span class="tag-chip">${s}</span>`).join('');
        return `
          <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px;">
            <h4 style="font-size: 16px; font-weight: 700; color: #fff;">${c.name || name}</h4>
            <p style="font-size: 13px; color: var(--accent-cyan); margin: 2px 0 8px 0;">${c.title || 'Applicant'} • ${c.years_of_experience || 0} yrs exp</p>
            <div style="margin-bottom: 12px;">${skills}</div>
            <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px; width: 100%;" onclick="openCandidateModal('${name}')">
              🔍 View Parsed Profile
            </button>
          </div>
        `;
      }).join('');
    }

    // Render ranked leaderboard table
    function renderRankedTable() {
      const tbody = document.getElementById('rankedTableBody');
      if (!currentRanked || currentRanked.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="7" style="text-align: center; color: var(--text-dim); padding: 36px;">
              No candidates scored yet. Click "Run Evaluation & Ranking" or upload resumes in the Resumes tab.
            </td>
          </tr>
        `;
        return;
      }

      tbody.innerHTML = currentRanked.map((c, i) => {
        const score = c.match_score || 0;
        let badgeClass = 'score-low';
        if (score >= 80) badgeClass = 'score-high';
        else if (score >= 60) badgeClass = 'score-mid';

        const strengths = (c.strengths || []).slice(0, 2).map(s => `<span class="tag-chip tag-strength">${s}</span>`).join('<br>');
        const gaps = (c.gaps || []).slice(0, 2).map(g => `<span class="tag-chip tag-gap">${g}</span>`).join('<br>');

        return `
          <tr>
            <td style="font-family: var(--font-display); font-weight: 800; font-size: 16px; color: ${i === 0 ? 'var(--accent-amber)' : 'var(--text-muted)'};">
              #${c.rank || i + 1}
            </td>
            <td>
              <strong>${c.candidate_name}</strong>
              <div style="font-size: 12px; color: var(--text-dim);">${(c.raw_candidate && c.raw_candidate.title) || 'Candidate'}</div>
            </td>
            <td>
              <span class="score-badge ${badgeClass}">${score} / 100</span>
            </td>
            <td>
              <span style="font-size: 12px; font-weight: 600; color: ${c.recommendation && c.recommendation.includes('Advance') ? '#6ee7b7' : 'var(--text-muted)'};">
                ${c.recommendation || 'Evaluated'}
              </span>
            </td>
            <td>${strengths || '—'}</td>
            <td>${gaps || '—'}</td>
            <td>
              <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="openCandidateModal('${c.candidate_name}')">
                📋 Deep-Dive & Questions
              </button>
            </td>
          </tr>
        `;
      }).join('');
    }

    // Modal Details & Interview Generator
    async function openCandidateModal(name) {
      const parsed = currentCandidates[name] || {};
      const scored = currentScored[name] || {};
      
      const modal = document.getElementById('detailModal');
      const content = document.getElementById('modalContent');

      content.innerHTML = `
        <h2 style="font-size: 24px; margin-bottom: 4px;">Candidate Evaluation: <span style="color: var(--accent-cyan);">${name}</span></h2>
        <p style="color: var(--text-muted); font-size: 14px; margin-bottom: 20px;">${parsed.title || 'Applicant'} • ${parsed.years_of_experience || 0} Years Experience • ${parsed.email || 'Email on file'}</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
          <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px;">
            <h4 style="color: #6ee7b7; margin-bottom: 8px;">✅ Top Strengths & Match</h4>
            <ul style="padding-left: 18px; font-size: 13px; color: var(--text-muted); line-height: 1.6;">
              ${(scored.strengths || ['Demonstrated relevant experience']).map(s => `<li>${s}</li>`).join('')}
            </ul>
          </div>
          <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px;">
            <h4 style="color: #fda4af; margin-bottom: 8px;">⚠️ Identified Gaps & Shortfalls</h4>
            <ul style="padding-left: 18px; font-size: 13px; color: var(--text-muted); line-height: 1.6;">
              ${(scored.gaps || ['None identified']).map(g => `<li>${g}</li>`).join('')}
            </ul>
          </div>
        </div>

        <div style="margin-bottom: 24px;">
          <h4 style="margin-bottom: 6px;">Evaluation Reasoning:</h4>
          <p style="background: rgba(0,0,0,0.3); padding: 12px 16px; border-radius: 8px; font-size: 13px; color: var(--text-muted); border-left: 3px solid var(--accent-indigo);">
            ${scored.reasoning || 'Run evaluation to generate grounded reasoning against the active job description.'}
          </p>
        </div>

        <div style="border-top: 1px solid var(--border-color); padding-top: 20px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <h3 style="font-size: 18px;">🎯 Tailored Interview Guide</h3>
            <button class="btn" onclick="generateQuestionsFor('${name}')" id="btnGenQ">
              ⚡ Generate / Refresh Questions
            </button>
          </div>
          <div id="modalQuestionsContainer">
            <p style="color: var(--text-dim); font-size: 13px;">Click above to generate targeted role-fit and gap-probing interview questions with evaluation rubrics.</p>
          </div>
        </div>
      `;

      modal.classList.add('active');

      // Check if questions already exist
      try {
        const stateRes = await fetch('/api/state');
        const stateData = await stateRes.json();
        if (stateData.interview_questions && stateData.interview_questions[name]) {
          renderQuestionsList(stateData.interview_questions[name]);
        }
      } catch(e) {}
    }

    async function generateQuestionsFor(name) {
      const container = document.getElementById('modalQuestionsContainer');
      const btn = document.getElementById('btnGenQ');
      btn.innerHTML = '<span class="spinner"></span> Generating...';
      btn.disabled = true;

      try {
        const res = await fetch(`/api/candidates/${encodeURIComponent(name)}/interview-questions`, {
          method: 'POST'
        });
        const data = await res.json();
        renderQuestionsList(data);
      } catch (err) {
        container.innerHTML = `<p style="color: var(--accent-rose);">Error generating questions: ${err.message}</p>`;
      } finally {
        btn.innerHTML = '⚡ Generate / Refresh Questions';
        btn.disabled = false;
      }
    }

    function renderQuestionsList(qData) {
      const container = document.getElementById('modalQuestionsContainer');
      const questions = qData.questions || [];
      if (questions.length === 0) {
        container.innerHTML = '<p style="color: var(--text-dim);">No questions generated.</p>';
        return;
      }

      container.innerHTML = questions.map((q, idx) => `
        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px; margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 12px; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase;">Q${idx+1} • ${q.category || 'Competency'}</span>
          </div>
          <p style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 8px;">${q.question}</p>
          <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;"><strong>Objective:</strong> ${q.objective || 'N/A'}</p>
          <p style="font-size: 12px; color: #a7f3d0;"><strong>Interviewer Rubric:</strong> ${q.what_to_listen_for || 'N/A'}</p>
        </div>
      `).join('');
    }

    // Run Full Scoring & Ranking
    async function runEvaluation() {
      const btn = document.getElementById('btnRunScoring');
      btn.innerHTML = '<span class="spinner"></span> Scoring Candidates...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/evaluate/all', { method: 'POST' });
        const data = await res.json();
        await refreshState();
        switchTab('dashboard');
      } catch (err) {
        alert("Evaluation error: " + err.message);
      } finally {
        btn.innerHTML = '🚀 Run Evaluation & Ranking';
        btn.disabled = false;
      }
    }

    // Upload Resumes
    async function handleFilesSelected(files) {
      if (!files || files.length === 0) return;
      const status = document.getElementById('uploadStatusText');
      status.innerText = `Screening ${files.length} resume(s)...`;

      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      try {
        const res = await fetch('/api/resumes/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        status.innerText = `Screened ${data.processed || files.length} resume(s) successfully!`;
        await refreshState();
      } catch (err) {
        status.innerText = `Error: ${err.message}`;
      }
    }

    // Load Built-in Samples
    async function loadSampleCandidates() {
      const status = document.getElementById('uploadStatusText');
      status.innerText = "Loading Alex, Priya, and Marcus...";
      try {
        const res = await fetch('/api/resumes/load-samples', { method: 'POST' });
        const data = await res.json();
        status.innerText = "Sample candidates loaded!";
        await refreshState();
      } catch (err) {
        status.innerText = `Error: ${err.message}`;
      }
    }

    // Job Description actions
    async function saveJobDescription() {
      const title = document.getElementById('jdTitleInput').value.trim();
      const text = document.getElementById('jdTextInput').value.trim();
      const status = document.getElementById('jdSaveStatus');

      if (!text) {
        alert("Please provide the Job Description text.");
        return;
      }

      status.innerText = "Chunking and indexing into Chroma vector store...";
      try {
        const res = await fetch('/api/jd/upload', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ title, text })
        });
        const data = await res.json();
        status.innerText = `Saved! Generated ${data.chunks} chunks in RAG index.`;
        await refreshState();
      } catch (err) {
        status.innerText = `Error: ${err.message}`;
      }
    }

    async function loadSampleJd() {
      try {
        const res = await fetch('/api/jd/sample');
        const data = await res.json();
        document.getElementById('jdTitleInput').value = data.title;
        document.getElementById('jdTextInput').value = data.text;
      } catch (err) {
        console.error("Error loading sample JD:", err);
      }
    }

    // Chat
    async function sendChatMessage() {
      const input = document.getElementById('chatInput');
      const text = input.value.trim();
      if (!text) return;

      const chatBox = document.getElementById('chatBox');
      chatBox.innerHTML += `<div class="chat-bubble chat-user">${text}</div>`;
      input.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/api/agent/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        chatBox.innerHTML += `<div class="chat-bubble chat-assistant">${data.response || data.final_output}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
        await refreshState();
      } catch (err) {
        chatBox.innerHTML += `<div class="chat-bubble chat-assistant" style="color: var(--accent-rose);">Error: ${err.message}</div>`;
      }
    }

    // Export CSV
    function exportCsv() {
      window.location.href = '/api/export/csv';
    }

    // Reset Session
    async function resetRound() {
      if (!confirm("Are you sure you want to reset the current recruitment session?")) return;
      await fetch('/api/session/reset', { method: 'POST' });
      await refreshState();
    }

    // Init on load
    window.addEventListener('DOMContentLoaded', async () => {
      await loadSampleJd();
      await refreshState();
    });
  </script>
</body>
</html>
"""


# API Endpoints
@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
async def get_index():
    """Serves the main recruiter dashboard UI."""
    return HTML_DASHBOARD


@app.get("/api/state")
async def get_current_state():
    """Returns the current recruitment session state."""
    return {
        "active_jd_title": session_manager.active_jd_title,
        "active_jd_text": session_manager.active_jd_text,
        "parsed_candidates": session_manager.parsed_candidates,
        "scored_candidates": session_manager.scored_candidates,
        "ranked_candidates": session_manager.ranked_candidates,
        "interview_questions": session_manager.interview_questions
    }


@app.get("/api/jd/sample")
async def get_sample_jd():
    """Returns the default sample JD text."""
    sample_file = BASE_DIR / "sample_data" / "sample_jd.txt"
    if sample_file.exists():
        text = sample_file.read_text(encoding="utf-8")
    else:
        text = "Senior Backend AI Engineer with Python, LangGraph, RAG and FastAPI experience."
    return {
        "title": "Senior Backend AI Engineer",
        "text": text
    }


@app.post("/api/jd/upload")
async def upload_job_description(payload: JDUploadRequest):
    """Chunks and indexes the job description into vector storage."""
    title = payload.title.strip() or "Job Description"
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Job description text is empty.")

    chunks_count = rag_manager.ingest_jd_text(jd_text=text, title=title)
    session_manager.set_job_description(title=title, text=text)
    return {
        "status": "success",
        "title": title,
        "chunks": chunks_count,
        "message": f"Successfully ingested {chunks_count} chunks into RAG vector index."
    }


@app.post("/api/resumes/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Accepts multiple resume files (PDF/DOCX/TXT) and parses them into structured JSON."""
    processed = []
    for file in files:
        temp_file = TEMP_UPLOAD_DIR / file.filename
        try:
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Invoke LangGraph resume screening branch
            result = execute_agent_workflow(
                intent="screen_resume",
                target_resume_path=str(temp_file)
            )
            cand = result.get("current_candidate_parsed")
            if cand:
                session_manager.add_candidate(cand)
                processed.append(cand.get("name", file.filename))
        except Exception as e:
            logger.error(f"Error parsing uploaded file {file.filename}: {e}")
        finally:
            # Privacy cleanup: delete temporary uploaded file
            if temp_file.exists():
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    return {
        "status": "success",
        "processed": len(processed),
        "candidates": processed
    }


@app.post("/api/resumes/load-samples")
async def load_sample_candidates():
    """Loads and parses the 3 generated sample resumes (Alex, Priya, Marcus)."""
    samples_dir = BASE_DIR / "sample_data"
    sample_files = [
        samples_dir / "alex_morgan_resume.pdf",
        samples_dir / "priya_sharma_resume.docx",
        samples_dir / "marcus_vance_resume.pdf"
    ]

    loaded = []
    for sf in sample_files:
        if sf.exists():
            try:
                cand = parse_resume(sf)
                session_manager.add_candidate(cand)
                loaded.append(cand.get("name", sf.name))
            except Exception as e:
                logger.error(f"Error loading sample {sf.name}: {e}")

    return {
        "status": "success",
        "loaded": len(loaded),
        "candidates": loaded
    }


@app.post("/api/evaluate/all")
async def evaluate_all_candidates():
    """Scores all parsed candidates against the active JD and produces a ranked shortlist."""
    parsed_candidates = session_manager.parsed_candidates
    if not parsed_candidates:
        raise HTTPException(status_code=400, detail="No candidates parsed yet. Please upload or load resumes first.")

    jd_text = session_manager.active_jd_text or rag_manager.active_jd_full_text
    if not jd_text:
        sample_file = BASE_DIR / "sample_data" / "sample_jd.txt"
        if sample_file.exists():
            jd_text = sample_file.read_text(encoding="utf-8")
            rag_manager.ingest_jd_text(jd_text, "Senior Backend AI Engineer")
            session_manager.set_job_description("Senior Backend AI Engineer", jd_text)

    scored_list = []
    for name, cand in parsed_candidates.items():
        scored = score_candidate(candidate_json=cand, jd_text=jd_text)
        session_manager.add_scored_candidate(scored)
        scored_list.append(scored)

    # Rank candidates
    ranked_result = rank_candidates(scored_list)
    session_manager.update_ranked(ranked_result.get("ranked_list", []))

    return {
        "status": "success",
        "total_scored": len(scored_list),
        "ranked_shortlist": ranked_result.get("ranked_list", [])
    }


@app.post("/api/candidates/{candidate_name}/interview-questions")
async def get_interview_questions(candidate_name: str):
    """Generates tailored interview questions (role-fit & gap-probing) for a candidate."""
    result = execute_agent_workflow(
        intent="generate_questions",
        target_candidate_name=candidate_name
    )
    questions = result.get("interview_questions", {}).get(candidate_name)
    if not questions:
        # Direct tool call fallback
        scored = session_manager.scored_candidates.get(candidate_name, {})
        parsed = session_manager.parsed_candidates.get(candidate_name, {})
        questions = generate_interview_questions(
            candidate_name=candidate_name,
            role=session_manager.active_jd_title,
            candidate_skills=parsed.get("skills", []),
            skill_gaps=scored.get("gaps", []),
            jd_text=session_manager.active_jd_text
        )
        session_manager.add_questions(candidate_name, questions)

    return questions


@app.post("/api/agent/chat")
async def agent_chat(payload: ChatMessageRequest):
    """Interacts with the LangGraph agent state machine."""
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty chat message.")

    result = execute_agent_workflow(user_input=user_msg)
    return {
        "response": result.get("final_output", "I processed your request."),
        "intent": result.get("intent", "general_chat")
    }


@app.get("/api/export/csv")
async def export_csv():
    """Exports the ranked candidate shortlist as a CSV file."""
    ranked = session_manager.ranked_candidates
    if not ranked:
        # Check if scored candidates exist to rank
        scored = list(session_manager.scored_candidates.values())
        if scored:
            ranked_res = rank_candidates(scored)
            ranked = ranked_res.get("ranked_list", [])
            session_manager.update_ranked(ranked)

    csv_content = generate_shortlist_csv(ranked)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=candidate_shortlist.csv"}
    )


@app.post("/api/session/reset")
async def reset_session():
    """Resets the active hiring session state."""
    session_manager.reset_round()
    return {"status": "success", "message": "Session reset successfully."}


if __name__ == "__main__":
    logger.info(f"Starting AI HR Recruitment Assistant on http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
