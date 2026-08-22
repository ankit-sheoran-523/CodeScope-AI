# CodeScope AI
### Intelligent GitHub Onboarding & Multi-File Architecture RAG Assistant

An end-to-end developer copilot engineered to eliminate codebase onboarding friction. **CodeScope AI** parses remote GitHub repositories, models cross-module execution flows, and delivers hallucination-audited answers using dual-layer RAG.

</div>

---

## The Problem & The Solution

Navigating and understanding an unfamiliar GitHub repository is often slow and fragmented:
* **Documentation Drift:** READMEs are frequently outdated, vague, or missing critical setup instructions.
* **Isolated File Inspection:** Standard code assistants analyze single files in isolation, missing cross-file imports, state transitions, and dependency pipelines.
* **Hallucinations:** LLMs tend to invent parameters, helper functions, and library methods when querying complex codebases.

**CodeScope AI** solves this by combining **Macro Documentation Synthesis** with **Micro Multi-File Graph Retrieval**. It constructs real-time inter-file dependency matrices and runs automated **LLM-as-a-Judge telemetry** on every query to guarantee grounded, verified answers.

---

## System Architecture & Workflow

```text
               ┌───────────────────────────────┐
               │    Remote GitHub Repository   │
               └──────────────┬────────────────┘
                              │ (PyGithub API)
              ┌───────────────┴───────────────┐
              ▼                               ▼
      [ README Extraction ]       [ Multi-File Selection ]
              │                               │
              ▼                               ▼
    [ `readme_summary.py` ]         [ `file_summary.py` ]
    (Executive Tech Summary)       (Cross-File Dependency Graph)
              │                               │
              └───────────────┬───────────────┘
                              ▼
        ┌───────────────────────────────────────────┐
        │        Dual-Layer FAISS Vector Store      │
        │  • Macro Layer: System Architecture Docs  │
        │  • Micro Layer: Chunked Raw Source Code   │
        └─────────────────────┬─────────────────────┘
                              │
  User Query ──► [  Security Guardrail ] ──► (Permit / Block)
                              │ (Permitted)
                              ▼
            [ Semantic Top-K Vector Retrieval ]
                              │
                              ▼
            [ Multi-Provider LLM Answer Engine ]
                   (Gemini / Groq )
                              │
                              ▼
         [  Real-Time LLM-as-a-Judge Telemetry ]
         • Faithfulness  • Groundedness  • Hallucination Rate
```

---

## Key Features
### 1. Repository-Wide Macro Analysis
- Automatic extraction and synthesis of remote README.md files into executive architecture briefs.
- Rapid contextual Q&A for repository prerequisites, environment variables, and execution steps.
### 2. Multi-File Cross-Codebase Synthesis
- Collective Ingestion: Select multiple interdependent scripts (main.py, model.py, utils.py) simultaneously.
- Dependency Matrix: Generates automated ASCII data flow pipelines and inter-file dependency tables.
- Cross-File Q&A: Traces function calls, argument passing, and variable scopes across file boundaries.
### 3. Upfront Technical Security Guardrail
- Rejects prompt injections, jailbreaks, and backend credential extraction attempts (os.environ["API_KEY"]).
- Permissive classification ensures valid debugging, training, testing, and algorithmic questions are never blocked.
### 4. LLM-as-a-Judge Telemetry Dashboard
- Every retrieved answer includes dynamic telemetry to verify response authenticity:<br>
        - Faithfulness (0–100%): Validates if generated claims are directly verifiable against retrieved code chunks.<br>
        - Groundedness (0–100%): Measures answer focus and prevents topical drift.<br>
        - Hallucination Rate (0–100%): Flags ungrounded assumptions ($0\% = \text{Strictly Factual}$).<br>
        - Latency & Token Usage: Live tracking of inference latency, input/output tokens, and remaining context window.<br>

---
##  Tech Stack & Engineering Toolkit

| Subsystem | Tool / Framework | Purpose |
| --- | --- | --- |
| Application Layer | Streamlit | Interactive UI |
| LLM Orchestration | LangChain | Prompt chaining & schema modeling |
| Inference Engines | Gemini / Groq | High‑density synthesis & low‑latency reasoning |
| Embeddings | Gemini Embedding | Dense vectorization |
| Vector Index | FAISS | Similarity search |
| Repo Ingestion | PyGithub | API traversal & blob decoding |

---
## Repository Directory Layout
```text
CodeScope-AI/
├── app.py                  # Main Streamlit application entrypoint & UI orchestrator
├── readme_summary.py       # Macro documentation extraction & summarization engine
├── file_summary.py         # Multi-file architectural synthesis & pipeline logic
├── git_raw_file.ipynb      # Prototyping scratchpad for RAG chunking & evaluations
├── requirements.txt        # Production Python dependencies
├── .env.example            # Environment variable configuration template
└── README.md               # Project documentation
```
---

## Quickstart & Setup
### Prerequisites
- Python 3.10 or higher
- Google Gemini API Key and/or Groq API Key

### 1. Clone the Repository
### 2. Create & Activate a Virtual Environment
``` 
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Run the Application
```
streamlit run app.py
```
---
### Usage Guide
- Configure Repository: Open the sidebar, enter any public GitHub username, and select a repository from the dynamic dropdown.
- Select LLM Provider: Choose between Google Gemini or Groq, and provide the respective API key.
- Tab 1 — README Analysis: Click Generate Repository Summary & Index to review high-level system prerequisites and query project setup details.
- Tab 2 — Multi-File Analysis: Select multiple related files from the multiselect widget (e.g., train.py, dataset.py, config.py), click Generate Unified Architecture, inspect the generated data flow diagram, and ask cross-file execution questions.
- Inspect Metrics: Expand the View Detailed Evaluation and View Context Chunks drawers below any assistant message to inspect retrieved code chunks and judge scores.