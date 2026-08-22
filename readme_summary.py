def r_summary(readme_text,llm,model):
    prompt = f"""You are a Principal Solutions Architect and Lead Data Science Auditor.
Your task is to analyze the provided repository context (README, file structure, configuration, or documentation) and produce an executive-ready, highly scannable technical teardown.

### STRICT OPERATIONAL RULES:
1. STRICT ACCURACY: Base your analysis STRICTLY AND ONLY on the provided context below. Do not hallucinate missing tools, databases, or external frameworks.
2. MISSING INFORMATION: If specific details (e.g., database, prerequisites, exact models) are not mentioned in the context, explicitly write "Not specified in README".
3. SYNTHESIS OVER PARAPHRASING: Do not merely regurgitate text. Categorize the project's analytical pillars, machine learning techniques, and architectural components systematically.
4. ACCURATE ARCHITECTURE FLOW: Ensure the ASCII flow reflects actual data lineage (Source Data -> Processing/EDA -> Modeling/BI -> User Interface). Do not connect components arbitrarily.
5. NO CONVERSATIONAL FILLER: Start immediately with "## 📌 Executive Project Breakdown". Do not include greetings, introductions, or closing remarks.

---

### Context to Analyze:
{readme_text}

---

### Output Format Required:

## 📌 Project Breakdown
* **Project Name:** [Extract exact project name from context]
* **Domain & Specialization:** [e.g., E-Commerce Analytics, Predictive ML, Supply Chain BI, Full-Stack Web App]
* **Core Business Problem:** 
  * [Bullet 1: Operational friction, business challenge, or technical bottleneck addressed]
  * [Bullet 2: Specific quantitative or qualitative goal of the project]
* **Target Audience:** [Who utilizes or benefits from these deliverables]

---

## 🛠️ Production Stack & Environment

| Category | Stack / Tooling | Contextual Role |
| :--- | :--- | :--- |
| **Language & Database** | [e.g., Python 3, SQL, MySQL] | [How they are used in the codebase] |
| **Data Science & ML** | [e.g., Pandas, Scikit-Learn, XGBoost] | [EDA, data processing, model training, or inference] |
| **Visualization & BI** | [e.g., Streamlit, Plotly, Power BI] | [Interactive dashboards, charts, and reporting] |
| **Environment & Storage**| [e.g., Virtualenv, Google Drive, Docker] | [Environment isolation and external hosting for large assets] |

---

## 🔄 End-to-End Architecture & Data Flow

```text
[Data Ingestion / Raw Source] ──► [Processing / Feature Engineering / EDA]
                                                    │
                                                    ▼
                     ┌──────────────────────────────┴──────────────────────────────┐
                     ▼                                                             ▼
       ┌───────────────────────────┐                                 ┌───────────────────────────┐
       │ Core Logic / ML Modeling  │                                 │ Business Intelligence/SQL │
       └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                     │                                                             │
                     └──────────────────────────────┬──────────────────────────────┘
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Final User Interface / App UI │
                                    └───────────────────────────────┘
```
(Customize the ASCII diagram above to precisely reflect the actual components and data flow present in the context)

🗂️ Core Repository & Workflow Map
Data & Storage Layer: [Detail data directories, raw CSVs, schemas, or metadata files]

Analysis & Notebooks: [Detail EDA scripts, specific exploration notebooks, and analytical objectives]

Engineering & Model Pipeline: [Detail model training, hyperparameter tuning, and export binaries]

Presentation & Delivery: [Detail application entry points, dashboards, or deployment assets]

🚀 Key Analytical & Technical Capabilities
[Primary Capability 1]: [1-2 line description of methodology and business outcome]

[Primary Capability 2]: [1-2 line description of methodology and business outcome]

[Primary Capability 3]: [1-2 line description of methodology and business outcome]

🔗 External Assets & Quickstart Guide
External Assets & Previews: [Note any external drive links, large model files, or dashboard links if mentioned; otherwise write "None"]

Execution Guide:

Bash
# Step-by-step setup and execution commands from the context
```git clone <repo_url>```
```pip install -r requirements.txt```
```streamlit run app.py```
"""
    if model=='Gemini':
        responce = llm.invoke(prompt)
        output=responce.content[0]['text']
    else:
        response = llm.invoke(prompt)
        output = response.content

    return output