def file_summary(file_data_map, llm, model):
    """Passes all selected files to the LLM to create a single unified architecture, pipeline, and interaction summary."""
    combined_blocks = []
    for f_path, code in file_data_map.items():
        combined_blocks.append(f"### FILE: {f_path}\n```\n{code[:6000]}\n```")

    files_context = "\n\n---\n\n".join(combined_blocks)

    prompt = f"""You are a Principal Software Architect & Code Auditor.
Analyze the following selected codebase files collectively and produce a unified System Architecture, Data Flow, and Implementation Summary.

Files & Code:
{files_context}

---
### Strict Rules:
1. Explain how these specific files interact, import, and depend on each other.
2. Focus on cross-file communication (e.g., File A imports Class X from File B to process Data Y).
3. Do not use generic conversational filler. Start directly with the structure below.

---
### Desired Structure:
# File's Architecture & Execution Flow

### 1. 🎯 Subsystem Overview & Core Purpose
* [Concise synthesis of what this cluster of files accomplishes together]

### 2. 🔗 Inter-File Dependency & Interaction Matrix
| Source File | Interacting / Dependent File | Imported Object / Function | Purpose & Data Exchanged |
| :--- | :--- | :--- | :--- |

### 3. 🔄 End-to-End Execution & Data Pipeline Flow
[File A: Entry/Input] ──(Data/Params)──> [File B: Core Logic/Model] ──(Output/State)──> [File C: Evaluation/Save]

* **Step 1 (Ingestion / Initialization):** [What happens first and where]
* **Step 2 (Transformation / Core Processing):** [How data flows between the files]
* **Step 3 (Execution / Inference / Output):** [Terminal output, artifact generation, or return value]

### 4. 🧩 Consolidated Key Components & Interfaces
* **`filename.py`:** [Key responsibility, major exported classes/functions]
"""
    if model=='Gemini':
        responce = llm.invoke(prompt)
        output=responce.content[0]['text']
    else:
        response = llm.invoke(prompt)
        output = response.content

    return output