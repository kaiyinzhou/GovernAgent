# An Efficient and Reliable Agent-Based System for Clinical Data Governance

This repository contains the inference and evaluation scripts used for clinical note governance experiments.

## 0. Environment Requirements

### Runtime

- Python `>=3.10`
- Linux recommended for vLLM serving
- NVIDIA GPU + CUDA (required for vLLM inference)

### Python Packages

Install minimal dependencies:

```bash
pip install vllm openai transformers tqdm jieba
```

If you only run local evaluation scripts (`2-1-*`, `2-2-*`) without model inference, standard library is mostly sufficient.

## 1. Predefined Interface Schemas

| Note Type | Required Sections |
| :--- | :--- |
| Admission Record | Admission Time, Chief Complaint, History of Present Illness, History of Past Illness, Specialist Examination, Social History, Menstrual History, History of Family Member Diseases, Physical Examination, Auxiliary Examination, Preliminary Diagnosis, Admission Diagnosis, Treatment Plan |
| Initial Progress Note | Diagnostic Criteria, Preliminary Diagnosis, Differential Diagnosis, Treatment Plan, **Case Characteristics** |
| Daily Progress Note | Daily Situations, Daily Orders, Treatment Plan |
| Consultation Record | Admission Diagnosis, Current Diagnosis, Hospital Course, Auxiliary Examination, Reason for Consultation, Treatment Plan, Consultation Opinions, **Consultation Time** |
| Surgical Record | Surgical Time, Preoperative Diagnosis, Postoperative Diagnosis, Surgical Procedure Name, Anesthesia Method, Intraoperative Medications, Intraoperative Course, **Surgical Grade** |
| Postoperative Note | Surgical Procedure Name, Postoperative Diagnosis, Surgical Time, Intraoperative Course, Postoperative Precautions, **Preoperative Diagnosis** |
| Discharge Record | Admission Time, Discharge Time, Admission Diagnosis, Discharge Diagnosis, Admission Condition, Treatment Course, Discharge Orders, **Discharge Condition** |
| Death Record | Admission Time, Death Time, Admission Diagnosis, Treatment Course, Cause of Death, Death Diagnosis, **Admission Condition** |

## 2. Repository Layout (Top-Level)

All main scripts are placed at repository root for easy reproduction:

- Inference (agent): `1-2-new-agent.py`, `1-2-new-agent-en.py`
- Inference (flat): `1-1-new-flat.py`, `1-1-new-flat-en.py`
- Evaluation: `2-1-f1-soft.py`, `2-1-f1-soft-en.py`, `2-2-f1-hard.py`, `2-2-f1-hard-en.py`
- Configs and prompts: `configs/`, `prompt_temp.py`
- Data examples: `datas/`

## 3. Quick Start

### Step 1: Start Model Service

Start your vLLM/OpenAI-compatible endpoint first, then set model name and URL in the target script.

### Step 2: Run Inference

From repository root:

```bash
python 1-2-new-agent.py
```

or

```bash
python 1-1-new-flat.py
```

### Step 3: Run Evaluation

```bash
python 2-1-f1-soft.py
python 2-2-f1-hard.py
```

## 4. Notes for Reproducibility

- Keep script files at repository root as provided.
- Keep prompt/config files unchanged to ensure consistent experimental behavior.
- Inference and evaluation scripts assume JSON/JSONL input formats used in this repository.

