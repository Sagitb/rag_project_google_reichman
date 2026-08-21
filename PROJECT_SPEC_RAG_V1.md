# Jira RAG Assistant — Project Specification & Team Guide

**Course:** Applied Language Models — Reichman University / Google (Dr. Barak Or)
**Timeline:** 2 weeks | **Team size:** 4
**Stack:** FastAPI + React · ChromaDB · Gemma 4 E4B (Apache 2.0) · QLoRA fine-tuning
**Execution model:** Cloud-first — GitHub is the source of truth, Colab is the compute, Hugging Face Hub is the artifact store. No dependency on any personal machine.

---

## 1. Project Overview

A full-stack RAG application that answers questions over Jira support tickets, with:

- **Live RAG:** one indexing pipeline fed by two sources — a bootstrap dataset (5 anonymized seed tickets + ~1,000 LLM-augmented synthetic tickets) and a live Jira Cloud sync (our dedicated free Jira site, not any production workspace).
- **Fine-tuned open model:** Gemma 4 E4B adapted with QLoRA on the augmented dataset; baseline vs. fine-tuned comparison is a core deliverable.
- **Two roles:** a chat user (RAG chatbot with source citations) and an admin (dashboard with logs, usage statistics, sync status).

**Grading alignment (100 pts):** Code & implementation 30 · Experiments & results 25 · Analysis 20 · Presentation 15 · Documentation 10.

---

## 2. Claude Code Specification

Paste everything in this section as the opening prompt to Claude Code. Ask it to propose the file structure and API contract for approval before building.

### Monorepo structure

- `backend/` — Python FastAPI service (RAG pipeline, auth, chat, logging, Jira sync)
- `frontend/` — React + Vite + Tailwind SPA (login, chat UI, admin dashboard)
- `ml/` — Colab-native notebooks and scripts (augmentation, QLoRA fine-tuning, evaluation). Nothing in the app depends on `ml/` at runtime; the app loads the LoRA adapter from the Hugging Face Hub.
- `data/` — versioned seed data: `seed_tickets.csv` (5 anonymized tickets), `synthetic_tickets.jsonl` (generated), augmentation prompts.
- Root: `README.md`, `docker-compose.yml` (backend + frontend + chroma), `.env.example`, `Makefile`.

### 2.1 Auth

- Username/password login, bcrypt-hashed passwords, JWT session tokens.
- SQLite via SQLAlchemy. Seed two users on first run:
  - `demo_user` (role: user) → chat interface only
  - `admin` (role: admin) → admin dashboard only (redirect by role after login)
- Seeded passwords from `.env` with safe local-dev defaults. No registration, no password reset. All API routes protected by role.

### 2.2 RAG pipeline

- **Vector store:** ChromaDB, persistent local directory, runs in docker-compose.
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (local, no API key).
- **Ingestion — two sources, one pipeline:**
  1. Bootstrap: load `data/seed_tickets.csv` + `data/synthetic_tickets.jsonl` on first run.
  2. Live Jira sync: background task polling Jira Cloud REST API (base URL, email, API token from `.env`) every N minutes; JQL `updated >= -1h`; normalize to the ticket schema; upsert into Chroma with stable IDs (= ticket key) so updates overwrite.
- **Chunking:** one chunk per logical section (summary+background, steps_to_reproduce, resolution, qa_criteria), each carrying `ticket_id` + `section` metadata.
- **Retrieval:** top-k (default 5), metadata returned. Grounded prompt = system instructions + retrieved chunks (with ticket IDs) + user question. The model must cite ticket IDs and answer "I don't have information on this" when retrieval score falls below a threshold.

### 2.3 Model serving — Gemma 4 E4B

Provider abstraction, selected via `.env`:

1. `mock` (default for app development): echoes retrieved context, clearly labeled. Lets any teammate build and test the full app with zero GPU and zero cost.
2. `hf_endpoint` (default for real answers + demo): remote HTTP endpoint — a Hugging Face Space running Gemma 4 E4B (quantized) that can load our LoRA adapter from the HF Hub by repo name.
3. `ollama` (optional): any machine with local Ollama at `localhost:11434` and `gemma4:e4b` pulled — backup demo path.

- Adapter identified by **HF Hub repo name** in `.env` (not a local path); README documents switching base vs. tuned.
- Streaming responses to frontend (SSE).

### 2.4 Chat interface (role: user)

- Message list with streaming responses.
- Collapsible "Sources" panel per answer: ticket ID, section, similarity score.
- Thumbs up/down feedback per answer (stored with log).
- Conversation history persisted per user; last N turns sent as context.

### 2.5 Admin dashboard (role: admin)

- Log table: timestamp, user, question, answer, retrieved ticket IDs, similarity scores, latency ms, feedback. Filters by date and feedback.
- Stats cards + charts (recharts): queries/day, avg latency, feedback ratio, most-retrieved tickets, "no answer" rate.
- Jira sync panel: last sync time, tickets indexed, index size, manual "Sync now" button.

### 2.6 Logging

- Every chat interaction → structured row in SQLite (all dashboard fields) **and** JSON line in `logs/chat.jsonl` for the `ml/` evaluation scripts.

### 2.7 ml/ scaffolds (created now, filled in during week 1–2)

- `01_augment_dataset.ipynb` — synthetic tickets from 5 seeds.
- `02_finetune_qlora.ipynb` — QLoRA fine-tune of Gemma 4 E4B (transformers + peft + bitsandbytes), checkpointing to Drive, final adapter pushed to HF Hub.
- `03_eval_retrieval.ipynb` — Hit@k, MRR over a question→ticket gold set.
- `04_eval_generation.ipynb` — baseline vs. fine-tuned (Rouge-L, qualitative examples).

### Non-goals — do NOT build

- No registration / password reset / email / OAuth.
- No multi-tenant, no i18n, no mobile layouts.
- No writing back to Jira — read-only integration.

### Quality bar

- README with mermaid architecture diagram, fresh-clone setup steps (`docker-compose up` + one seed command), `.env.example` documenting every variable, and "Open in Colab" badges for all `ml/` notebooks.
- Python type hints; pydantic models for all API schemas.
- `make demo`: boots everything with seed data and the `mock` provider — a grader runs the full system in under 5 minutes with no GPU.

### Build order

backend auth + DB → RAG pipeline with bootstrap data → chat UI → model provider layer → Jira sync → admin dashboard → ml scaffolds.

---

## 3. Training Workflow: Colab + GitHub (cloud-first)

**Principle:** GitHub is the single source of truth for all code and notebooks. Colab is where everything executes. The Hugging Face Hub stores every artifact that must outlive a Colab session (adapters, processed datasets). No step depends on anyone's personal hardware.

### 3.1 Accounts & one-time setup (day 1)

1. **Colab Pro** for the training owner (~$10/month, one month). Everyone else uses the free tier — evaluation and augmentation are CPU-bound or light.
2. **Hugging Face org or shared account** with two private repos:
   - `<team>/jira-rag-adapter` (model repo) — LoRA adapters, one revision per experiment.
   - `<team>/jira-rag-data` (dataset repo) — processed fine-tuning pairs and the gold eval set.
3. **GitHub repo** (private) — authorize Colab once (File → Open notebook → GitHub tab).
4. Each member stores three secrets in **Colab Secrets** (key icon, left sidebar — never in cells): `HF_TOKEN`, `GITHUB_PAT` (fine-grained, this repo only), `JIRA_TOKEN` (only who needs it).

### 3.2 Notebook conventions (Colab-native)

Every notebook in `ml/` must:

- Start with a setup cell: `!pip install -r requirements.txt` (raw GitHub URL), read secrets via `google.colab.userdata`, optional Drive mount for checkpoints.
- Route all paths through a single `DATA_DIR` variable.
- Pull base models from the HF Hub by name; pull/push our artifacts to the HF repos by name — **never rely on the Colab VM's disk between sessions.**
- Save training checkpoints to Drive every N steps (survives disconnects); push the **final** adapter to the HF model repo with a tagged revision (e.g., `exp-03-rank16-lr2e4`).

### 3.3 The working loop

1. Open a notebook **from GitHub** in Colab (badge link in README or File → Open → GitHub).
2. Run / edit. Training runs are sized to fit comfortably in a session: E4B QLoRA at a few hundred steps to 1–3 epochs is roughly 1–3 hours on an A100/L4 — no overnight runs needed. If a session dies, resume from the last Drive checkpoint.
3. **Save a copy in GitHub** (File menu) with a meaningful commit message — this is how notebook work gets "pushed."
4. Log every experiment in `ml/EXPERIMENTS.md`: config, HF revision tag, metrics. This table goes almost verbatim into the report.

### 3.4 GitHub ↔ Colab reference

- **Open from GitHub:** `https://colab.research.google.com/github/<org>/<repo>/blob/main/ml/02_finetune_qlora.ipynb` — works for private repos after one-time authorization.
- **README badges:**
  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ORG/REPO/blob/main/ml/02_finetune_qlora.ipynb)
  ```
- **Save back:** File → Save a copy in GitHub → repo + branch + commit message. Avoid Drive copies of notebooks — the repo stays canonical.
- **Cloning inside a session** (for scripts/data in the repo): `git clone` with `GITHUB_PAT` from Colab Secrets.

### 3.5 Serving & demo plan

- **Primary:** Hugging Face Space (Docker or vLLM template) serving Gemma 4 E4B quantized + our adapter from the Hub. The app's `hf_endpoint` provider points at it. Free CPU tier is too slow for live inference — budget a small GPU Space (~$0.60–1/hr, turn on only for testing sessions and the demo day).
- **Backup for demo day:** `ollama` provider on any available machine (including the RTX 5090 laptop) — pre-tested, switched by one `.env` line.
- **Grader path:** `make demo` with the `mock` provider — full system, no GPU, no cost.

### 3.6 Session-limit discipline (the cost of cloud-first)

- Colab Pro sessions can still disconnect; **checkpoint to Drive is mandatory, not optional.**
- Keep the fine-tuning task sized to the course requirement (several hundred steps / 1–3 epochs) — do not design experiments that need >4h of continuous GPU.
- Do quick smoke tests (10 steps, tiny subset) before launching a full run — session time is the scarce resource now.

---

## 4. Data Workflow

1. **Seeds:** 5 anonymized closed tickets exported from Jira → `data/seed_tickets.csv`. Schema: `ticket_id, issue_type, priority, status, summary, description_raw, background, steps_to_reproduce, expected_behavior, actual_behavior, resolve_description_raw, resolution_summary, qa_criteria, reporter, assignee, created_date, resolved_date`. IDs renumbered TCKT-001…005; people/companies replaced with consistent placeholders (USER_1, CLIENT_A).
2. **Augmentation:** LLM-generated synthetic tickets (~1,000) from the seeds → `data/synthetic_tickets.jsonl`. The generation prompt and script are **committed** to the repo — graders must see the full lineage: 5 seeds → prompt → dataset.
3. **Fine-tuning data:** derived (input → output) pairs from the synthetic tickets (task to be finalized: raw description → structured breakdown, and/or grounded Q&A pairs). Pushed to the HF dataset repo.
4. **Gold eval set:** ~50 hand-written questions mapped to the ticket(s) that answer them, for Hit@k / MRR. Written by a team member who did **not** write the augmentation prompt (avoids phrasing leakage that inflates retrieval metrics).
5. **Team Jira site:** fresh free Jira Cloud site (10-user free tier) with the anonymized data imported via CSV. The live sync points **only** at this site. Scoped API token in `.env` / Colab Secrets (never committed).

---

## 5. Team Roles & Timeline

| Role | Owns | Compute |
|---|---|---|
| Training owner | `02_finetune` notebook, HF adapter repo, experiment log | Colab Pro |
| Data & augmentation | Augmentation prompt/script, dataset QA, HF dataset repo | Colab free |
| Evaluation | `03`/`04` notebooks, metrics, plots for report | Colab free |
| App & serving | Claude Code app, HF Space setup, admin dashboard QA, report & slides lead | None / Space |

**Week 1:** repo + Colab/HF/GitHub wiring (day 1) · seed CSV finalized · team Jira site live · augmentation done · app scaffolded via Claude Code · first baseline RAG answers via mock + hf_endpoint.
**Week 2:** fine-tuning experiments (logged in EXPERIMENTS.md) · retrieval + generation eval · HF Space with tuned adapter live · report + slides · live-demo rehearsal (create a Jira ticket → ask the bot about it 30 seconds later) · pre-test the ollama backup path.

**Presentation reminder:** every team member must speak (rubric requirement). Suggested split: motivation & architecture / data & augmentation / training & experiments / results, demo & limitations.

---

## 6. Course Requirements & Coverage Map

### 6.1 How RAG and LoRA relate in this project

They are **two independent axes**, not alternatives — and the assignment requires both (Workflow stage 2 mandates LoRA/QLoRA; stage 3 mandates retrieval evaluation *and* baseline-vs-fine-tuned comparison).

- **RAG changes what the model sees** — retrieved ticket chunks injected at inference time. Evaluated with Hit@k / MRR.
- **LoRA changes how the model behaves** with what it sees — grounded, ticket-citing answers in a consistent format. Evaluated with Rouge-L + qualitative examples.

In our architecture they are already decoupled: retrieval runs in the backend; the adapter is loaded by HF Hub repo name in the serving layer. Either can be toggled independently, which gives us the experiment grid below for free.

### 6.2 The 2×2 experiment grid (core results slide)

| | Base Gemma 4 E4B | LoRA-tuned |
|---|---|---|
| **No retrieval** | Cell 1 — hallucination baseline | Cell 2 — memorization check |
| **With retrieval** | Cell 3 — standard RAG | Cell 4 — full system |

Reading the grid for the Analysis section:

- **1 → 3:** isolates the contribution of retrieval.
- **3 → 4:** isolates the contribution of fine-tuning.
- **Cell 2 (the interesting one):** did the model memorize ticket *facts* (undesirable — memorization masquerading as capability, and a hallucination source) or learn answer *format and grounding behavior* (desirable)? Failure cases here feed the required hallucination/limitations discussion directly.

### 6.3 Requirements coverage

| Requirement (from the brief) | Where it's satisfied |
|---|---|
| Dataset described (size, classes, balance) | Report §2 + `data/README`; 5 seeds → ~1,000 synthetic tickets, distribution by issue type/priority |
| Train / validation / test split | Augmentation script writes explicit splits (not one blob) — must be a script output, described in report |
| Preprocessing: cleaning, tokenization, chunking | Backend ingestion pipeline: section-level chunking (summary+background / steps / resolution / qa_criteria) |
| Embedding generation + vector store | sentence-transformers `all-MiniLM-L6-v2` → ChromaDB (persistent) |
| Pretrained LLM chosen | Gemma 4 E4B (Apache 2.0, 128K context) — licensing rationale noted in report |
| LoRA / QLoRA fine-tuning | `ml/02_finetune_qlora.ipynb` — QLoRA via transformers + peft + bitsandbytes on Colab |
| Several hundred steps / 1–3 epochs | Training config sized to fit one Colab session (see §3.6) |
| **≥1 hyperparameter experiment** | LoRA rank sweep (r = 8 / 16 / 32); optionally learning rate. All runs logged in `ml/EXPERIMENTS.md` with HF revision tags |
| Optional: agentic workflow | **Out of scope** — deliberately skipped to protect the 2-week timeline. Noted in report as future work |
| Training loss + validation curves | Plots from `02` notebook, exported to `ml/plots/` and embedded in README + report |
| Standard NLP metrics | Rouge-L for generation (`04`), plus accuracy/F1 if a classification sub-task is included |
| **Retrieval quality (Hit@k, MRR)** | `ml/03_eval_retrieval.ipynb` against the ~50-question gold set |
| Baseline vs. fine-tuned comparison | The 2×2 grid above (§6.2) |
| Qualitative examples (prompt vs. response) | `04` notebook + a slide; also visible live in the app's Sources panel |
| Overfitting / underfitting / hallucination analysis | Report §5, anchored on Cell 2 findings and low-similarity "I don't have information" cases |
| Limitations + suggested improvements | Report §5–6: synthetic-data domain gap, single-embedding-model retrieval, no reranker, no agentic layer, small seed set |
| ≥1,000 examples (own-dataset track) | ~1,000 augmented synthetic tickets in `data/synthetic_tickets.jsonl` |
| Clean, commented, reproducible code | Claude Code spec §2 quality bar; type hints, pydantic schemas, docker-compose |
| README (summary, team, dataset, methods, results, setup) | Root `README.md` — structure mirrors this list exactly |
| requirements.txt + Colab notebook | `ml/requirements.txt` (pinned) + "Open in Colab" badges (§3.4) |
| Short report, 2–3 pages PDF | Written week 2 by the app/report owner; sections map to the rows above |
| Presentation 7–10 min, all members speak | Split: motivation & architecture / data & augmentation / training & experiments / results, demo & limitations |

### 6.4 Deliverables checklist (final week)

- [ ] GitHub repo public or shared with instructor; README complete with plots and qualitative examples
- [ ] `ml/requirements.txt` pinned; all notebooks verified to run top-to-bottom on Colab
- [ ] `ml/EXPERIMENTS.md` filled in (config → metrics → HF revision)
- [ ] Report PDF, 2–3 pages
- [ ] Slides with: architecture diagram, retrieval pipeline diagram, loss curves, 2×2 results table, live demo
- [ ] Live demo rehearsed end-to-end, plus the `ollama` backup path pre-tested
- [ ] Every team member has a speaking part
