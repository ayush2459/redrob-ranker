# Redrob AI Hackathon — Intelligent Candidate Ranker

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![Runtime](https://img.shields.io/badge/Runtime-~30s%20CPU-00A896?style=flat)
![Candidates](https://img.shields.io/badge/Candidates-100K-0D1B2A?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![No GPU](https://img.shields.io/badge/GPU-Not%20Required-lightgrey?style=flat)
![No API](https://img.shields.io/badge/External%20APIs-None-red?style=flat)

**Team:** Ayush | **Challenge:** Intelligent Candidate Discovery & Ranking | **Role:** Senior AI Engineer — Redrob AI (Series A)

> Recruiters go through hundreds of profiles and still often miss the right person — not because the talent isn't there, but because keyword filters can't see what actually matters. This system ranks candidates the way a great recruiter would: not by matching keywords, but by understanding who genuinely fits.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Setup & Installation](#setup--installation)
- [Architecture](#architecture)
- [Scoring Breakdown](#scoring-breakdown)
- [Honeypot Guard](#honeypot-guard)
- [Results](#results)
- [File Structure](#file-structure)
- [Design Decisions](#key-design-decisions)
- [Reproduce](#reproduce-command)

---

## Quick Start

```bash
git clone https://github.com/ayush2459/redrob-ranker.git
cd redrob-ranker
pip install -r requirements.txt
python rank.py --candidates candidates.jsonl --out team_ayush.csv
python validate_submission.py team_ayush.csv
```

---

## Setup & Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| RAM | < 2 GB |
| GPU | Not required |
| Network | Not required during ranking |

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` only needs:
```
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
```

No sentence-transformers, no PyTorch, no LLM APIs. Pure Python + stdlib for the core logic.

### Run the ranker

```bash
# Basic usage
python rank.py --candidates candidates.jsonl --out team_ayush.csv

# With custom top-N
python rank.py --candidates candidates.jsonl --out output.csv --top 100

# Validate output before submitting
python validate_submission.py team_ayush.csv
```

### Expected output

```
Loading candidates from candidates.jsonl...
Loaded 100,000 candidates.
Scoring candidates...
Valid candidates (non-honeypot): 99,981
Writing top 100 to team_ayush.csv...
Done. Output: team_ayush.csv

Top 5 preview:
  #1 CAND_0077337 | score=0.7003 | Staff Machine Learning Engineer | 7.0yrs | Kochi, Kerala
  #2 CAND_0002025 | score=0.6814 | Senior AI Engineer | 5.9yrs | Trivandrum, Kerala
  #3 CAND_0088025 | score=0.6627 | Staff Machine Learning Engineer | 8.6yrs | Jaipur, Rajasthan
```

---

## Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                   candidates.jsonl (100K)                    ║
╚══════════════════════════════╤═══════════════════════════════╝
                               │
                               ▼
              ┌────────────────────────────────┐
              │        HONEYPOT GUARD          │
              │  ✗ Keyword stuffers            │
              │  ✗ Impossible YOE profiles     │
              │  ✗ All-binary assessments      │
              └───────────────┬────────────────┘
                              │  19 filtered out
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │              6 SCORING COMPONENTS                   │
    │                                                     │
    │  ┌──────────────┐  ┌──────────────┐                │
    │  │ Skill Match  │  │ Career Fit   │                │
    │  │    35%       │  │    25%       │                │
    │  │              │  │              │                │
    │  │ Tiered JD    │  │ Title align  │                │
    │  │ taxonomy     │  │ + prod proof │                │
    │  └──────┬───────┘  └──────┬───────┘                │
    │         │                 │                         │
    │  ┌──────▼───────┐  ┌──────▼───────┐                │
    │  │  Experience  │  │  Assessment  │                │
    │  │    15%       │  │    10%       │                │
    │  │              │  │              │                │
    │  │ 5-9yr = 1.0  │  │ Platform     │                │
    │  │ sweet spot   │  │ test scores  │                │
    │  └──────┬───────┘  └──────┬───────┘                │
    │         │                 │                         │
    │  ┌──────▼───────┐  ┌──────▼───────┐                │
    │  │   Location   │  │  Education   │                │
    │  │    10%       │  │     5%       │                │
    │  │              │  │              │                │
    │  │ India = 1.0  │  │ IIT/NIT      │                │
    │  │ Reloc = 0.75 │  │ tier weight  │                │
    │  └──────┬───────┘  └──────┬───────┘                │
    │         └────────┬─────────┘                        │
    └──────────────────┼──────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │     BASE SCORE       │
            │  weighted Σ of 6     │
            │  components          │
            └──────────┬───────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │         ENGAGEMENT MULTIPLIER            │
    │                                          │
    │  Sub-signals:                            │
    │  • Availability   (last_active + flag)   │
    │  • Responsiveness (response_rate + time) │
    │  • Hirability     (interview + offer)    │
    │  • Platform activity (views + GitHub)    │
    │  • Profile quality  (completeness)       │
    │                                          │
    │  Range: [0.20 → 1.10]                   │
    │  Applied as: final = base × multiplier   │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │    FINAL SCORE       │
            │  + Reasoning text    │
            │  Sorted, top 100     │
            │  → team_ayush.csv    │
            └──────────────────────┘
```

---

## Scoring Breakdown

### Component Weights

| Component | Weight | Range | Key Signal |
|-----------|--------|-------|-----------|
| Skill Match | **35%** | 0–1 | Tiered JD taxonomy match |
| Career Fit | **25%** | 0–1 | Title + production evidence |
| Experience Years | **15%** | 0–1 | 5–9yr sweet spot |
| Assessment Scores | **10%** | 0–1 | Platform test scores |
| Location | **10%** | 0–1 | India / relocation |
| Education Tier | **5%** | 0–1 | Institution quality |
| **Engagement Multiplier** | **×** | 0.2–1.1 | Behavioral signals |

---

### 1. Skill Match (35%)

Scores skills against a hand-crafted 50+ term JD taxonomy in three tiers:

| Tier | Weight | Examples |
|------|--------|---------|
| Core | 3.0× | `embeddings`, `BM25`, `FAISS`, `RAG`, `semantic search`, `fine-tuning`, `NLP`, `retrieval` |
| Strong | 2.0× | `vector database`, `Pinecone`, `Weaviate`, `Qdrant`, `re-ranking`, `HuggingFace`, `PyTorch` |
| Supporting | 1.0× | `Python`, `Docker`, `AWS`, `FastAPI`, `SQL` |
| Downweighted | 0.5× | `LangChain`, `OpenAI`, `GPT` *(per explicit JD guidance)* |

Each matched skill is further adjusted by:
- **Proficiency multiplier:** `expert=1.2`, `advanced=1.0`, `intermediate=0.7`, `beginner=0.4`
- **Endorsement bonus:** log-scaled up to +0.3 (real experts earn endorsements)
- **Duration bonus:** up to +0.2 for 24+ months of use

> **Example:** A candidate with `advanced` NLP (37 endorsements, 26 months) scores:
> `3.0 × 1.0 × (1 + 0.25 + 0.18) = 4.29` for that skill alone.

---

### 2. Career Fit (25%)

Three sub-signals combined:

**Title Alignment**
```
High (1.0):   ML Engineer, AI Engineer, NLP Engineer, Search Engineer, Applied Scientist
Medium (0.6): Data Engineer, Platform Engineer, Tech Lead
Low (0.2):    Analyst, Manager, Designer, QA
```

**Production Evidence** — career descriptions scanned for:
`deployed` · `serving` · `scale` · `latency` · `a/b testing` · `inference` · `production`

Each keyword hit adds to a bonus (max +0.2). A candidate who shipped a vector search system scores higher than one who only studied it.

**Research-Only Penalty** — titles containing `research scientist`, `postdoc`, `academic researcher` receive a 50% penalty, directly reflecting the JD's explicit disqualifier: *"We've tried pure researchers twice and it didn't work for either side."*

---

### 3. Experience Years (15%)

Calibrated directly from the JD's stated preference of 5–9 years:

| Years | Score | Reasoning |
|-------|-------|-----------|
| 5–9 | 1.00 | Sweet spot per JD |
| 4–5 | 0.85 | Slightly junior but viable |
| 9–12 | 0.80 | Slightly over-experienced |
| 3–4 | 0.60 | Under threshold |
| 12–15 | 0.65 | May be over-scoped for role |
| < 3 | 0.35 | Too junior |
| > 15 | 0.50 | Risk of overqualification |

---

### 4. Assessment Scores (10%)

Redrob platform skill test scores (`skill_assessment_scores`) are ground-truth signals — they can't be keyword-stuffed. Scores for relevant skills (NLP, ML, Python, retrieval, embeddings) are weighted at full value; irrelevant assessments at half value.

---

### 5. Location (10%)

| Situation | Score |
|-----------|-------|
| India-based | 1.00 |
| Overseas + willing to relocate | 0.75 |
| Overseas, fixed | 0.30 |

The role is Pune/Noida hybrid, open to Tier-1 Indian city relocations.

---

### 6. Education Tier (5%)

| Institution | Score |
|-------------|-------|
| IIT / NIT / BITS / IISc / IIIT | 1.00 |
| VIT / Manipal / SRM / Symbiosis | 0.75 |
| Tier-3 colleges | 0.55 |
| Unknown | 0.45 |

Education is the lowest-weighted component intentionally — the JD emphasises production judgment over pedigree.

---

### Engagement Multiplier (×0.2 – ×1.1)

Applied **multiplicatively** — not additively — so a ghost candidate cannot outscore a reachable one regardless of skills.

Five sub-signals averaged:

| Signal | Source Fields | Logic |
|--------|--------------|-------|
| Availability | `last_active_date` + `open_to_work_flag` | Active ≤7d + open = 1.0; inactive >180d = 0.1 |
| Responsiveness | `recruiter_response_rate` + `avg_response_time_hours` | Rate is primary; <4hr response = +0.2 bonus |
| Hirability | `interview_completion_rate` + `offer_acceptance_rate` | 60/40 weighted average |
| Platform activity | `profile_views_30d` + `saved_by_recruiters` + `github_activity_score` | Log-scaled social proof |
| Profile quality | `profile_completeness_score` + verified email/phone/LinkedIn | Trust signals |

> **Why multiplier not additive?**
> Additive: `ghost_score = 0.90 (skills) + 0.10 (engagement) = 1.00` → ranked first, recruiter gets ghosted.
> Multiplier: `ghost_score = 0.90 × 0.22 = 0.20` → ranked appropriately low.

---

## Honeypot Guard

The dataset contains ~80 synthetic traps. Submissions with >10 honeypots in top 100 are auto-disqualified.

Three detection checks run before scoring:

| Check | Logic | Target |
|-------|-------|--------|
| **Keyword Stuffer** | 8+ skills with zero endorsements AND `expert` proficiency | Fake keyword stuffers |
| **Binary Assessments** | 3+ assessment scores all exactly 0 or 100 | Synthetic profile artifacts |
| **YOE Inflation** | Claimed `years_of_experience` > 3× total career history months | Impossible seniority claims |

**Result:** 19 candidates flagged and excluded before ranking.

---

## Results

### Score Distribution (Top 100)

```
Score Range  │ Count │ Bar
─────────────┼───────┼──────────────────────────────────
0.68 – 0.70  │   2   │ ██
0.64 – 0.68  │   5   │ █████
0.60 – 0.64  │   6   │ ██████
0.56 – 0.60  │  30   │ ██████████████████████████████
0.52 – 0.56  │  57   │ █████████████████████████████████████████████████████████
```

### Top 10 Candidates

| Rank | Candidate ID | Score | Title | Exp | Location | Response Rate | GitHub | Top Skills |
|------|-------------|-------|-------|-----|----------|--------------|--------|-----------|
| 1 | CAND_0077337 | 0.7003 | Staff ML Engineer | 7.0 yr | Kochi, KL | 95% | 68 | LlamaIndex, LLMs, Information Retrieval |
| 2 | CAND_0002025 | 0.6814 | Senior AI Engineer | 5.9 yr | Trivandrum, KL | 80% | 97 | Deep Learning, NLP, FAISS |
| 3 | CAND_0088025 | 0.6627 | Staff ML Engineer | 8.6 yr | Jaipur, RJ | 83% | 75 | Elasticsearch, Learning to Rank, RAG |
| 4 | CAND_0005538 | 0.6625 | Senior AI Engineer | 5.9 yr | Kolkata, WB | 81% | 58 | Information Retrieval, pgvector |
| 5 | CAND_0018499 | 0.6605 | Senior ML Engineer | 7.2 yr | Noida, UP | 61% | 95 | scikit-learn, Recommendation Systems, Learning to Rank |
| 6 | CAND_0081846 | 0.6461 | Lead AI Engineer | 6.7 yr | Jaipur, RJ | 73% | 34 | Information Retrieval, Learning to Rank, Vector Search |
| 7 | CAND_0046525 | 0.6455 | Senior ML Engineer | 6.1 yr | Pune, MH | 88% | 37 | pgvector, Information Retrieval |
| 8 | CAND_0011687 | 0.6331 | Senior NLP Engineer | 7.8 yr | Indore, MP | 89% | 76 | Semantic Search, LangChain |
| 9 | CAND_0071974 | 0.6291 | Senior AI Engineer | 7.8 yr | Vizag, AP | 76% | 83 | Speech Recognition, PEFT, Qdrant |
| 10 | CAND_0005260 | 0.6245 | Senior NLP Engineer | 5.2 yr | Chennai, TN | 86% | — | NLP, Semantic Search, Python |

### Key observations
- All top 10 are India-based, 5–9 yr experience band, with production ML/retrieval titles
- Average response rate in top 10: **81%** — these are reachable candidates
- 7 of top 10 have GitHub activity score > 50
- Zero honeypots in top 100

---

## File Structure

```
redrob-ranker/
├── rank.py                   # Main ranker — single file, no heavy deps
├── requirements.txt          # numpy, pandas, tqdm only
├── README.md
├── validate_submission.py    # Official validator (provided by Redrob)
├── submission_metadata.yaml  # Team + compute + methodology declaration
└── team_ayush.csv            # Final submission — top 100 ranked candidates
```

---

## Key Design Decisions

### Why not sentence-transformers or embeddings?
The submission spec enforces CPU-only, 5-minute runtime, no network. Loading a 400 MB model and encoding 100K candidates violates all three constraints. Feature engineering on structured data achieves comparable ranking quality at ~30s runtime.

### Why downweight LangChain/OpenAI in the taxonomy?
The JD explicitly states: *"If your AI experience consists primarily of recent LangChain-to-OpenAI projects, we probably won't move forward."* The taxonomy reflects the actual hiring bar, not the buzzword list.

### Why six components instead of one composite score?
Each component targets a different failure mode: experience range catches under/over-seniority, location catches overseas-fixed candidates, engagement catches ghost candidates, assessment scores catch keyword stuffers who failed the actual test.

### Why is engagement a multiplier, not additive?
See [Engagement Multiplier](#engagement-multiplier-02--11) section above. Short answer: additive engagement lets ghost candidates win.

---

## Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_ayush.csv
```

Runs end-to-end in ~30 seconds on CPU with 16 GB RAM. No GPU, no network, no external APIs.

---

## License

MIT License — free to use, modify, and distribute.
