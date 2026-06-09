# Redrob AI Hackathon — Candidate Ranker

**Team:** Ayush  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role:** Senior AI Engineer — Redrob AI (Series A)

---

## Quick Start

```bash
pip install -r requirements.txt
python rank.py --candidates candidates.jsonl --out team_ayush.csv
python validate_submission.py team_ayush.csv
```

**Runtime:** ~30s on CPU | **Memory:** <2 GB | **No GPU or network required**

---

## Architecture

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              Honeypot Guard                         │
│  - Zero-endorsement expert stuffers                 │
│  - All-zero/all-100 assessments                     │
│  - Phantom response rate / impossible YOE           │
└─────────────────────┬───────────────────────────────┘
                      │
      ┌───────────────┼───────────────────────────┐
      ▼               ▼               ▼           ▼
 Skill Score    Career Score   Exp Score   Location Score
 (TF-IDF-      (title align +  (5-9yr      (India +
  weighted)     desc keyword    sweet spot)  relocate)
                density +
                prod evidence)
      │               │               │           │
      └───────────────┴───────────────┴───────────┘
                      │
              ┌───────▼───────┐
              │  Base Score   │
              │  (weighted Σ) │
              └───────┬───────┘
                      │
              ┌───────▼───────────────────┐
              │  Engagement Multiplier    │
              │  - Availability (active?) │
              │  - Response rate          │
              │  - Interview completion   │
              │  - GitHub + platform      │
              │  Range: [0.2, 1.1]        │
              └───────┬───────────────────┘
                      │
              ┌───────▼───────┐
              │  Final Score  │
              │  + Reasoning  │
              └───────────────┘
```

## Scoring Weights

| Component | Weight | Signal |
|-----------|--------|--------|
| Skill Match | 35% | TF-IDF weighted match vs JD skill taxonomy |
| Career Fit | 25% | Title alignment + production keyword density |
| Experience | 15% | 5-9yr sweet spot scoring (per JD spec) |
| Assessment | 10% | Redrob skill assessment scores |
| Location | 10% | India-based / willing to relocate |
| Education | 5% | Tier-weighted institution quality |

**Engagement Multiplier (0.2–1.1x):** Availability, response rate, interview completion, GitHub activity, profile completeness.

## Key Design Decisions

### 1. Engagement as Multiplier, Not Additive
A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is a ghost hire. Engagement gates the base score rather than adding to it — a highly engaged mediocre candidate won't outscore a strong-but-inactive one, but a completely inactive candidate will be ranked below an equally-skilled but reachable peer.

### 2. Skill Taxonomy with Tier Weights
The JD explicitly distinguishes between candidates who understood retrieval/ranking *before* LLMs became fashionable vs. those who "just call OpenAI." The skill taxonomy assigns:
- **Weight 3.0:** Embeddings, BM25, FAISS, semantic search, RAG, NLP, fine-tuning
- **Weight 2.0:** Vector DBs, re-ranking, Hugging Face, PyTorch
- **Weight 0.5:** LangChain, OpenAI (explicitly downweighted per JD guidance)

### 3. Production Evidence Bonus
Career descriptions are scanned for production deployment signals (`deployed`, `serving`, `scale`, `latency`, `a/b`). Candidates who write about production systems rather than research demos get a bonus — directly addressing the JD's "we've tried pure researchers twice" disqualifier.

### 4. Honeypot Guard
Three checks: (a) 8+ skills with zero endorsements at "expert" level → keyword stuffer, (b) all assessment scores exactly 0 or 100 → synthetic outlier, (c) years-of-experience claimed >3x what career history shows → impossible profile.

## File Structure

```
redrob-ranker/
├── rank.py                  # Main ranker (single-file, no heavy deps)
├── requirements.txt
├── README.md
├── validate_submission.py   # Official validator (provided by Redrob)
├── submission_metadata.yaml
└── team_ayush.csv           # Final submission output
```

## Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_ayush.csv
```
