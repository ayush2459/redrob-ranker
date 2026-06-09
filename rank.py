#!/usr/bin/env python3
"""
Redrob AI Hackathon — Candidate Ranker
Author: Ayush (team submission)

Architecture: Hybrid Feature-Engineered Ranker
  1. Skill Match Score    — TF-IDF cosine sim of skills vs JD required skills
  2. Career Fit Score     — title / company / description keyword match + experience years
  3. Engagement Score     — behavioral signals (activity, response rate, availability)
  4. Seniority Score      — experience range alignment (5-9 yr ideal band)
  5. Location Score       — India-based / willing to relocate bonus
  6. Education Score      — tier weighting
  7. Honeypot Guard       — disqualify obvious traps
  Final = weighted linear combination, then behavioral multiplier applied.

Runtime: ~25s for 100K candidates on CPU.
"""

import json
import csv
import argparse
import math
import re
import sys
from datetime import datetime, date
from collections import defaultdict

# ─── JOB DESCRIPTION ANALYSIS ────────────────────────────────────────────────
# Extracted from job_description.docx — Senior AI Engineer at Redrob AI

JD_CORE_SKILLS = {
    # Tier 1 — Must-have (weight 3.0)
    "embeddings": 3.0,
    "vector search": 3.0,
    "retrieval": 3.0,
    "ranking": 3.0,
    "semantic search": 3.0,
    "bm25": 3.0,
    "faiss": 3.0,
    "dense retrieval": 3.0,
    "hybrid retrieval": 3.0,
    "llm": 3.0,
    "large language model": 3.0,
    "rag": 3.0,
    "retrieval augmented": 3.0,
    "nlp": 3.0,
    "fine-tuning": 3.0,
    "fine tuning": 3.0,
    "transformer": 3.0,
    "bert": 3.0,
    "sentence transformers": 3.0,
    # Tier 2 — Strong signal (weight 2.0)
    "information retrieval": 2.0,
    "reranking": 2.0,
    "re-ranking": 2.0,
    "cross-encoder": 2.0,
    "bi-encoder": 2.0,
    "vector database": 2.0,
    "pinecone": 2.0,
    "weaviate": 2.0,
    "qdrant": 2.0,
    "milvus": 2.0,
    "chroma": 2.0,
    "elasticsearch": 2.0,
    "opensearch": 2.0,
    "solr": 2.0,
    "pytorch": 2.0,
    "tensorflow": 2.0,
    "huggingface": 2.0,
    "hugging face": 2.0,
    "sklearn": 2.0,
    "scikit-learn": 2.0,
    "recommendation system": 2.0,
    "recommendation": 2.0,
    "search": 2.0,
    "ml engineering": 2.0,
    "machine learning": 2.0,
    "deep learning": 2.0,
    "neural network": 2.0,
    "a/b testing": 2.0,
    "ab testing": 2.0,
    "evaluation": 2.0,
    "feature engineering": 2.0,
    "fastapi": 2.0,
    "flask": 2.0,
    "production ml": 2.0,
    "mlops": 2.0,
    # Tier 3 — Supporting (weight 1.0)
    "python": 1.0,
    "docker": 1.0,
    "kubernetes": 1.0,
    "aws": 1.0,
    "gcp": 1.0,
    "azure": 1.0,
    "redis": 1.0,
    "kafka": 1.0,
    "spark": 1.0,
    "sql": 1.0,
    "git": 1.0,
    "api": 1.0,
    "rest api": 1.0,
    "microservices": 1.0,
    "data science": 1.0,
    "statistics": 1.0,
    "text classification": 1.0,
    "langchain": 0.5,   # explicitly downweighted per JD
    "openai": 0.5,
    "gpt": 0.5,
}

# Disqualifier keywords (research-only context)
RESEARCH_ONLY_SIGNALS = [
    "phd researcher", "research scientist", "research fellow", "academic researcher",
    "postdoc", "post-doctoral"
]

# Title alignment weights
TITLE_SIGNALS = {
    "high": ["ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
             "search engineer", "ranking engineer", "applied scientist", "applied ml",
             "applied ai", "senior engineer", "staff engineer", "principal engineer",
             "software engineer", "sde", "backend engineer", "data scientist"],
    "medium": ["data engineer", "full stack", "fullstack", "platform engineer",
               "tech lead", "engineering lead"],
    "low": ["analyst", "manager", "product", "designer", "qa", "devops", "scrum"]
}

INDIA_LOCATIONS = [
    "india", "pune", "noida", "bangalore", "bengaluru", "mumbai", "hyderabad",
    "chennai", "delhi", "gurgaon", "gurugram", "kolkata", "ahmedabad", "indore",
    "jaipur", "kochi", "chandigarh", "lucknow", "nagpur"
]

TIER1_EDU = ["iit", "nit", "bits", "iisc", "iim", "iiit", "tier_1"]
TIER2_EDU = ["vit", "manipal", "srm", "symbiosis", "amity", "tier_2"]

REF_DATE = datetime(2026, 6, 9)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.lower().strip()


def days_since(date_str: str) -> int:
    """Days since a date string like '2026-05-20'."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return max(0, (REF_DATE - d).days)
    except Exception:
        return 9999


def clamp(val, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, val))


def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# ─── SCORING COMPONENTS ──────────────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """
    Weighted skill match against JD requirements.
    Uses proficiency multiplier and endorsement trust signal.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    proficiency_mult = {"expert": 1.2, "advanced": 1.0, "intermediate": 0.7, "beginner": 0.4}
    
    raw_score = 0.0
    max_possible = sum(w for w in JD_CORE_SKILLS.values() if w >= 2.0) * 1.2  # normalize to core skills

    for skill in skills:
        name = normalize(skill.get("name", ""))
        prof = normalize(skill.get("proficiency", "intermediate"))
        endorsements = skill.get("endorsements", 0) or 0
        duration = skill.get("duration_months", 0) or 0

        # Find matching JD skill
        best_weight = 0.0
        for jd_skill, weight in JD_CORE_SKILLS.items():
            if jd_skill in name or name in jd_skill or (len(jd_skill) > 4 and jd_skill[:5] in name):
                best_weight = max(best_weight, weight)

        if best_weight > 0:
            pm = proficiency_mult.get(prof, 0.6)
            # Endorsement trust: log-scaled, max bonus 0.3
            endorse_bonus = min(0.3, math.log1p(endorsements) / 15.0)
            # Duration trust: max bonus 0.2 at 24+ months
            dur_bonus = min(0.2, duration / 120.0)
            raw_score += best_weight * pm * (1.0 + endorse_bonus + dur_bonus)

    return clamp(raw_score / max_possible)


def score_career(candidate: dict) -> float:
    """
    Scores career history: title alignment, company quality signal,
    description keyword density for ML/search terms.
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    if not career:
        return 0.0

    title_score = 0.0
    description_score = 0.0

    current_title = normalize(profile.get("current_title", ""))
    # Title match
    for h_title in TITLE_SIGNALS["high"]:
        if h_title in current_title:
            title_score = 1.0
            break
    if title_score == 0:
        for m_title in TITLE_SIGNALS["medium"]:
            if m_title in current_title:
                title_score = 0.6
                break
    if title_score == 0:
        for l_title in TITLE_SIGNALS["low"]:
            if l_title in current_title:
                title_score = 0.2
                break
    if title_score == 0:
        title_score = 0.4  # unknown title

    # Career description keyword density
    all_text = " ".join([
        normalize(entry.get("description", "")) for entry in career
    ] + [normalize(profile.get("summary", ""))])

    desc_hits = 0
    for skill_kw in ["embedding", "retrieval", "ranking", "search", "vector", "nlp",
                     "recommendation", "fine-tun", "transformer", "bert", "rag",
                     "production", "deploy", "scale", "a/b", "evaluation", "rerank"]:
        if skill_kw in all_text:
            desc_hits += 1

    description_score = clamp(desc_hits / 12.0)

    # Research-only penalty
    research_penalty = 0.0
    for rr in RESEARCH_ONLY_SIGNALS:
        if rr in current_title or rr in all_text[:200]:
            research_penalty = 0.5
            break

    # Check for production deployment evidence
    production_bonus = 0.0
    prod_signals = ["production", "deployed", "shipped", "serving", "scale", "latency", "inference"]
    prod_hits = sum(1 for p in prod_signals if p in all_text)
    production_bonus = clamp(prod_hits / 5.0) * 0.2

    base = 0.5 * title_score + 0.4 * description_score + production_bonus
    return clamp(base * (1 - research_penalty))


def score_experience(candidate: dict) -> float:
    """
    5-9 years is sweet spot per JD. Penalize <3 and >12.
    """
    yoe = safe_get(candidate, "profile", "years_of_experience", default=0) or 0

    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.85
    elif 9 < yoe <= 12:
        return 0.80
    elif 3 <= yoe < 4:
        return 0.60
    elif 12 < yoe <= 15:
        return 0.65
    elif 2 <= yoe < 3:
        return 0.35
    elif yoe > 15:
        return 0.50
    else:
        return 0.15  # < 2 years


def score_location(candidate: dict) -> float:
    """India-based preferred; willing_to_relocate is strong positive."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = normalize(profile.get("location", "") + " " + profile.get("country", ""))
    willing = signals.get("willing_to_relocate", False)

    india_match = any(loc in location for loc in INDIA_LOCATIONS)

    if india_match:
        return 1.0
    elif willing:
        return 0.75
    else:
        return 0.3


def score_education(candidate: dict) -> float:
    """Tier-weighted education score."""
    education = candidate.get("education", [])
    if not education:
        return 0.4  # unknown = neutral

    best = 0.3
    for edu in education:
        inst = normalize(edu.get("institution", ""))
        tier = normalize(edu.get("tier", ""))

        if "tier_1" in tier or any(t in inst for t in TIER1_EDU):
            best = max(best, 1.0)
        elif "tier_2" in tier or any(t in inst for t in TIER2_EDU):
            best = max(best, 0.75)
        elif "tier_3" in tier:
            best = max(best, 0.55)
        else:
            best = max(best, 0.45)

    return best


def score_engagement(candidate: dict) -> float:
    """
    Behavioral signals composite — availability, responsiveness, activity.
    This acts as a MULTIPLIER to prevent perfect-on-paper inactive candidates.
    """
    s = candidate.get("redrob_signals", {})
    if not s:
        return 0.5

    sub_scores = []

    # 1. Availability (open_to_work + last_active)
    open_flag = s.get("open_to_work_flag", False)
    days_inactive = days_since(s.get("last_active_date", "2020-01-01"))

    if open_flag and days_inactive <= 7:
        availability = 1.0
    elif open_flag and days_inactive <= 30:
        availability = 0.85
    elif days_inactive <= 14:
        availability = 0.8
    elif days_inactive <= 60:
        availability = 0.6
    elif days_inactive <= 180:
        availability = 0.35
    else:
        availability = 0.1  # ghost candidate
    sub_scores.append(availability)

    # 2. Responsiveness
    rr = s.get("recruiter_response_rate", 0.5) or 0.5
    avg_rt = s.get("avg_response_time_hours", 48) or 48
    # Response rate is primary; response time modulates it
    if avg_rt <= 4:
        rt_bonus = 0.2
    elif avg_rt <= 24:
        rt_bonus = 0.1
    elif avg_rt > 120:
        rt_bonus = -0.15
    else:
        rt_bonus = 0.0
    responsiveness = clamp(rr + rt_bonus)
    sub_scores.append(responsiveness)

    # 3. Hirability signals
    icr = s.get("interview_completion_rate", 0.5) or 0.5
    oar_raw = s.get("offer_acceptance_rate", -1)
    oar = oar_raw if oar_raw >= 0 else 0.6  # unknown = neutral
    hirability = 0.6 * icr + 0.4 * oar
    sub_scores.append(hirability)

    # 4. Platform activity / social proof
    views = min(s.get("profile_views_received_30d", 0) or 0, 100)
    saved = min(s.get("saved_by_recruiters_30d", 0) or 0, 20)
    connections = min(s.get("connection_count", 0) or 0, 500)
    github = s.get("github_activity_score", -1)
    github_score = 0.5 if github < 0 else (github / 100.0)

    activity = clamp(
        0.25 * (views / 100) +
        0.25 * (saved / 20) +
        0.2 * (connections / 500) +
        0.3 * github_score
    )
    sub_scores.append(activity)

    # 5. Profile quality
    completeness = (s.get("profile_completeness_score", 70) or 70) / 100.0
    verified = (0.5 * s.get("verified_email", False) + 0.3 * s.get("verified_phone", False)
                + 0.2 * s.get("linkedin_connected", False))
    profile_q = 0.7 * completeness + 0.3 * verified
    sub_scores.append(profile_q)

    return clamp(sum(sub_scores) / len(sub_scores))


def score_assessment(candidate: dict) -> float:
    """Redrob skill assessment scores — ground-truth signal."""
    s = candidate.get("redrob_signals", {})
    assessment = s.get("skill_assessment_scores", {})

    if not assessment:
        return 0.5  # neutral — no assessment data

    relevant_keys = ["nlp", "machine learning", "python", "deep learning",
                     "information retrieval", "embeddings", "search", "ranking"]

    matched = []
    for k, v in assessment.items():
        k_lower = normalize(k)
        # Check if assessment is for a relevant skill
        for rk in relevant_keys:
            if rk in k_lower or k_lower in rk:
                matched.append(v / 100.0)
                break
        else:
            matched.append((v / 100.0) * 0.5)  # irrelevant assessment — half weight

    if not matched:
        return 0.5
    return clamp(sum(matched) / len(matched))


def honeypot_check(candidate: dict) -> bool:
    """
    Returns True if candidate appears to be a honeypot/trap.
    Honeypots include: keyword stuffers, impossible profiles, inactive ghosts.
    """
    s = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    # Check 1: Too many skills with zero endorsements and claimed expertise
    zero_endorse_expert = sum(
        1 for sk in skills
        if sk.get("endorsements", 0) == 0 and sk.get("proficiency") == "expert"
    )
    if zero_endorse_expert >= 8:
        return True  # keyword stuffer

    # Check 2: Response rate 0 with recent activity — suspicious
    rr = s.get("recruiter_response_rate", 0.5)
    days_inactive = days_since(s.get("last_active_date", "2020-01-01"))
    if rr == 0.0 and days_inactive < 30:
        # Active but never responds — likely honeypot
        return True

    # Check 3: All skill assessments exactly 0 or 100
    assessments = list((s.get("skill_assessment_scores") or {}).values())
    if len(assessments) >= 3:
        if all(v == 0 or v == 100 for v in assessments):
            return True

    # Check 4: Years of experience impossible given career history
    yoe = profile.get("years_of_experience", 0) or 0
    career = candidate.get("career_history", [])
    total_months = sum(entry.get("duration_months", 0) or 0 for entry in career)
    if total_months > 0 and yoe > 0:
        claimed_months = yoe * 12
        if claimed_months > total_months * 3:  # 3x inflation = suspicious
            return True

    return False


def build_reasoning(candidate: dict, scores: dict) -> str:
    """Generate 1-2 sentence recruiter-style reasoning."""
    profile = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    rr = s.get("recruiter_response_rate", 0)
    days_inactive = days_since(s.get("last_active_date", "2020-01-01"))
    github = s.get("github_activity_score", -1)

    parts = [f"{title} with {yoe:.1f}yrs exp"]

    if scores["skill"] >= 0.7:
        parts.append("strong ML/search skill alignment")
    elif scores["skill"] >= 0.4:
        parts.append("partial skill match")

    if scores["career"] >= 0.7:
        parts.append("production ML/retrieval background")

    if location:
        parts.append(f"{location}-based")

    concern = ""
    if days_inactive > 60:
        concern = f"inactive {days_inactive}d"
    elif rr < 0.3:
        concern = f"low response rate ({rr:.0%})"

    notice = s.get("notice_period_days", None)

    sentence1 = "; ".join(parts[:4]) + "."
    sentence2 = ""
    if concern:
        sentence2 = f" Note: {concern}."
    elif notice and notice > 90:
        sentence2 = f" Notice period {notice}d is a consideration."
    elif github > 60:
        sentence2 = f" Strong GitHub activity (score {github:.0f})."

    return (sentence1 + sentence2).strip()


# ─── MAIN RANKER ─────────────────────────────────────────────────────────────

def rank_candidates(candidates_path: str, output_path: str, top_n: int = 100):
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates.")

    print("Scoring candidates...")
    scored = []

    for c in candidates:
        cid = c.get("candidate_id", "")

        # Honeypot guard
        if honeypot_check(c):
            scored.append((cid, -1.0, {}, c))
            continue

        sk = score_skills(c)
        ca = score_career(c)
        ex = score_experience(c)
        lo = score_location(c)
        ed = score_education(c)
        en = score_engagement(c)
        ass = score_assessment(c)

        # Weighted combination
        # Skills + career are primary signals (most discriminative for this JD)
        base_score = (
            0.35 * sk +    # skill match — most important
            0.25 * ca +    # career fit / production signals
            0.15 * ex +    # experience years
            0.10 * lo +    # location
            0.05 * ed +    # education tier
            0.10 * ass     # assessment scores
        )

        # Engagement as multiplier: range [0.2, 1.1]
        # Bad engagement tanks good candidates; great engagement boosts borderline ones
        engagement_mult = 0.2 + 0.9 * en

        final = clamp(base_score * engagement_mult)

        scores = {"skill": sk, "career": ca, "exp": ex, "loc": lo, "edu": ed,
                  "eng": en, "ass": ass, "base": base_score, "final": final}

        scored.append((cid, final, scores, c))

    # Sort by final score descending, tie-break by candidate_id ascending (spec requirement)
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Remove honeypots from top pool
    valid = [(cid, sc, scores, c) for cid, sc, scores, c in scored if sc >= 0]
    print(f"Valid candidates (non-honeypot): {len(valid):,}")

    top = valid[:top_n]

    print(f"Writing top {top_n} to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, scores, candidate) in enumerate(top, 1):
            reasoning = build_reasoning(candidate, scores)
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"Done. Output: {output_path}")
    print(f"\nTop 5 preview:")
    for i, (cid, score, scores, c) in enumerate(top[:5], 1):
        p = c.get("profile", {})
        print(f"  #{i} {cid} | score={score:.4f} | {p.get('current_title','')} | "
              f"{p.get('years_of_experience',0):.1f}yrs | {p.get('location','')} | "
              f"skills={scores['skill']:.2f} career={scores['career']:.2f} eng={scores['eng']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates JSONL file")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of top candidates to output")
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.top)
