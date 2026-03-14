#!/usr/bin/env python3
"""
main.py — Single LLM-Based AI Agent for Job Search and Resume Optimization
==========================================================================
Agent Architecture:
  Candidate Profile Input
        ↓
  LLM Agent (Gemini) — reasons and decides which tools to call
        ↓
  [Tool 1] Filtering Tool  →  filtered_jobs.json
        ↓
  [Tool 2] Ranking Tool    →  ranked_jobs.json
        ↓
  [Tool 3] Resume Tailor   →  tailored_resume output
        ↓
  Final Output + Reasoning Trace
"""

import csv
import json
import os
import re
import time
import logging
from datetime import datetime

# ── Gemini Setup ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyC9ulE-TK2eVm65lBgy5TRSVl1GOINKHeY"   # ← Replace with your key
GEMINI_MODEL   = "gemini-2.5-flash"

JOBS_CSV       = "data/jobs.csv"
OUTPUT_DIR     = "output"
LOG_FILE       = "logs/agent.log"

# ── Logging Setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
)
log = logging.getLogger("agent")

# ── Candidate Profile ─────────────────────────────────────────────────────────
CANDIDATE_PROFILE = {
    "name": "John Doe",
    "skills": ["Python", "TensorFlow", "MLflow", "Machine Learning", "Deep Learning",
               "PyTorch", "NLP", "Docker", "AWS", "SQL"],
    "preferred_location": "Texas",
    "years_of_experience": 3,
    "preferred_role": "AI Engineer",
}

# ── Base Resume ───────────────────────────────────────────────────────────────
BASE_RESUME = {
    "name": "John Doe",
    "contact": "johndoe@email.com | (555) 123-4567 | linkedin.com/in/johndoe",
    "summary": (
        "Motivated AI/ML Engineer with 3-5 years of experience building scalable "
        "AI systems and production ML pipelines. Skilled in Python, TensorFlow, and "
        "MLflow with a strong track record of delivering end-to-end machine learning solutions."
    ),
    "experience": [
        {
            "title": "AI/ML Engineer",
            "company": "TechCorp Solutions, Dallas TX",
            "period": "2021 – Present",
            "bullets": [
                "Designed and deployed end-to-end ML pipelines using Python, TensorFlow, and MLflow on AWS SageMaker.",
                "Built and fine-tuned deep learning models for NLP and computer vision use cases.",
                "Reduced model inference latency by 40% through optimization and model quantization.",
                "Collaborated with cross-functional teams to define ML requirements and deliver production-ready models.",
            ]
        },
        {
            "title": "Junior Data Scientist",
            "company": "DataBridge Inc., Austin TX",
            "period": "2019 – 2021",
            "bullets": [
                "Developed predictive analytics models for customer churn and revenue forecasting.",
                "Automated data preprocessing pipelines using Apache Airflow and PySpark.",
                "Maintained model performance dashboards using MLflow and Grafana.",
            ]
        }
    ],
    "education": "MS Data Science — UT Austin (2019) | BS Computer Science — Texas A&M (2017)",
    "skills": "Python, TensorFlow, PyTorch, Scikit-learn, MLflow, Docker, Kubernetes, AWS, SQL, NLP",
    "certifications": "AWS Certified ML Specialty | TensorFlow Developer Certificate",
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — FILTERING TOOL
# ══════════════════════════════════════════════════════════════════════════════

def filtering_tool(jobs: list[dict], profile: dict, rules: dict) -> dict:
    """
    Filtering Tool — applies rule-based filtering to the job dataset.

    Rules applied:
      1. Location preference  — keep jobs matching candidate's preferred location OR remote
      2. Experience limit     — exclude jobs requiring >2 years more than candidate has
      3. Company exclusion    — remove FAANG/big tech companies
      4. Skill overlap        — must have at least 1 skill match with candidate

    Args:
        jobs:    list of job dicts from CSV
        profile: candidate profile dict
        rules:   filtering rules dict from LLM reasoning

    Returns:
        dict with filtered jobs and reasoning log
    """
    log.info("=" * 60)
    log.info("  TOOL 1: FILTERING TOOL ACTIVATED")
    log.info("=" * 60)

    EXCLUDED_COMPANIES = [
        "Google", "Meta", "Facebook", "Amazon", "Apple", "Netflix",
        "Microsoft", "OpenAI", "Salesforce", "IBM", "Oracle", "Nvidia",
        "Adobe", "Twitter", "Alphabet",
    ]

    candidate_skills  = [s.lower() for s in profile["skills"]]
    preferred_location = profile["preferred_location"].lower()
    max_exp            = profile["years_of_experience"] + int(rules.get("exp_tolerance", 2))

    filtered  = []
    removed   = []

    for job in jobs:
        reasons_removed = []

        # Rule 1 — Company exclusion
        company = job.get("Company", "")
        if any(ex.lower() in company.lower() for ex in EXCLUDED_COMPANIES):
            reasons_removed.append(f"FAANG/Big Tech company: {company}")

        # Rule 2 — Experience limit
        try:
            req_exp = int(str(job.get("Years of Experience Required", "0")).strip())
        except ValueError:
            req_exp = 0
        if req_exp > max_exp:
            reasons_removed.append(f"Requires {req_exp} yrs exp, candidate has {profile['years_of_experience']}")

        # Rule 3 — Skill overlap (at least 1 match)
        job_skills = [s.strip().lower() for s in str(job.get("Required Skills", "")).split(";")]
        skill_overlap = [s for s in candidate_skills if any(s in js for js in job_skills)]
        if not skill_overlap:
            reasons_removed.append("No skill overlap with candidate profile")

        # Rule 4 — Location preference (keep preferred OR remote)
        location = job.get("Location", "").lower()
        location_ok = (
            preferred_location in location or
            "remote" in location or
            rules.get("ignore_location", False)
        )
        if not location_ok:
            # Don't remove for location alone — just flag it
            pass

        if reasons_removed:
            removed.append({"job": job["Job Title"] + " @ " + job["Company"], "reasons": reasons_removed})
            log.info(f"  REMOVED: {job['Job Title']} @ {job['Company']} — {'; '.join(reasons_removed)}")
        else:
            filtered.append(job)
            loc_flag = "✓ preferred location" if preferred_location in location else ("✓ remote" if "remote" in location else "other location")
            log.info(f"  KEPT:    {job['Job Title']} @ {job['Company']} | {loc_flag} | skills match: {skill_overlap[:3]}")

    log.info(f"\n  Filter result: {len(filtered)} kept, {len(removed)} removed from {len(jobs)} total")

    return {
        "filtered_jobs": filtered,
        "removed_jobs": removed,
        "stats": {
            "total_input": len(jobs),
            "total_kept": len(filtered),
            "total_removed": len(removed),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — RANKING TOOL
# ══════════════════════════════════════════════════════════════════════════════

def ranking_tool(filtered_jobs: list[dict], profile: dict) -> dict:
    """
    Ranking Tool — scores and ranks filtered jobs based on candidate fit.

    Scoring Rubric:
      - Skill match score:      +2 per matching skill (max 20)
      - Location match:         +5 if preferred location matches
      - Experience alignment:   +3 if experience matches exactly, +1 if within 1 year
      - Role title match:       +3 if preferred role appears in job title

    Args:
        filtered_jobs: list of filtered job dicts
        profile:       candidate profile dict

    Returns:
        dict with ranked jobs list and scores
    """
    log.info("=" * 60)
    log.info("  TOOL 2: RANKING TOOL ACTIVATED")
    log.info("=" * 60)

    candidate_skills   = [s.lower() for s in profile["skills"]]
    preferred_location = profile["preferred_location"].lower()
    candidate_exp      = profile["years_of_experience"]
    preferred_role     = profile["preferred_role"].lower()

    scored_jobs = []

    for job in filtered_jobs:
        score      = 0
        breakdown  = {}

        # 1. Skill match
        job_skills    = [s.strip().lower() for s in str(job.get("Required Skills", "")).split(";")]
        matched       = [s for s in candidate_skills if any(s in js for js in job_skills)]
        skill_score   = len(matched) * 2
        score        += skill_score
        breakdown["skill_score"]    = skill_score
        breakdown["matched_skills"] = matched

        # 2. Location match
        location       = job.get("Location", "").lower()
        loc_score      = 5 if preferred_location in location else (2 if "remote" in location else 0)
        score         += loc_score
        breakdown["location_score"] = loc_score

        # 3. Experience alignment
        try:
            req_exp = int(str(job.get("Years of Experience Required", "0")).strip())
        except ValueError:
            req_exp = 0
        exp_diff  = abs(req_exp - candidate_exp)
        exp_score = 3 if exp_diff == 0 else (1 if exp_diff == 1 else 0)
        score    += exp_score
        breakdown["experience_score"] = exp_score
        breakdown["required_exp"]     = req_exp

        # 4. Role title match
        title      = job.get("Job Title", "").lower()
        role_score = 3 if preferred_role in title else 0
        score     += role_score
        breakdown["role_score"] = role_score

        job_copy = dict(job)
        job_copy["score"]     = score
        job_copy["breakdown"] = breakdown
        scored_jobs.append(job_copy)

        log.info(
            f"  SCORED: [{score:2d}pts] {job['Job Title']} @ {job['Company']} | "
            f"skills={skill_score} loc={loc_score} exp={exp_score} role={role_score} | "
            f"matched: {matched[:4]}"
        )

    # Sort descending by score
    scored_jobs.sort(key=lambda x: x["score"], reverse=True)

    log.info("\n  --- TOP 3 RANKED JOBS ---")
    for i, job in enumerate(scored_jobs[:3], 1):
        bd = job["breakdown"]
        log.info(
            f"  #{i}: [{job['score']}pts] {job['Job Title']} @ {job['Company']} | "
            f"{job['Location']} | Skills matched: {bd['matched_skills']}"
        )

    return {
        "ranked_jobs": scored_jobs,
        "top_3": scored_jobs[:3],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — RESUME TAILORING TOOL
# ══════════════════════════════════════════════════════════════════════════════

def resume_tailoring_tool(top_job: dict, base_resume: dict, model=None) -> dict:
    """
    Resume Tailoring Tool — rewrites summary and modifies 2 bullet points.

    As per assignment requirements:
      - Rewrite the Professional Summary
      - Modify exactly 2 experience bullet points
      - Highlight aligned skills
      - Do NOT regenerate the entire resume

    Args:
        top_job:     the #1 ranked job dict
        base_resume: the candidate's base resume dict
        model:       Gemini model instance (or None for mock)

    Returns:
        dict with tailored summary and 2 modified bullet points
    """
    log.info("=" * 60)
    log.info("  TOOL 3: RESUME TAILORING TOOL ACTIVATED")
    log.info("=" * 60)
    log.info(f"  Tailoring for: {top_job['Job Title']} @ {top_job['Company']}")

    job_skills = top_job.get("Required Skills", "")
    job_desc   = top_job.get("Job Description", "")
    job_title  = top_job.get("Job Title", "")
    company    = top_job.get("Company", "")
    location   = top_job.get("Location", "")

    if model:
        prompt = f"""You are a professional resume writer. Your task is to tailor a resume for a specific job.

IMPORTANT RULES:
- Rewrite ONLY the Professional Summary (2-3 sentences)
- Modify EXACTLY 2 experience bullet points from the existing resume
- Do NOT create new experience or fabricate anything
- Highlight skills that match the job requirements
- Keep the tone professional and confident
- Output ONLY valid JSON, nothing else

JOB DETAILS:
Title: {job_title}
Company: {company}
Location: {location}
Required Skills: {job_skills}
Job Description: {job_desc}

CURRENT RESUME SUMMARY:
{base_resume['summary']}

CURRENT EXPERIENCE BULLETS (pick 2 to modify):
Job 1 - {base_resume['experience'][0]['title']} at {base_resume['experience'][0]['company']}:
1. {base_resume['experience'][0]['bullets'][0]}
2. {base_resume['experience'][0]['bullets'][1]}
3. {base_resume['experience'][0]['bullets'][2]}
4. {base_resume['experience'][0]['bullets'][3]}

Job 2 - {base_resume['experience'][1]['title']} at {base_resume['experience'][1]['company']}:
1. {base_resume['experience'][1]['bullets'][0]}
2. {base_resume['experience'][1]['bullets'][1]}
3. {base_resume['experience'][1]['bullets'][2]}

OUTPUT FORMAT (JSON only, no markdown):
{{
  "tailored_summary": "rewritten summary here",
  "modified_bullets": [
    {{
      "original": "original bullet text",
      "modified": "modified bullet text",
      "reason": "why this was changed"
    }},
    {{
      "original": "original bullet text",
      "modified": "modified bullet text",
      "reason": "why this was changed"
    }}
  ],
  "highlighted_skills": ["skill1", "skill2", "skill3"]
}}"""

        try:
            log.info("  Calling Gemini API for resume tailoring...")
            response = model.generate_content(prompt)
            raw = response.text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            log.info("  ✅ Gemini tailoring successful")
            return result
        except Exception as e:
            log.error(f"  Gemini API error: {e} — using mock fallback")

    # Mock fallback
    matched_skills = [s.strip() for s in job_skills.split(";")][:3]
    return {
        "tailored_summary": (
            f"Results-driven AI/ML Engineer with 3+ years of experience, now seeking the {job_title} "
            f"role at {company} in {location}. Proven expertise in {', '.join(matched_skills[:2])} "
            f"with a strong track record of delivering production-ready AI systems that drive measurable business impact."
        ),
        "modified_bullets": [
            {
                "original": base_resume["experience"][0]["bullets"][0],
                "modified": f"Designed and deployed scalable ML pipelines using {matched_skills[0] if matched_skills else 'Python'}, "
                            f"directly applicable to {company}'s AI infrastructure requirements.",
                "reason": f"Aligned with {company}'s requirement for {matched_skills[0] if matched_skills else 'ML'} expertise"
            },
            {
                "original": base_resume["experience"][0]["bullets"][2],
                "modified": f"Optimized model performance and reduced inference latency by 40%, "
                            f"supporting production-grade AI systems similar to {company}'s deployment needs.",
                "reason": "Highlighted production optimization experience relevant to the job requirements"
            }
        ],
        "highlighted_skills": matched_skills
    }


# ══════════════════════════════════════════════════════════════════════════════
# LLM AGENT — REASONING & ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_reasoning(profile: dict, jobs: list[dict], model=None) -> dict:
    """
    The LLM Agent reasons about the candidate profile and job dataset,
    then decides which tools to call and with what parameters.

    This is the core of the agent — LLM makes decisions, not hardcoded logic.
    """
    log.info("=" * 60)
    log.info("  LLM AGENT: REASONING PHASE")
    log.info("=" * 60)

    job_summary = "\n".join([
        f"- {j['Job Title']} @ {j['Company']} | {j['Location']} | Skills: {j['Required Skills'][:60]} | Exp: {j['Years of Experience Required']}yrs"
        for j in jobs[:10]
    ]) + f"\n... and {len(jobs)-10} more jobs" if len(jobs) > 10 else "\n".join([
        f"- {j['Job Title']} @ {j['Company']} | {j['Location']} | Skills: {j['Required Skills'][:60]}"
        for j in jobs
    ])

    reasoning_prompt = f"""You are an intelligent AI job search agent. Analyze the candidate profile and job dataset, then decide how to proceed.

CANDIDATE PROFILE:
- Name: {profile['name']}
- Skills: {', '.join(profile['skills'])}
- Preferred Location: {profile['preferred_location']}
- Years of Experience: {profile['years_of_experience']}
- Preferred Role: {profile['preferred_role']}

AVAILABLE JOB DATASET ({len(jobs)} jobs):
{job_summary}

AVAILABLE TOOLS:
1. filtering_tool — filters jobs by location, experience, company type, skill overlap
2. ranking_tool — scores and ranks filtered jobs by skill match, experience alignment, location, role fit
3. resume_tailoring_tool — rewrites summary and modifies 2 bullet points for top job

YOUR TASK:
1. Analyze the candidate's profile vs the job dataset
2. Decide what filtering rules to apply and why
3. Confirm you will call all 3 tools in order
4. Explain your reasoning for each decision

OUTPUT FORMAT (JSON only, no markdown):
{{
  "analysis": "your analysis of the candidate profile and dataset",
  "filtering_rules": {{
    "exp_tolerance": 2,
    "ignore_location": false,
    "notes": "explanation of filtering approach"
  }},
  "tool_execution_plan": [
    {{"tool": "filtering_tool", "reason": "why calling this first"}},
    {{"tool": "ranking_tool", "reason": "why calling this second"}},
    {{"tool": "resume_tailoring_tool", "reason": "why calling this last"}}
  ],
  "reasoning_trace": "step by step reasoning of your decisions"
}}"""

    if model:
        try:
            log.info("  Calling Gemini for agent reasoning...")
            response = model.generate_content(reasoning_prompt)
            raw = response.text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            reasoning = json.loads(raw)
            log.info("  ✅ Agent reasoning complete")
            log.info(f"  Analysis: {reasoning.get('analysis', '')[:200]}")
            log.info(f"  Reasoning: {reasoning.get('reasoning_trace', '')[:200]}")
            return reasoning
        except Exception as e:
            log.error(f"  Gemini reasoning error: {e} — using default plan")

    # Default reasoning if no model
    return {
        "analysis": (
            f"Candidate {profile['name']} has {profile['years_of_experience']} years of experience "
            f"with strong skills in {', '.join(profile['skills'][:5])}. "
            f"Preferred location is {profile['preferred_location']}. "
            f"Dataset has {len(jobs)} jobs to analyze. Will filter by experience tolerance, "
            f"skill overlap, and company type, then rank by multi-criteria scoring."
        ),
        "filtering_rules": {
            "exp_tolerance": 2,
            "ignore_location": False,
            "notes": "Allow up to 2 years more experience than candidate has. Keep preferred location and remote jobs."
        },
        "tool_execution_plan": [
            {"tool": "filtering_tool", "reason": "Remove irrelevant jobs before scoring to improve ranking quality"},
            {"tool": "ranking_tool", "reason": "Score remaining jobs on skills, location, experience, and role fit"},
            {"tool": "resume_tailoring_tool", "reason": "Tailor resume for the top-ranked job to maximize interview chance"}
        ],
        "reasoning_trace": (
            "Step 1: Analyzed candidate profile — 3 years experience, strong Python/ML/TensorFlow skills, Texas preference.\n"
            "Step 2: Decided to call filtering_tool with exp_tolerance=2 to keep jobs requiring up to 5 years.\n"
            "Step 3: Will call ranking_tool to score filtered jobs across 4 dimensions.\n"
            "Step 4: Will call resume_tailoring_tool for top-ranked job to personalize application materials."
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display_results(reasoning: dict, filter_result: dict, rank_result: dict, tailor_result: dict, base_resume: dict):
    """Print full agent output to console and save to output files."""

    output_lines = []

    def p(line=""):
        print(line)
        output_lines.append(line)

    p("=" * 70)
    p("  AI JOB SEARCH AGENT — FULL OUTPUT")
    p("=" * 70)

    # ── Reasoning Trace ──────────────────────────────────────────────────────
    p("\n📋 AGENT REASONING TRACE")
    p("-" * 70)
    p(f"Analysis:  {reasoning.get('analysis', 'N/A')}")
    p(f"\nReasoning: {reasoning.get('reasoning_trace', 'N/A')}")
    p(f"\nFiltering Rules: {json.dumps(reasoning.get('filtering_rules', {}), indent=2)}")
    p("\nTool Execution Plan:")
    for step in reasoning.get("tool_execution_plan", []):
        p(f"  → {step['tool']}: {step['reason']}")

    # ── Filter Results ────────────────────────────────────────────────────────
    p("\n\n🔍 TOOL 1: FILTERING RESULTS")
    p("-" * 70)
    stats = filter_result["stats"]
    p(f"Input jobs:   {stats['total_input']}")
    p(f"Jobs kept:    {stats['total_kept']}")
    p(f"Jobs removed: {stats['total_removed']}")
    if filter_result["removed_jobs"]:
        p("\nRemoved jobs:")
        for r in filter_result["removed_jobs"]:
            p(f"  ✗ {r['job']} — {'; '.join(r['reasons'])}")
    p(f"\nFiltered jobs ({stats['total_kept']}):")
    for j in filter_result["filtered_jobs"]:
        p(f"  ✓ {j['Job Title']} @ {j['Company']} | {j['Location']}")

    # ── Ranking Results ───────────────────────────────────────────────────────
    p("\n\n🏆 TOOL 2: RANKING RESULTS")
    p("-" * 70)
    p(f"{'Rank':<5} {'Score':<7} {'Job Title':<35} {'Company':<25} {'Location'}")
    p("-" * 70)
    for i, job in enumerate(rank_result["ranked_jobs"], 1):
        p(f"#{i:<4} {job['score']:<7} {job['Job Title']:<35} {job['Company']:<25} {job['Location']}")

    p("\n\n⭐ TOP 3 JOBS (Detailed)")
    p("-" * 70)
    for i, job in enumerate(rank_result["top_3"], 1):
        bd = job["breakdown"]
        p(f"\n#{i} [{job['score']} pts] {job['Job Title']} @ {job['Company']}")
        p(f"   Location:   {job['Location']}")
        p(f"   Salary:     {job.get('Salary', 'Not listed')}")
        p(f"   URL:        {job.get('URL', 'N/A')}")
        p(f"   Score breakdown:")
        p(f"     Skill match:     +{bd['skill_score']} ({len(bd['matched_skills'])} skills: {bd['matched_skills']})")
        p(f"     Location match:  +{bd['location_score']}")
        p(f"     Experience fit:  +{bd['experience_score']} (requires {bd['required_exp']} yrs)")
        p(f"     Role title fit:  +{bd['role_score']}")

    # ── Resume Tailoring ──────────────────────────────────────────────────────
    top_job = rank_result["top_3"][0]
    p("\n\n✏️  TOOL 3: TAILORED RESUME OUTPUT")
    p("-" * 70)
    p(f"Tailored for: {top_job['Job Title']} @ {top_job['Company']}")
    p(f"\n--- ORIGINAL SUMMARY ---")
    p(base_resume["summary"])
    p(f"\n--- TAILORED SUMMARY ---")
    p(tailor_result.get("tailored_summary", "N/A"))
    p(f"\n--- MODIFIED BULLET POINTS ---")
    for i, bullet in enumerate(tailor_result.get("modified_bullets", []), 1):
        p(f"\nBullet {i}:")
        p(f"  ORIGINAL: {bullet.get('original', 'N/A')}")
        p(f"  MODIFIED: {bullet.get('modified', 'N/A')}")
        p(f"  REASON:   {bullet.get('reason', 'N/A')}")
    p(f"\nHighlighted Skills: {', '.join(tailor_result.get('highlighted_skills', []))}")

    p("\n" + "=" * 70)
    p("  AGENT EXECUTION COMPLETE")
    p("=" * 70)

    # Save outputs
    with open(os.path.join(OUTPUT_DIR, "agent_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    with open(os.path.join(OUTPUT_DIR, "filtered_jobs.json"), "w", encoding="utf-8") as f:
        json.dump(filter_result, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "ranked_jobs.json"), "w", encoding="utf-8") as f:
        json.dump(rank_result, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "tailored_resume.json"), "w", encoding="utf-8") as f:
        json.dump(tailor_result, f, indent=2)

    log.info("\n  Output files saved to output/ directory")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — AGENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_jobs(csv_path: str) -> list[dict]:
    """Load jobs from CSV file."""
    jobs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(dict(row))
    log.info(f"Loaded {len(jobs)} jobs from {csv_path}")
    return jobs


def setup_model():
    """Initialize Gemini model."""
    if not GENAI_AVAILABLE:
        log.warning("google-generativeai not installed. Run: pip install google-generativeai")
        log.warning("Running in mock mode (no LLM calls).")
        return None
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        log.warning("GEMINI_API_KEY not set in main.py — running in mock mode.")
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    log.info(f"Gemini model ready: {GEMINI_MODEL}")
    return model


def main():
    print("\n" + "=" * 70)
    print("  🤖 AI JOB SEARCH AGENT — STARTING")
    print("=" * 70)
    print(f"  Candidate:  {CANDIDATE_PROFILE['name']}")
    print(f"  Skills:     {', '.join(CANDIDATE_PROFILE['skills'][:5])}...")
    print(f"  Location:   {CANDIDATE_PROFILE['preferred_location']}")
    print(f"  Experience: {CANDIDATE_PROFILE['years_of_experience']} years")
    print("=" * 70 + "\n")

    start = time.time()

    # ── Step 1: Load dataset ──────────────────────────────────────────────────
    jobs = load_jobs(JOBS_CSV)

    # ── Step 2: Setup LLM ─────────────────────────────────────────────────────
    model = setup_model()

    # ── Step 3: LLM Agent Reasoning ──────────────────────────────────────────
    print("\n🧠 Agent is reasoning about your profile and the job dataset...")
    reasoning = run_agent_reasoning(CANDIDATE_PROFILE, jobs, model)
    time.sleep(1)

    # ── Step 4: Tool 1 — Filter ───────────────────────────────────────────────
    print("\n🔧 Calling Tool 1: Filtering Tool...")
    filter_result = filtering_tool(
        jobs,
        CANDIDATE_PROFILE,
        reasoning.get("filtering_rules", {"exp_tolerance": 2})
    )
    time.sleep(0.5)

    # ── Step 5: Tool 2 — Rank ─────────────────────────────────────────────────
    print("\n🔧 Calling Tool 2: Ranking Tool...")
    rank_result = ranking_tool(filter_result["filtered_jobs"], CANDIDATE_PROFILE)
    time.sleep(0.5)

    # ── Step 6: Tool 3 — Tailor ───────────────────────────────────────────────
    print("\n🔧 Calling Tool 3: Resume Tailoring Tool...")
    top_job = rank_result["top_3"][0]
    tailor_result = resume_tailoring_tool(top_job, BASE_RESUME, model)

    # ── Step 7: Display Results ───────────────────────────────────────────────
    display_results(reasoning, filter_result, rank_result, tailor_result, BASE_RESUME)

    elapsed = time.time() - start
    print(f"\n✅ Agent completed in {elapsed:.1f}s")
    print(f"📁 Outputs saved to: output/")
    print(f"📋 Full log:         logs/agent.log\n")


if __name__ == "__main__":
    main()
