# FairTeam AI

**FairTeam AI turns scattered teamwork records into a transparent, review-ready picture of contribution, workload balance, collaboration risk, and evidence quality.**

It is designed for instructors, project mentors, bootcamp operators, student teams, and organizations that need a fairer way to review team projects without relying only on self-reported percentages or raw activity counts.

FairTeam AI does **not** automatically assign a final grade. It helps a human reviewer understand what happened, identify where evidence is weak, ask better follow-up questions, and document a defensible review process.

---

## Why users choose FairTeam AI

### See contribution across the whole project

FairTeam AI combines multiple evidence sources instead of treating commit count as the entire project:

- GitHub commits, pull requests, issues, reviews, tests, and bug fixes
- Document revision history
- Presentation and script work
- Meeting attendance, decisions, and action items
- Assigned and completed responsibilities
- Self-evaluation and peer comments

### Reduce activity-count bias

Large copy-and-paste changes, repetitive commits, generated files, last-minute activity, and single-source dependence can make raw activity look more meaningful than it is. FairTeam AI keeps the original activity signal visible while also producing a quality-adjusted contribution view.

### Find problems before the final deadline

The dashboard highlights:

- Low-evidence participation
- Workload overload
- Missed or substituted responsibilities
- Communication and response issues
- Self-evaluation gaps
- Evidence that needs manual verification

This allows teams and instructors to intervene while the project can still be corrected.

### Keep every conclusion reviewable

Every score is accompanied by evidence, source coverage, confidence, quality flags, and recommended review actions. The system is designed to point reviewers back to original files rather than hide decisions inside a black box.

### Give team members a formal appeal path

Members can submit missing evidence or challenge an incorrect interpretation. Reviewers can move each appeal through submitted, under-review, accepted, or rejected states and leave a written decision note.

---

## Product highlights

### Evidence-adjusted contribution analysis

FairTeam AI calculates category-level signals for code, documents, slides, meetings, and role completion. Project-type presets and custom weights make the analysis usable for development, report-heavy, presentation-heavy, or balanced projects.

### Review Readiness Score

Before an instructor relies on a report, FairTeam AI checks whether the available evidence is sufficient for human review. The score considers:

- Core source coverage
- Average analysis confidence
- Conflict risk
- Unresolved appeals
- Quality and anti-gaming flags

The result is shown as:

- `Ready for human review`
- `Needs more evidence`
- `Insufficient for decision`

### AI-structured meeting evidence

When an OpenAI API key is configured, meeting notes can be converted into validated structured records with the OpenAI Responses API and schema-constrained outputs.

The AI layer extracts signals such as:

- Completed work
- Assigned action items
- Missed deadlines
- No-response incidents
- Task substitution and recovery
- Review activity
- Workload overload
- Conflict evidence

If no API key is available, or if the API call fails, FairTeam AI automatically uses a deterministic rule-based engine with the same output schema.

### AI Review Brief

Reviewers can generate a structured brief containing:

- Executive summary
- Decision-readiness assessment
- Priority findings
- Questions to ask the team
- Recommended next steps
- Review caveats

The brief never assigns a grade and is designed to support, not replace, human judgment. A deterministic brief is always available; OpenAI enhancement is optional.

### Privacy-first AI workflow

Before meeting notes are sent to OpenAI, FairTeam AI can mask:

- Email addresses
- Phone numbers
- Student IDs
- Tokens embedded in URLs

API keys are read from the current session or environment and are not written into reports, manifests, snapshots, or exported review packages.

### Reproducible analysis snapshots

Each analysis receives a deterministic fingerprint based on the active evidence and settings. Saved results include a versioned manifest with:

- Workspace and reviewer information
- Analysis ID
- Project type and normalized weights
- Member list
- Evidence source label
- AI model and privacy settings
- Review Readiness result
- Decision-support notice

### One-click review package

Users can download or save a ZIP package containing:

```text
reports/
├── professor_report.md
├── team_report.md
├── ai_review_brief.md
└── scoring_policy.md

data/
├── member_scores.csv
├── quality_audit.csv
└── meeting_insights.csv

manifest.json
```

This makes it easier to archive a review, share it with an authorized reviewer, or attach it to an assessment record.

### Multi-team comparison

Instructors can analyze multiple team folders and compare:

- Contribution inequality
- Conflict risk
- Number of high-risk members
- Average confidence
- Team size

---

## Typical user workflows

### Instructor or mentor

1. Open FairTeam AI.
2. Use the included sample project or upload the team's evidence files.
3. Select the project type and adjust category weights when necessary.
4. Run the analysis.
5. Check Review Readiness before interpreting contribution percentages.
6. Review member-level evidence, quality flags, and meeting signals.
7. Generate an AI Review Brief when an OpenAI key is available.
8. Resolve submitted appeals.
9. Download the complete review package for your records.

### Team member

1. Review the evidence shown for your work.
2. Check whether offline contributions are missing.
3. Submit an appeal with a file name, pull request, document section, or meeting record.
4. Track the review status and provide additional evidence when requested.

### Program or course operator

1. Organize each team as a separate input folder.
2. Run multi-team comparison.
3. Identify teams that need an early intervention.
4. Preserve review packages and snapshots for a consistent assessment process.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure optional environment variables

Copy the example file:

```bash
cp .env.example .env
```

Open `.env` and add your own values:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.6
OPENAI_TIMEOUT_SECONDS=45
OPENAI_MAX_RETRIES=2
FAIRTEAM_USE_LLM=false
GITHUB_TOKEN=
```

The application works without an OpenAI key. Add a key only when you want OpenAI-powered meeting analysis and review briefs.

Never commit the real `.env` file.

### 3. Start the dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser. Select **Sample Data** to explore the complete workflow immediately.

### 4. Run the CLI demo

```bash
python run_demo.py
```

The CLI generates a complete review package under `outputs/`.

To enable OpenAI features in the CLI:

```env
FAIRTEAM_USE_LLM=true
OPENAI_API_KEY=your_key_here
```

---

## Input files

A complete project bundle can contain the following files:

```text
meeting_notes.txt
github_log.csv
docs_revision.csv
slides_revision.csv
roles.csv
self_evaluation.csv
```

CSV templates can be downloaded directly from the dashboard.

### GitHub activity

Required columns:

```csv
member,commits,additions,deletions,files_changed,issues_closed,prs_merged,reviews,bugfix_commits,test_commits
A,28,3920,1110,46,9,12,8,7,6
```

Optional fields improve quality and anti-gaming checks:

```csv
commit_messages,unique_message_ratio,dominant_file_ratio,generated_files,timestamp
"fix | fix | update",0.33,0.72,4,2026-05-06 22:10:00
```

### Document revisions

```csv
member,edits,words_added,comments_resolved,sections_owned,suggestions_accepted,references_added
B,31,4820,18,5,14,8
```

### Presentation work

```csv
member,slides_edited,visuals_created,script_words,presenter_minutes
B,16,7,980,7
```

### Responsibilities

```csv
member,assigned_tasks,completed_tasks,late_tasks,critical_tasks
C,5,1,3,0
```

### Self-evaluation

```csv
member,self_claim_percent,claimed_main_work,peer_comment
C,25,Research and report support,I contributed ideas offline.
```

---

## Importing directly from GitHub

Create `github_log.csv` from a public repository:

```bash
python scripts/ingest_github.py \
  --repo owner/repo \
  --branch main \
  --max-commits 120 \
  --out sample_data/github_log.csv
```

For a private repository or higher rate limits:

```bash
python scripts/ingest_github.py \
  --repo https://github.com/owner/repo \
  --token "$GITHUB_TOKEN" \
  --out sample_data/github_log.csv
```

A token can also be entered temporarily in the dashboard. Session input is not persisted by FairTeam AI.

---

## Multi-team folder layout

```text
class_teams/
├── team01/
│   ├── meeting_notes.txt
│   ├── github_log.csv
│   ├── docs_revision.csv
│   ├── slides_revision.csv
│   ├── roles.csv
│   └── self_evaluation.csv
└── team02/
    ├── meeting_notes.txt
    ├── github_log.csv
    ├── docs_revision.csv
    ├── slides_revision.csv
    ├── roles.csv
    └── self_evaluation.csv
```

Enter the parent directory in the dashboard's **Multi-team Comparison** setting.

---

## Output files

Saving from the dashboard or running the CLI creates files such as:

```text
outputs/
├── fairteam_member_scores.csv
├── fairteam_quality_audit.csv
├── fairteam_meeting_insights.csv
├── professor_report.md
├── team_report.md
├── ai_review_brief.md
├── scoring_policy.md
├── analysis_manifest.json
├── fairteam_review_package.zip
└── snapshots/
```

---

## Trust, fairness, and responsible use

FairTeam AI follows these principles:

1. **Human review is mandatory.** The system does not produce a final grade or disciplinary decision.
2. **Missing logs are not proof of missing work.** Offline contributions must be collected through the appeal process.
3. **Original evidence takes priority.** Reviewers should inspect the actual commit, file, revision, or meeting record behind any important finding.
4. **Quality matters more than raw volume.** Large counts are compressed or quality-adjusted when appropriate.
5. **Workload overload is not misconduct.** Overload is treated as a team-management signal, not an automatic penalty.
6. **AI is optional and fault-tolerant.** Deterministic analysis remains available when an API key is missing or an API request fails.
7. **Secrets are not exported.** API keys are excluded from reports, manifests, snapshots, and ZIP packages.

Before using the product in a real educational or workplace setting, define who can access raw evidence, how long records are retained, how appeals are handled, and who makes the final decision.

---

## Testing

```bash
pytest -q
```

The test suite covers scoring, meeting attribution, GitHub ingestion, review readiness, environment configuration, and review-package generation.

---

## Project structure

```text
FairTeamAI/
├── app.py
├── run_demo.py
├── requirements.txt
├── .env.example
├── fairteam_ai/
│   ├── ai_review.py
│   ├── analyzers.py
│   ├── appeals.py
│   ├── batch.py
│   ├── config.py
│   ├── github_ingest.py
│   ├── interventions.py
│   ├── loaders.py
│   ├── meeting_ai.py
│   ├── models.py
│   ├── privacy.py
│   ├── quality.py
│   ├── readiness.py
│   ├── reporting.py
│   ├── scoring.py
│   ├── scoring_policy.py
│   ├── settings.py
│   └── workspace.py
├── sample_data/
├── scripts/
├── tests/
└── outputs/
```

---

## Current product boundary

This repository is a strong local-first review application suitable for demonstrations, pilots, classrooms, bootcamps, and small-team assessment workflows.

For a large institutional deployment, the next layer would typically include authenticated user accounts, organization-level permissions, encrypted database storage, centralized audit logs, retention policies, background job processing, and deployment-specific compliance controls.
