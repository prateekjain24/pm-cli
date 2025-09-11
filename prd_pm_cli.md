PRD — pm-kit (Docs-as-Code for PMs) — MVP

1) Product summary (why this exists)

Vision: PM artifacts (PRDs, roadmaps, OKRs, personas, release notes) live in Git, flow through code-grade review gates, and mirror to Confluence/Notion—no flags, no copy-paste.  ￼

Problem today (personal project reality):
	•	You draft in Markdown or a doc tool, then manually sync to Confluence/Notion for stakeholders. It’s slow and drifts.
	•	Review quality is inconsistent; vague language sneaks through; metrics go missing.
	•	Stories/epics in Jira/GitHub don’t trace cleanly to the PRD lines that spawned them.

Why now / confidence: The industry already embraces docs-as-code—documentation in repos, reviewed via PRs, built via CI/CD. Backstage TechDocs and Doctave are visible proof points of this workflow.  ￼ ￼

⸻

2) Primary users & goals

Primary user: Individual PM (you) who prefers Git + Markdown but must publish to a company wiki.

Secondary users:
	•	Design & Engineering leads (reviewers): want clear gates and fast PR reviews.
	•	Leaders/CS/Marketing (readers): want stable wiki URLs and up-to-date content.

Top user goals (JTBD style):
	1.	Author once, publish everywhere: Draft locally in Markdown; push a button to mirror to Confluence (MVP). Validate via Confluence REST API (create/update pages).  ￼
	2.	Quality by default: Run opinionated gates (lint for ambiguity, missing metrics, broken links) that fail the PR and annotate inline using Reviewdog + GitHub Actions; protect main with required checks.  ￼ ￼
	3.	Traceability without toil: Turn PRD stories into Jira issues with back-links; re-running is idempotent (no dupes). Uses Jira Cloud REST “create issue.”  ￼
	4.	Review where comments are: Pull wiki comments and regenerate from the earliest impacted phase, then re-publish. Confluence v2 exposes endpoints for footer/inline comments; nested retrieval may require multiple calls.  ￼ ￼

⸻

3) Scope for MVP (what we will ship)

CLI verbs (zero-flag defaults): pm new, pm run, pm status, pm publish, pm sync issues, pm release draft. Phased, deterministic pipeline with hashes; repo layout is opinionated.  ￼

In-scope MVP capabilities
	•	Repo scaffolding: .pmrc.yaml, template pack pinning, CI job, PR template, CODEOWNERS.  ￼
	•	PRD creation: scaffold /product/prds/<slug>/01..05 + manifest.yaml; pm run composes final PRD. Blocks TBD/ambiguous terms.  ￼
	•	Quality gates in PR: local pm status, then GitHub Action with Reviewdog annotations; branch protection requires successful checks before merge.  ￼ ￼
	•	Publish to Confluence (MVP): idempotent mirror of a tree (parent page, stable titles, body updates with storage format versioning).  ￼ ￼
	•	Traceability: pm sync issues turns user stories into Jira issues and writes back deep links.  ￼
	•	Release notes draft: from merged PRs/issues since last tag.  ￼

Nice-to-have (stretch, not required for MVP)
	•	Notion publisher (create/update pages/blocks; read comments).  ￼
	•	Roadmap/OKRs as YAML → rendered docs (Mermaid/Gantt).  ￼

Out of scope (MVP)
	•	SSO/RBAC, enterprise packs, audit exports.  ￼
	•	Rich WYSIWYG editors; the bet is Markdown + CLI.

⸻

4) User experience (flows)

4.1 Setup
	1.	pm new project → writes .pmrc.yaml and CI workflow, detects Confluence env vars (space, parent ID).  ￼
	2.	Push to GitHub; enable branch protection to require “pm-gates” status.  ￼

4.2 Author & gate
	1.	pm new prd "Session Replay × Funnel Analysis" → scaffolds phases.
	2.	Edit Markdown.
	3.	pm status → local gate run with the same rules as CI.
	4.	Open PR → Reviewdog adds inline annotations for failures (ambiguous words, missing metrics/persona refs).  ￼

4.3 Publish to wiki
	1.	pm publish → creates/updates Confluence pages; uses storage/atlas_doc formats and versioning. Idempotent by content hash.  ￼ ￼

4.4 Traceability
	1.	Add stories in 03_requirements.md with front-matter.
	2.	pm sync issues → creates Jira issues; stores mapping in manifest.yaml.  ￼

4.5 Comment-driven regen
	1.	Stakeholders comment on Confluence pages.
	2.	pm review <wiki-url> → fetches top-level footer/inline comments; if nested, follows links to load children; maps to phases → rerun from earliest phase → pm publish.  ￼ ￼

⸻

5) Functional requirements (MVP)

FR-1 Repo scaffolding
	•	Generate .pmrc.yaml (publisher config, gates, template pack).
	•	Add CI workflow pm-gates.yml + CODEOWNERS for /product/**.
	•	Acceptance: Fresh repo can run pm new prd "<name>" without manual config.  ￼

FR-2 Deterministic phases & cache
	•	Hash inputs per phase; pm run is a no-op when unchanged.
	•	Acceptance: second run produces zero file diffs; .cache/ updated only on change.  ￼

FR-3 Gates & annotations
	•	Rules: “no TBD/ambiguous”, metrics present, personas referenced, links valid.
	•	CI posts inline annotations via Reviewdog; main protected with required status.
	•	Acceptance: Failing gates block merge until fixed.  ￼ ￼

FR-4 Confluence publisher
	•	Create/update pages with correct parent, title, and body; maintain versioning; respect storage/atlas_doc formats.
	•	Acceptance: Re-publishing unchanged content performs no updates; publishing changed content bumps version.  ￼ ￼

FR-5 Traceability to Jira
	•	Parse stories with IDs and acceptance criteria; create/update issues; write back deep links.
	•	Acceptance: Re-running doesn’t duplicate issues; each story shows an issue link and back-link.  ￼

FR-6 Release notes draft
	•	Collect merged PRs/issues since last tag; categorize New/Improvements/Fixes/Breaking; inject “why it matters” from PRD where linked.
	•	Acceptance: A single Markdown file generated under /product/releases/.  ￼

⸻

6) Non-functional requirements
	•	Local-first: Works offline except on publish/sync.
	•	Idempotent & deterministic: Same inputs → same outputs; hashes gate updates.  ￼
	•	Security: Tokens via env vars; never logged.
	•	Performance: pm status < 5s on typical PRD; pm publish network-bound.
	•	Extensibility: Adapters for publishers/issues are pluggable (Confluence first; Notion later).  ￼

⸻

7) Success metrics (first 60–90 days)
	•	≥70% of authored PM docs start in the repo (not wiki).  ￼
	•	Median PRD PR cycle time ≤ 48h (open→merge).  ￼
	•	≥90% of merged PRDs have stories synced to issues with back-links.  ￼
	•	A release notes draft is generated for every Git tag.  ￼

⸻

8) User stories & acceptance criteria (MVP)
	1.	Create a PRD quickly
As a PM, I can scaffold a new PRD and run phases without tweaking flags.
AC: pm new prd "<name>" creates all phase files; pm run produces 05_final_prd.md with no TBD/ambiguous words.  ￼
	2.	Catch vagueness early
As a reviewer, I see actionable annotations in the PR for ambiguous terms/missing metrics.
AC: Opening a PR triggers Reviewdog; merge is blocked until status checks pass.  ￼ ￼
	3.	Publish once to Confluence
As a PM, I publish to a known space/parent; updates are versioned.
AC: First publish creates pages; re-publish updates only changed ones (hash-based skip).  ￼ ￼
	4.	Create issues from stories
As a PM, each story becomes a Jira issue with back-links; re-running doesn’t duplicate.
AC: pm sync issues creates/updates issues and writes links into the PRD.  ￼
	5.	Regenerate from comments
As a PM, I paste a Confluence URL and regenerate from the earliest commented phase.
AC: pm review <url> fetches comments (footer/inline), maps to phase anchors, re-runs pipeline, and pm publish mirrors back.  ￼

⸻

9) Competitive/adjacent references (brief)
	•	Backstage TechDocs: “docs-like-code” model: Markdown lives with code; CI builds a site. Validates the pattern we’re applying to PM docs.  ￼
	•	Doctave: Docs-as-code workflow (Git + CI + reviews). Confirms that repos + PRs are the right primitives.  ￼

⸻

10) Risks & mitigations
	•	Confluence comment APIs are fragmented: v2 splits footer/inline comments; nested replies require extra calls. Mitigation: implement top-level first, add child-fetch traversal iteratively.  ￼ ￼
	•	PR annotations from forks: GitHub limits token capabilities; Reviewdog gracefully degrades via logging annotations. Mitigation: document limitations; prefer same-repo PRs when possible.  ￼
	•	Version/format drift in Confluence: Need correct body formats (storage/atlas_doc) and version bumps. Mitigation: encapsulate publisher with strict tests and explicit version handling.  ￼

⸻

11) Analytics (lightweight)
	•	CLI emits anonymized event counts for: new prd, status pass/fail, publish, sync issues, review. Toggle via .pmrc.yaml. (No PII.)

⸻

12) Launch plan (personal-project scale)

Milestone A — Core loop (1–2 weeks)
	•	Scaffold + phase orchestrator + gates (pm new, pm run, pm status).
	•	GitHub Action + branch protection recipe; Reviewdog integration.  ￼ ￼

Milestone B — Confluence publisher (1 week)
	•	Create/update pages with parent/space config; version & format handling; hash-based skip.  ￼ ￼

Milestone C — Traceability (3–4 days)
	•	Jira create/update; back-links in PRD; idempotency.  ￼

Milestone D — Comment-driven regen (stretch)
	•	Fetch footer/inline comments; phase mapping; regen window; re-publish.  ￼

⸻

13) Open questions
	•	Start with Confluence-only for MVP, or add Notion publisher behind a flag? (Notion blocks & comments are supported but add surface area.)  ￼
	•	Minimal default gate dictionary for “ambiguous” terms—how strict should the initial set be?
	•	Where to store story→issue mapping? (Current proposal: manifest.yaml.)

⸻

14) Appendix — Information architecture & commands

/product/
  prds/<slug>/
    01_problem.md
    02_solution.md
    03_requirements.md
    04_prototype_prompts.md
    05_final_prd.md
    manifest.yaml
    .cache/
  personas/*.md
  okrs/*.yaml
  roadmap/roadmap.yaml
  releases/*.md
.pmrc.yaml

CLI: pm new project | pm new prd "<name>" | pm run | pm status | pm publish | pm sync issues | pm release draft  ￼

