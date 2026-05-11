# Multi-Agent Codegen Workflow (Detailed Reference)

## 1) Objective
Build a reliable workflow that:
1. Creates folders and base files from user prompt requirements.
2. Retrieves grounding context from a common vector database.
3. Generates and inserts code only from retrieved evidence.
4. Self-improves through validation-driven reflection and bounded retries.
5. Tracks progress with a state table for restartability and observability.

---

## 2) High-Level Architecture

### Agents and Responsibilities
1. Intake and Intent Agent
- Parse user prompt into structured requirements.
- Identify project type, language, target modules, constraints.
- Produce a task graph.

2. Scaffold Agent
- Create folder structure and base files.
- Register created artifacts in state.

3. Retrieval Agent
- Build query from each coding task.
- Fetch top-k chunks from the common DB.
- Return evidence package with source IDs and scores.

4. Code Synthesis Agent
- Generate patch/content grounded strictly in evidence package.
- Emit file-level changes.

5. Validation Agent
- Run syntax/lint/compile/tests.
- Return machine-readable diagnostics.

6. Reflection Agent
- Triggered only on validation failure.
- Classify failure cause (retrieval miss, logic error, formatting, wrong file).
- Adjust retrieval strategy/prompt and schedule retry.

7. Orchestrator
- Executes task graph.
- Controls routing, retries, and terminal run status.
- Performs atomic state transitions.

---

## 3) Branch Flow Diagram (Reference)

```text
User Prompt
  |
  v
Intake and Intent Agent
  |
  +--> Parse requirements
  +--> Build task graph
  |
  v
Scaffold Agent
  |
  +--> Create folders
  +--> Create base files
  |
  v
For each coding task/file
  |
  v
Retrieval Agent --> Evidence Package
  |
  v
Code Synthesis Agent --> Patch
  |
  v
Validation Agent
  |
  +--> PASS -------------------------> Task Success
  |
  +--> FAIL
          |
          v
      Reflection Agent
          |
          +--> Diagnose root cause
          +--> Modify retrieval/prompt
          +--> retry_count += 1
          |
          +--> retry_count < max_retries ? YES -> back to Retrieval Agent
          |
          +--> NO ------------------------> Task Failed

All tasks done
  |
  +--> all critical tasks success? YES -> Run Completed
  |
  +--> otherwise ---------------------> Run Failed/Blocked
```

---

## 4) State Table Design
Use event-driven updates (not 10ms polling). Update only on transitions.

### A) run_state
Purpose: run-level lifecycle.

Fields:
- run_id (PK)
- user_prompt
- status: pending | running | blocked | completed | failed
- current_step
- total_tasks
- completed_tasks
- failed_tasks
- created_at
- updated_at

### B) task_state
Purpose: per-task execution, retries, and dependency control.

Fields:
- task_id (PK)
- run_id (FK)
- parent_task_id (nullable)
- task_type: scaffold | retrieve | synthesize | validate | reflect
- target_path
- input_hash
- status: queued | in_progress | success | failed | skipped
- retry_count
- max_retries
- priority
- dependency_ids (JSON/array)
- last_error_code
- last_error_message
- started_at
- finished_at
- updated_at

### C) artifact_state
Purpose: versioned output tracking and grounding traceability.

Fields:
- artifact_id (PK)
- run_id (FK)
- task_id (FK)
- path
- version
- checksum
- source_refs (retrieval chunk IDs)
- validation_status
- created_at

### D) event_log (optional but recommended)
Purpose: debugging and audit.

Fields:
- event_id (PK)
- run_id
- task_id (nullable)
- event_type
- payload (JSON)
- created_at

---

## 5) State Transitions

### Run transitions
- pending -> running (orchestrator starts)
- running -> blocked (external dependency unavailable)
- running -> failed (critical task exhausted retries)
- running -> completed (all critical tasks success)

### Task transitions
- queued -> in_progress (worker picks task)
- in_progress -> success (step passed)
- in_progress -> failed (step failed)
- failed -> queued (reflection schedules bounded retry)
- failed -> failed terminal (retry_count >= max_retries)

Rules:
1. Every transition must be atomic.
2. Increment retry_count only when entering retry path.
3. Never overwrite successful artifact versions; append new versions.

---

## 6) Retry and Self-Improvement Policy

Retry budget per coding task: 3 attempts.

Attempt strategy:
1. Attempt 1 (default)
- top_k = 5
- standard retrieval query from task text

2. Attempt 2 (recall boost)
- rewrite query with synonyms and function/register aliases
- top_k = 8
- include neighboring chunks from same source section

3. Attempt 3 (precision correction)
- keep best evidence from prior attempts
- restrict generation to failing function/file
- enforce stricter grounding instruction

Stop conditions:
1. Validation pass.
2. max_retries reached.
3. Critical dependency unavailable.

Learning signals to persist:
- Best-performing query rewrite patterns.
- Frequent failure signatures and corrective actions.
- Source sections repeatedly useful for specific module types.

---

## 7) Retrieval Contract

Evidence package contract returned by Retrieval Agent:
- query_used
- top_k
- chunks: list of
  - chunk_id
  - score
  - document_text
  - source_path
  - section_hint (optional)
- retrieval_timestamp

Grounding policy for synthesis:
1. Generate only from evidence package.
2. If evidence is insufficient, return explicit insufficiency signal.
3. Do not fabricate APIs/registers not present in evidence.

---

## 8) Validation Contract

Validation result contract:
- status: pass | fail
- checks:
  - syntax_check
  - lint_check
  - build_check
  - tests_check
- diagnostics:
  - file
  - line
  - code
  - message
- failure_category:
  - retrieval_miss
  - logic_error
  - compilation_error
  - formatting_error
  - dependency_error

Reflection routing uses failure_category to choose next correction.

---

## 9) Orchestrator Pseudocode

```python
initialize_run(prompt)
set_run_status("running")

plan = intake_agent(prompt)
scaffold_result = scaffold_agent(plan.scaffold_tasks)
record_artifacts(scaffold_result)

for task in plan.codegen_tasks:
    enqueue(task, max_retries=3)

while has_pending_or_retryable_tasks(run_id):
    task = pick_next_ready_task(run_id)
    mark_in_progress(task)

    evidence = retrieval_agent(task)
    patch = synthesis_agent(task, evidence)
    validation = validation_agent(task, patch)

    if validation.status == "pass":
        persist_patch_and_artifacts(task, patch, evidence)
        mark_success(task)
        continue

    reflection = reflection_agent(task, validation, evidence)
    if task.retry_count < task.max_retries and reflection.retry_recommended:
        apply_retry_strategy(task, reflection)
        requeue(task)
    else:
        mark_failed(task, validation)

finalize_run_status(run_id)
```

---

## 10) Polling and Scheduling Guidance

Do not run state updates every 10ms.

Recommended:
1. Event-driven transitions on task events.
2. Worker polling interval 500ms to 2s if queue-based.
3. Heartbeat for stuck-task detection every 5s to 30s.

This avoids DB pressure and improves stability.

---

## 11) Minimal Implementation Mapping to Your Current Project

Current components already present:
- Retrieval and ingestion flow exists.
- Reflection placeholder exists.

Next upgrades:
1. Add persistent state store (SQLite/Postgres).
2. Replace string-based error heuristic with validator-driven categories.
3. Add orchestrator loop with bounded retries.
4. Add artifact and event logging.
5. Keep successful tasks immutable during retries.

---

## 12) Suggested SQL Skeleton (Optional)

```sql
CREATE TABLE run_state (
  run_id TEXT PRIMARY KEY,
  user_prompt TEXT NOT NULL,
  status TEXT NOT NULL,
  current_step TEXT,
  total_tasks INTEGER DEFAULT 0,
  completed_tasks INTEGER DEFAULT 0,
  failed_tasks INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE task_state (
  task_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  parent_task_id TEXT,
  task_type TEXT NOT NULL,
  target_path TEXT,
  input_hash TEXT,
  status TEXT NOT NULL,
  retry_count INTEGER DEFAULT 0,
  max_retries INTEGER DEFAULT 3,
  priority INTEGER DEFAULT 100,
  dependency_ids TEXT,
  last_error_code TEXT,
  last_error_message TEXT,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(run_id) REFERENCES run_state(run_id)
);

CREATE TABLE artifact_state (
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  path TEXT NOT NULL,
  version INTEGER NOT NULL,
  checksum TEXT,
  source_refs TEXT,
  validation_status TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(run_id) REFERENCES run_state(run_id),
  FOREIGN KEY(task_id) REFERENCES task_state(task_id)
);
```

---

## 13) Acceptance Criteria
1. User prompt can produce folder tree and file scaffolding.
2. Each generated file has source_refs from retrieval evidence.
3. Validation failures trigger bounded reflection retries.
4. Run can be resumed after interruption from persistent state.
5. Final run report includes successful tasks, failed tasks, and evidence trace.
