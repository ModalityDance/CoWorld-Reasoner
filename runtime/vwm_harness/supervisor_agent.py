"""Stage-wise co-evolution with a persistent Codex supervisor."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from util.harness_state import (
    save_harness, editable_scope, validate_revision, install_source_file,
    remove_source_file, restore_harness, _source_paths,
)
from runtime.execution import _write
from runtime.model_config import shared_vlm_model, live_api_key, default_live_base_url
from runtime.vwm_harness.codex_session import CodexSession, ToolResponse
from runtime.vwm_harness.candidate_evaluation import candidate_record, validate_candidates
from runtime.vwm_harness.supervisor_feedback import comparison_rows, decision_comparison, sample_cases
from runtime.vwm_harness.experiment_context import prepare_candidate_context, case_evidence
from runtime.vwm_harness.stage_assessment import STAGE_INSTRUCTIONS
from runtime.vwm_harness.project_context import project_guide, project_feedback, trial_feedback


SUPERVISOR_ROLE = """You are the supervisor for CoWorld Reasoner harness co-evolution.
Improve the harness's reusable methods by learning from successful, borderline and unsuccessful executions.
Distill concrete observations into workflows, reasoning principles and implementation improvements
that help with new cases; use reruns to test how these methods improve task performance.
Correct executions can reveal reusable methods and opportunities for refinement too.
Explore the project with native Codex tools, read complete trajectories and view original
images and generated frames. Own the investigation, implementation and candidate comparison,
including coordinated changes, new modules and refactoring across the task harness.
Look across multiple samples before turning an observation into a revision: identify a repeated
mechanism, state the method that should transfer to new cases, and place it in the component
that owns that method. After each rerun, verify that changed guidance or implementation was
actually delivered and used in the execution evidence, not merely changed on disk.

Develop systematic workflows and reusable task-specific experience for similar tasks in the
corresponding skills. Keep stage instructions focused on cross-task general principles, stage
responsibilities and basic output requirements. Organize new and existing guidance by these
responsibilities, rather than following its current file location. Express learned methods
through roles, relations and procedures that transfer to new cases.
Tools define and execute visual operations;
the VLM verifier interprets generated frames only, with question and request text, while the main
agent compares with originals and answers. Execution/context code delivers ordered media,
guidance, tool results and continuing history. Use these roles to organize improvements and
follow their interactions through the implementation.

The co-evolution proceeds through P1 capability assessment, P2 visual assistance,
P3 video interpretation and P4 evidence integration. Each stage has its own goal.
Candidates within a stage share incoming histories; the selected code and histories
continue to the next stage. Each outer round starts fresh histories from the selected harness.
The task agent chooses its skills, operations and answers. Model settings, tool budgets,
orchestration, scoring and datasets are fixed run conditions.

experiment_state provides current state and project/evidence locations. The working project
and conversation persist across stages. Independent answer scores are available at P4.
After an execution failure, retry unchanged code for a service interruption, submit a repair
from a completed candidate, or select an earlier completed candidate. Failed revisions count
toward the stage budget; retrying the same revision does not.
"""


def _tool(name: str, description: str, properties: dict, *, optional: tuple = ()) -> dict:
    return {"type": "function", "name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": [key for key in properties if key not in optional],
                            "additionalProperties": False}}


TEXT = {"type": "string", "minLength": 1}
IDS = {"type": "array", "items": {"type": "string"}, "minItems": 1}
INSPECT_CASE_DESCRIPTION = """Locate the exact question, candidate outputs, complete readable
interaction history and all ordered media of one sample. Supply one candidate or two to compare. Returns
the skill guidance actually delivered, an ordered execution chain from history through tools,
generation and verifier, executed/current source locations, and history.md paths. Labeled original
and generated images are attached to the current turn in task/frame order.
"""
TOOLS = [
    _tool("experiment_state", "Locate experiment state, candidates, scores, media and editable scope.", {}),
    _tool("retry_execution", "Resume a pending execution after service recovery, reusing completed samples and request receipts. This creates no new revision and consumes no revision budget.", {}),
    _tool("inspect_case", INSPECT_CASE_DESCRIPTION, {
        "candidate_ids": {**IDS, "maxItems": 2}, "sample_id": TEXT}),
    _tool("run_revision", "Execute the working project's changes relative to the named completed base candidate on the full stage population from fixed incoming histories. May replace a failed pending revision with a repair. The hypothesis should state the repeated mechanism and transferable method. Returns paired results, source changes and execution evidence so delivery and use can be checked; consumes one revision cycle regardless of how many files changed.", {
        "base_candidate_id": TEXT, "hypothesis": TEXT,
        "target_case_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional cases to highlight in the comparison; every sample still reruns."},
        "expected_behavior": TEXT, "preserve_behavior": TEXT},
        optional=("target_case_ids", "expected_behavior", "preserve_behavior")),
    _tool("select_candidate", "Select a completed candidate and restore its code to the working project. Complete the stage goal after selection; the next stage then inherits this candidate's code and histories.", {
        "candidate_id": TEXT, "comparison": TEXT, "stop_reason": TEXT,
        "unresolved_issues": {"type": "array", "items": {"type": "string"}}},
        optional=("stop_reason", "unresolved_issues")),
]


class EvolutionExperiment:
    """Coordinate serial candidate execution and stage transitions."""

    def __init__(self, root: Path, output: Path, args):
        self.root, self.output, self.args = root, output, args
        self.path = output / "EXPERIMENT_STATE.json"
        if self.path.is_file():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            initial = output / "initial_snapshot"
            save_harness(root, initial)
            self.state = {"status": "running", "round": 1, "stage_index": 0,
                          "upstream": str(args.initial_upstream.resolve()) if args.initial_upstream else None,
                          "candidates": [], "selections": [], "pending": None,
                          "initial_snapshot": str(initial)}
            self.save()
        self.state.setdefault("failed_executions", [])

    @property
    def stage(self) -> str:
        return self.args.stages[self.state["stage_index"]]

    def save(self) -> None:
        _write(self.path, self.state)

    def current_candidates(self) -> list[dict]:
        return [c for c in self.state["candidates"]
                if c["evolution_round"] == self.state["round"] and c["stage"] == self.stage]

    def current_selection(self) -> dict | None:
        return next((s for s in self.state["selections"]
                     if s["round"] == self.state["round"] and s["stage"] == self.stage), None)

    def current_failures(self) -> list[dict]:
        return [item for item in self.state["failed_executions"]
                if item["evolution_round"] == self.state["round"] and item["stage"] == self.stage]

    def revision_count(self) -> int:
        revisions = [c["round_index"] for c in self.current_candidates()]
        revisions.extend(item["revision"] for item in self.current_failures())
        if self.state["pending"]:
            revisions.append(self.state["pending"]["revision"])
        return max(revisions, default=0)

    def _release_failed_execution(self) -> None:
        pending = self.state["pending"]
        if pending:
            if not pending.get("error"):
                raise ValueError("execution_pending:use_retry_execution_before_revising")
            self.state["failed_executions"].append(pending)
            self.state["pending"] = None

    def _new_pending(self, revision: int, source: str, experiment: dict | None = None,
                     changes: list[dict] | None = None) -> dict:
        cid = f"{self.args.candidate_id}-e{self.state['round']}-{self.stage.lower()}-r{revision}"
        return {"candidate_id": cid, "revision": revision, "stage": self.stage,
                "evolution_round": self.state["round"], "source": source,
                "upstream": self.state["upstream"], "experiment": experiment,
                "changes": changes or [], "candidate_dir": str(self.output /
                    f"round_{self.state['round']}" / "runs" / f"{self.stage.lower()}_r{revision}")}

    def ensure_initial(self, *, retry: bool = False) -> None:
        if self.state["status"] == "completed":
            return
        if self.state["pending"]:
            if self.state["pending"].get("error") and self.current_candidates() and not retry:
                return
            self.execute_pending()
        if not self.current_candidates():
            source = self.state["upstream"] or (
                self.state["selections"][-1]["candidate_dir"] if self.state["selections"]
                else self.state["initial_snapshot"])
            self.state["pending"] = self._new_pending(0, source)
            self.save()
            self.execute_pending()

    def execute_pending(self) -> None:
        from experiments.co_evolution import _restore_selected_harness, _stage_command, _scorer_command, _run_candidate_with_retries, _run
        pending = self.state["pending"]
        if not pending:
            return
        directory = Path(pending["candidate_dir"])
        _restore_selected_harness(self.root, Path(pending["source"]),
                                  self.output / "source_backups" / pending["candidate_id"])
        for change in pending["changes"]:
            if change["operation"] == "delete":
                remove_source_file(self.root / change["path"])
                continue
            staged = Path(change["staged_path"])
            if hashlib.sha256(staged.read_bytes()).hexdigest() != change["sha256"]:
                raise ValueError("submitted_revision_changed")
            install_source_file(staged, self.root / change["path"])
        command = _stage_command(self.root, pending["stage"], self.args.input_jsonl.resolve(),
            self.args.input_root.resolve(), directory, Path(pending["upstream"]) if pending["upstream"] else None,
            pending["candidate_id"], pending["revision"], self.args.workers)
        try:
            _run_candidate_with_retries(command, cwd=self.root, candidate_dir=directory,
                                       output_root=self.output, retries=self.args.stage_retries)
            if pending["stage"] == "P4" and self.args.evaluator_jsonl:
                _run(_scorer_command(self.root, directory, self.args.input_jsonl.resolve(),
                     self.args.evaluator_jsonl.resolve(), self.root / "scorer_bundle", self.args.timeout_sec), cwd=self.root)
            item, records = candidate_record(directory)
            item.update(evolution_round=pending["evolution_round"], experiment=pending["experiment"],
                        revision_diff=[{k: v for k, v in c.items() if k != "staged_path"} for c in pending["changes"]],
                        record_path=str(directory / "records.jsonl"))
            baselines = prepare_candidate_context(self.output, [item])
            item["baseline_hashes"] = {p: hashlib.sha256(b).hexdigest() for p, b in baselines[item["candidate_id"]].items()}
        except Exception as exc:
            pending["error"] = f"{type(exc).__name__}:{exc}"
            self.save()
            raise
        self.state["candidates"].append(item)
        self.state["working_base_candidate_id"] = item["candidate_id"]
        self.state["pending"] = None
        self.save()
        self.write_context()

    def write_context(self) -> Path:
        candidates = self.current_candidates()
        records = {}
        for c in candidates:
            records[c["candidate_id"]] = candidate_record(Path(c["candidate_dir"]))[1]
        latest = candidates[-1] if candidates else None
        experiment = latest.get("experiment") if latest else None
        base_id = experiment.get("base_candidate_id") if experiment else None
        reference = records.get(base_id, {})
        current = records.get(latest["candidate_id"], {}) if latest else {}
        selected, pools = sample_cases(current, reference, self.args.max_cases,
                                       (experiment or {}).get("target_case_ids", [])) if current else ([], {})
        comparison_path = self.output / "comparison.jsonl"
        comparisons = comparison_rows(current, reference)
        self.decision_changes = decision_comparison(comparisons)
        self.latest_trial = trial_feedback(latest, self.decision_changes) if latest else None
        comparison_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
            for row in comparisons), encoding="utf-8")
        history_path = self.output / "candidate_history.json"
        _write(history_path, [{key: c.get(key) for key in (
            "candidate_id", "evolution_round", "stage", "candidate_dir", "workspace", "harness_source_dir",
            "experiment", "metrics", "revision_diff", "population_path", "interaction_histories")}
            for c in self.state["candidates"]])
        context = {"status": self.state["status"], "evolution_round": self.state["round"],
            "python_executable": sys.executable,
            "stage_findings_path": str(self.output / "workspace" / "stage_findings.md"),
            "stage": self.stage, "criteria": STAGE_INSTRUCTIONS[self.stage],
            "remaining_revisions": self.args.max_revisions - self.revision_count(),
            "pending": self.state["pending"], "failed_executions": self.current_failures(),
            "upstream": self.state["upstream"], "edit_scope": editable_scope(),
            "candidates": [{k: c.get(k) for k in (
                "candidate_id", "workspace", "harness_source_dir", "metrics", "population_path", "record_path",
                "eligible_for_selection")} for c in candidates],
            "comparison_reference_id": base_id, "comparison_path": str(comparison_path),
            "suggested_cases": selected, "case_categories": pools,
            "selected_stages": self.state["selections"],
            "available_commands": {name: shutil.which(name) for name in ("git", "rg", "jq")},
            "experiment_history_path": str(history_path),
            "record_fields": {"scoring": "P4 independent result: status, correct, score; not present at P1-P3",
                              "images": "original and generated paths labeled by sample, generation and frame index",
                              "interaction_history_path": "complete visible interaction; image payloads extracted to local files"}}
        path = self.output / "experiment.json"
        _write(path, context)
        return path

    def describe(self) -> dict:
        candidates = self.current_candidates()
        revisions = max(0, len(candidates) - 1)
        return {"status": self.state["status"], "round": self.state["round"], "stage": self.stage,
                "stage_selection": self.current_selection(),
                "stage_feedback": STAGE_INSTRUCTIONS[self.stage],
                "workspace": str(self.output / "workspace" / "project"),
                "working_base_candidate_id": self.state.get("working_base_candidate_id"),
                "context_path": str(self.write_context()), "pending": self.state["pending"],
                "comparison_reference_id": (candidates[-1].get("experiment") or {}).get("base_candidate_id") if candidates else None,
                "decision_comparison": self.decision_changes,
                "trial_feedback": self.latest_trial,
                "project": project_feedback(self.output, Path(self.state["initial_snapshot"]) / "harness"),
                "selected_stage_findings": [{"stage": s["stage"], "round": s["round"],
                    "candidate_id": s["candidate_id"], "comparison": s.get("comparison"),
                    "unresolved_issues": s.get("unresolved_issues", [])} for s in self.state["selections"]],
                "python_executable": sys.executable,
                "comparison_path": str(self.output / "comparison.jsonl"),
                "experiment_history_path": str(self.output / "candidate_history.json"),
                "stage_findings_path": str(self.output / "workspace" / "stage_findings.md"),
                "completed_revisions": revisions, "max_revisions": self.args.max_revisions,
                "attempted_revisions": self.revision_count(),
                "remaining_revisions": self.args.max_revisions - self.revision_count(),
                "failed_executions": self.current_failures(),
                "candidates": [{"id": c["candidate_id"], "snapshot": c["harness_source_dir"],
                                "metrics": c["metrics"]} for c in self.current_candidates()]}

    @staticmethod
    def verify_snapshot(candidate: dict) -> None:
        source = Path(candidate["harness_source_dir"])
        expected = candidate["baseline_hashes"]
        actual = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in _source_paths(source)}
        if actual != expected:
            raise ValueError("executed_snapshot_modified:" + candidate["candidate_id"])

    def revise(self, arguments: dict) -> dict:
        if self.state["status"] != "running":
            raise ValueError("experiment_is_not_running")
        if self.current_selection():
            raise ValueError("stage_candidate_already_selected:complete_the_current_goal")
        if self.state["pending"] and not self.state["pending"].get("error"):
            raise ValueError("execution_pending:use_retry_execution_before_revising")
        candidates = self.current_candidates()
        base = next((c for c in candidates if c["candidate_id"] == arguments["base_candidate_id"]), None)
        if base is None:
            raise ValueError("revision_base_must_be_a_current_stage_candidate")
        self.verify_snapshot(base)
        if not set(arguments.get("target_case_ids", [])).issubset(base["sample_ids"]):
            raise ValueError("experiment_targets_not_in_population")
        workspace = Path(base["workspace"])
        changes = []
        files = {p.relative_to(workspace).as_posix(): p for p in _source_paths(workspace)}
        for relative in sorted(set(base["baseline_hashes"]) | set(files)):
            expected = base["baseline_hashes"].get(relative)
            content = files[relative].read_bytes() if relative in files else None
            digest = hashlib.sha256(content).hexdigest() if content is not None else None
            if digest == expected:
                continue
            validate_revision(workspace, relative, content)
            before = (Path(base["harness_source_dir"]) / relative).read_bytes() if expected is not None else b""
            try:
                diff = "".join(difflib.unified_diff(
                    before.decode("utf-8").splitlines(keepends=True),
                    (content or b"").decode("utf-8").splitlines(keepends=True),
                    fromfile="before/" + relative, tofile="after/" + relative))
            except UnicodeDecodeError:
                diff = "Binary content changed"
            changes.append({"path": relative, "sha256": digest, "diff": diff,
                            "operation": "delete" if content is None else "add" if expected is None else "modify"})
        if not changes:
            raise ValueError("no_effective_revision:edit_the_candidate_before_running")
        desired = dict(base["baseline_hashes"])
        for change in changes:
            if change["operation"] == "delete":
                desired.pop(change["path"])
            else:
                desired[change["path"]] = change["sha256"]
        for candidate in candidates:
            if candidate["baseline_hashes"] == desired:
                return {"already_executed_candidate_id": candidate["candidate_id"], **self.describe()}
        revision = self.revision_count() + 1
        if revision > self.args.max_revisions:
            raise ValueError("stage_revision_budget_exhausted")
        pending = self._new_pending(revision, base["candidate_dir"], arguments, changes)
        for change in changes:
            if change["operation"] == "delete":
                continue
            target = self.output / "submitted" / pending["candidate_id"] / change["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / change["path"], target)
            change["staged_path"] = str(target)
        self._release_failed_execution()
        self.state["pending"] = pending
        self.save()
        self.execute_pending()
        return self.describe()

    def select(self, arguments: dict) -> dict:
        if any(s["candidate_id"] == arguments["candidate_id"] for s in self.state["selections"]):
            return self.describe()
        if self.state["status"] != "running" or (
                self.state["pending"] and not self.state["pending"].get("error")):
            raise ValueError("finish_pending_execution_before_selection")
        if self.current_selection():
            raise ValueError("stage_candidate_already_selected:complete_the_current_goal")
        candidates = self.current_candidates()
        validate_candidates(candidates, self.stage)
        for candidate in candidates:
            self.verify_snapshot(candidate)
        selected = next((c for c in candidates if c["candidate_id"] == arguments["candidate_id"]), None)
        if not selected or not selected["eligible_for_selection"]:
            raise ValueError("select_an_eligible_completed_current_stage_candidate")
        for key in ("comparison",):
            if not str(arguments.get(key, "")).strip():
                raise ValueError(f"selection_requires_{key}")
        restore_harness(self.output / "workspace" / "project", Path(selected["candidate_dir"]),
                        self.output / "source_backups" / "project" / selected["candidate_id"], resume=True)
        self._release_failed_execution()
        self.state["working_base_candidate_id"] = selected["candidate_id"]
        selection = {**arguments, "stage": self.stage, "round": self.state["round"],
                     "candidate_dir": selected["candidate_dir"]}
        self.state["selections"].append(selection)
        self.save()
        return self.describe()

    def advance_stage(self, goal: dict) -> None:
        """Advance only after the selected stage's native goal has finished."""
        selected = self.current_selection()
        if not selected or goal.get("status") != "complete":
            raise RuntimeError("stage_requires_selected_candidate_and_completed_goal")
        candidate = next(c for c in self.current_candidates()
                         if c["candidate_id"] == selected["candidate_id"])
        self.verify_snapshot(candidate)
        restore_harness(self.output / "workspace" / "project", Path(selected["candidate_dir"]),
                        self.output / "source_backups" / "stage_completion" / selected["candidate_id"],
                        resume=True)
        selected["goal"] = goal
        self.state["upstream"] = selected["candidate_dir"]
        if self.state["stage_index"] + 1 < len(self.args.stages):
            self.state["stage_index"] += 1
        elif self.state["round"] < self.args.evolution_rounds:
            self.state["round"] += 1
            self.state["stage_index"] = 0
            self.state["upstream"] = None
        else:
            self.state["status"] = "completed"
            self.state["final_candidate_dir"] = selected["candidate_dir"]
            from experiments.co_evolution import _restore_selected_harness
            _restore_selected_harness(self.root, Path(selected["candidate_dir"]), self.output / "final_snapshot")
        self.save()

    def dispatch(self, name: str, arguments: dict) -> dict | ToolResponse:
        print(f"[supervisor] {name} ({self.stage}, round {self.state['round']})", flush=True)
        if name == "experiment_state":
            return self.describe()
        if name == "retry_execution":
            self.ensure_initial(retry=True)
            return self.describe()
        if name == "inspect_case":
            ids = arguments.get("candidate_ids") or []
            if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
                raise ValueError("inspect_one_or_two_distinct_candidates")
            candidates = {c["candidate_id"]: c for c in self.state["candidates"]}
            selected = [candidates[cid] for cid in ids]
            if any(arguments["sample_id"] not in c["sample_ids"] for c in selected):
                raise ValueError("inspection_sample_not_in_candidates")
            return ToolResponse(case_evidence(selected, arguments["sample_id"]))
        if name == "run_revision":
            return self.revise(arguments)
        if name == "select_candidate":
            return self.select(arguments)
        raise ValueError(f"unknown_experiment_operation:{name}")


def stage_objective(experiment: EvolutionExperiment) -> str:
    return (
        f"Outer round {experiment.state['round']}, {experiment.stage}.\n"
        + STAGE_INSTRUCTIONS[experiment.stage]
        + "\n\nLearn from executions and refine the harness's reusable methods for new cases. "
        "Compare multiple representative cases to identify repeated mechanisms and formulate methods "
        "that transfer to new cases. Put task-specific methods in skills and cross-task general principles "
        "in stage instructions. After rerun, verify delivery and use in execution evidence; compare "
        "capabilities, performance, and how the learned methods transfer and fit their components. "
        f"Budget: at most {experiment.args.max_revisions} complete revision cycles, each comprising "
        "a set of edits, a stage rerun and comparison. When no further promising improvement "
        "is supported by the evidence, or the budget is exhausted, select the best candidate, "
        "summarize the comparison and complete the goal."
    )


def run(root: Path, output: Path, args) -> dict:
    experiment = EvolutionExperiment(root, output, args)
    experiment.ensure_initial()
    session_path = output / "SUPERVISOR_SESSION.json"
    saved = json.loads(session_path.read_text(encoding="utf-8")) if session_path.is_file() else {}
    if experiment.state["status"] == "completed" and saved.get("goal_status") == "complete":
        _write(output / "RUN_RESULT.json", experiment.state)
        _write(output / "RUN_STATE.json", experiment.state)
        return experiment.state
    workspace = Path(saved.get("workspace") or experiment.current_candidates()[-1]["workspace"])
    with CodexSession(workspace, output / "supervisor", model=shared_vlm_model(),
                      base_url=os.environ.get("AGENTIC_VLM_BASE_URL") or default_live_base_url(),
                      api_key=live_api_key("AGENTIC_VLM_API_KEY", "AGENTIC_LLM_API_KEY", "OPENAI_API_KEY"),
                      timeout=args.timeout_sec, writable_roots=(output / "workspace",)) as codex:
        thread_id = codex.start(SUPERVISOR_ROLE + "\n\nPROJECT GUIDE\n" + project_guide(workspace),
                               TOOLS, thread_id=saved.get("thread_id"))
        def progress(event):
            if event.get("method") == "item/completed":
                item = event.get("params", {}).get("item", {})
                if item.get("type") == "agentMessage":
                    print("[supervisor] " + item.get("text", "")[:800], flush=True)
        while experiment.state["status"] != "completed":
            experiment.ensure_initial()
            scope = {"round": experiment.state["round"], "stage": experiment.stage}
            objective = stage_objective(experiment)
            goal = codex.get_goal()
            # Record the stage scope before starting the goal so a resumed session
            # continues with the same objective.
            same_goal = (saved.get("goal_scope") == scope and goal is not None
                         and goal.get("objective") == objective)
            saved = {"thread_id": thread_id, "workspace": str(workspace),
                     "goal_scope": scope, "goal_status": "starting"}
            _write(session_path, saved)
            if not same_goal:
                goal = codex.set_goal(objective)
            elif goal["status"] != "complete":
                goal = codex.set_goal(status="active")
            saved.update(goal_status=goal["status"], goal=goal)
            _write(session_path, saved)
            if goal["status"] != "complete":
                goal = codex.follow_goal(experiment.dispatch, progress)
            saved.update(goal_status=goal["status"], goal=goal)
            _write(session_path, saved)
            if goal["status"] != "complete":
                raise RuntimeError(f"supervisor_stage_goal_incomplete:{goal['status']}")
            experiment.advance_stage(goal)
    _write(output / "RUN_RESULT.json", experiment.state)
    _write(output / "RUN_STATE.json", experiment.state)
    return experiment.state
