"""Workspace self-monitoring update workflow.

Implements the algorithm from AlgoTodo_SelfMonitoring.md:
- retrieve a prior workspace snapshot by prompt similarity
- scan current workspace manifest
- compute structural diff
- select target files
- validate/apply explicit operations safely
- run optional checks
- persist new snapshot

This module is intentionally verbose and heavily commented so each step
is easy to follow during maintenance and debugging.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict
from uuid import uuid4

try:
    from agent.store_Db import search_workspace_structures, semantic_search, store_workspace_structure
except ModuleNotFoundError:
    from store_Db import search_workspace_structures, semantic_search, store_workspace_structure


# Supported operation kinds produced by the planning stage.
# String-backed Enum gives C/clang-like named constants with JSON-friendly values.
class OperationType(str, Enum):
    MKDIR = "mkdir"
    ADD_FILE = "add_file"
    REMOVE_FILE = "remove_file"
    PATCH_FILE = "patch_file"
    MOVE_FILE = "move_file"


def _normalize_operation_type(raw: Any) -> OperationType:
    """Convert raw op type value into OperationType enum."""
    if isinstance(raw, OperationType):
        return raw
    try:
        return OperationType(str(raw))
    except ValueError as exc:
        raise ValueError(f"Unsupported operation type: {raw}") from exc


class FileManifestEntry(TypedDict):
    # Stable content hash used to detect file changes.
    hash: str
    # File size in bytes.
    size: int
    # Last modification time in UTC ISO format.
    modified_at: str
    # Lightweight language hint inferred from file extension.
    language: str


class WorkspaceSnapshot(TypedDict, total=False):
    # Unique snapshot identifier.
    snapshot_id: str
    # Prompt that produced this snapshot.
    prompt: str
    # Snapshot creation time.
    timestamp: str
    # Top-level workspace folders.
    root_folders: List[str]
    # Manifest captured for this snapshot.
    file_manifest: Dict[str, FileManifestEntry]
    # Optional free-form summary.
    summary: str


class PatchHunk(TypedDict):
    # Context lines expected in file before applying this hunk.
    before_context: List[str]
    # Existing lines to remove.
    remove_lines: List[str]
    # New lines to add.
    add_lines: List[str]


class Operation(TypedDict, total=False):
    # Operation discriminator.
    type: OperationType | str
    # Target path for mkdir/add/remove/patch.
    path: str
    # Full content for add_file.
    content: str
    # Patch hunks for patch_file.
    hunks: List[PatchHunk]
    # Source path for move_file.
    from_path: str
    # Destination path for move_file.
    to_path: str
    # Optional planner rationale.
    reason: str


class ChangePlan(TypedDict, total=False):
    # Request id tied to the current workflow run.
    request_id: str
    # Snapshot id used as baseline for planning.
    snapshot_base_id: str
    # User prompt for this plan.
    prompt: str
    # Ordered/validated operations.
    operations: List[Operation]
    # Validation status: pending/passed/failed.
    validation_status: str


class DiffResult(TypedDict):
    # Paths present now but absent in baseline.
    added_paths: List[str]
    # Paths present in baseline but absent now.
    removed_paths: List[str]
    # Paths present in both but hash-changed.
    modified_paths: List[str]
    # Paths present in both and hash-identical.
    unchanged_paths: List[str]


class ApplySummary(TypedDict):
    # Operations successfully applied.
    applied: List[Operation]
    # Failed operation entries with error metadata.
    failed: List[Dict[str, Any]]


class WorkflowResult(TypedDict, total=False):
    # Overall status: ok/failed.
    status: str
    # Run-level request id.
    request_id: str
    # Selected base snapshot id.
    base_snapshot_id: str
    # Newly persisted snapshot id on success.
    new_snapshot_id: str
    # Files selected for possible edits.
    targets: List[str]
    # Structural diff from base->current.
    diff: DiffResult
    # Final plan used for apply.
    plan: ChangePlan
    # Apply stage outcome.
    apply_summary: ApplySummary
    # Optional check result payload.
    check_result: Dict[str, Any]
    # Error reason when failed.
    reason: str


@dataclass
class CheckResult:
    # Whether checks passed.
    ok: bool
    # Optional failure explanation.
    reason: str = ""


PlannerFn = Callable[[Dict[str, Any]], List[Operation]]
ChecksFn = Callable[[Path], CheckResult]

# Example planner callback signature:
# def my_planner(context: Dict[str, Any]) -> List[Operation]:
#     # context keys include: request_id, prompt, workspace_root, base_snapshot, diff, targets
#     return [
#         {
#             "type": OperationType.ADD_FILE,
#             "path": "notes/todo.txt",
#             "content": "Generated from planner callback\n",
#             "reason": "Create a starter note file",
#         }
#     ]
#
# Example checks callback signature:
# def my_checks(root: Path) -> CheckResult:
#     # Return ok=False with a reason to fail the workflow after apply phase.
#     required = root / "notes" / "todo.txt"
#     if required.exists() and required.is_file():
#         return CheckResult(ok=True, reason="Required file exists")
#     return CheckResult(ok=False, reason="Required file missing: notes/todo.txt")


def _now_iso() -> str:
    # Always store timestamps in UTC for consistent cross-machine comparison.
    return datetime.now(timezone.utc).isoformat()


def _guess_language(file_path: str) -> str:
    # Infer language from extension to enrich manifest metadata.
    suffix = Path(file_path).suffix.lower()
    mapping = {
        ".py": "python",
        ".c": "c",
        ".h": "c-header",
        ".md": "markdown",
        ".txt": "text",
        ".json": "json",
        ".toml": "toml",
        ".yml": "yaml",
        ".yaml": "yaml",
    }
    return mapping.get(suffix, "text")


def _hash_file(path: Path) -> str:
    # Stream file bytes to avoid loading very large files entirely in memory.
    # Parameter 'path: Path' - passed to get the file location we need to hash for content comparison.
    digest = hashlib.sha256()
    # .open("rb") - opens file in BINARY READ mode (rb) to read raw bytes, needed for hashing any file type.
    # Example: Path("/workspace/file.txt").open("rb") returns file handle for binary read.
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_workspace_files(workspace_root: Path) -> Iterable[Path]:
    # Ignore generated/cache directories to keep snapshots focused and stable.
    ignored_dirs = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "data/qdrant/collection",
    }

    for root, dirs, files in os.walk(workspace_root):
        # Convert walk root into workspace-relative form for ignore checks.
        rel_root = Path(root).relative_to(workspace_root).as_posix()
        dirs[:] = [
            d
            for d in dirs
            if not _is_ignored_dir((Path(rel_root) / d).as_posix(), ignored_dirs)
        ]
        for filename in files:
            # Yield absolute file path; caller converts to relative path as needed.
            yield Path(root) / filename


def _is_ignored_dir(relative_dir: str, ignored_dirs: set[str]) -> bool:
    # Normalize ./ prefixes that may appear from os.walk conversions.
    normalized = relative_dir.strip("./")
    if not normalized:
        return False
    return any(
        normalized == ignored or normalized.startswith(f"{ignored}/")
        for ignored in ignored_dirs
    )


def scan_workspace_manifest(workspace_root: str | Path) -> Dict[str, FileManifestEntry]:
    """Traverse workspace and build a file manifest."""
    # Parameter 'workspace_root: str | Path' - passed as either string path or Path object for flexibility.
    # Accepts both types so callers don't need to convert manually before calling.
    # .resolve() - converts any relative path to absolute path and resolves symlinks.
    # Example: Path("./project").resolve() -> Path("/home/user/project") if symlinks exist, they're followed.
    # This ensures consistent path representation across different working directories.
    root = Path(workspace_root).resolve()
    # Manifest key is POSIX-style relative path for portability.
    manifest: Dict[str, FileManifestEntry] = {}

    for file_path in _iter_workspace_files(root):
        # Keep keys stable across OS path separators.
        # .relative_to(root) - converts absolute path to relative path from root.
        # Example: Path("/home/user/project/src/main.py").relative_to(Path("/home/user/project")) -> Path("src/main.py").
        # .as_posix() - converts path to POSIX format (forward slashes) for portability across Windows/Linux/Mac.
        # Example: Path("src\\main.py").as_posix() -> "src/main.py".
        rel_path = file_path.relative_to(root).as_posix()
        # .stat() - retrieves file metadata (size, timestamps, permissions) without reading content.
        # Returns: os.stat_result object with st_size, st_mtime, etc. Much cheaper than reading file.
        stat = file_path.stat()
        manifest[rel_path] = {
            "hash": _hash_file(file_path),
            "size": int(stat.st_size),  # stat.st_size is already int but explicit cast for clarity.
            # .fromtimestamp(stat.st_mtime, tz=timezone.utc) - converts Unix timestamp to UTC datetime.
            # stat.st_mtime is seconds since epoch as float. tz=timezone.utc ensures UTC normalization.
            # Example: 1715161600.0 -> datetime(2024, 5, 8, 12, 0, 0, tzinfo=timezone.utc).
            # .isoformat() - converts datetime to ISO 8601 string for JSON serialization.
            # Example: datetime(2024, 5, 8, 12, 0, 0) -> "2024-05-08T12:00:00+00:00".
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "language": _guess_language(rel_path),
        }

    return manifest


def compare_manifests(
    base_manifest: Dict[str, FileManifestEntry],
    current_manifest: Dict[str, FileManifestEntry],
) -> DiffResult:
    """Compute structural diff from base to current manifest."""
    # Parameters passed to enable comparison:
    # 'base_manifest' - previous snapshot manifest to compare FROM.
    # 'current_manifest' - current workspace manifest to compare TO.
    # Passed separately so function is pure and doesn't modify input state.
    # Convert to sets for efficient O(1) membership and diff operations.
    # set(dict) extracts keys only, enabling fast set operations.
    # Example: set({"a.txt": {...}, "b.txt": {...}}) -> {"a.txt", "b.txt"}.
    base_paths = set(base_manifest)
    current_paths = set(current_manifest)

    # Set difference operations:
    # (current_paths - base_paths) - files in current that weren't in base = newly added.
    # (base_paths - current_paths) - files in base that aren't in current = deleted.
    # sorted() ensures deterministic ordering for consistent diffs across runs.
    added_paths = sorted(current_paths - base_paths)
    removed_paths = sorted(base_paths - current_paths)

    modified_paths: List[str] = []
    unchanged_paths: List[str] = []
    # (base_paths & current_paths) - set intersection finds files present in BOTH manifests.
    # sorted() ensures deterministic iteration order.
    for path in sorted(base_paths & current_paths):
        # Compare content hashes to detect if file changed since last snapshot.
        # Hash mismatch indicates file was modified even if size/timestamp changed.
        if base_manifest[path]["hash"] == current_manifest[path]["hash"]:
            unchanged_paths.append(path)
        else:
            modified_paths.append(path)

    return {
        "added_paths": added_paths,
        "removed_paths": removed_paths,
        "modified_paths": modified_paths,
        "unchanged_paths": unchanged_paths,
    }


def retrieve_base_snapshot(user_prompt: str, similarity_threshold: float = 0.3) -> WorkspaceSnapshot | None:
    """Retrieve best workspace snapshot by prompt similarity."""
    # Query top few candidates and accept first one above confidence threshold.
    matches = search_workspace_structures(query=user_prompt, top_k=3)
    for hit in matches:
        score = float(hit.get("score", 0.0))
        if score < similarity_threshold:
            continue

        structure = hit.get("structure", {})
        if not isinstance(structure, dict):
            continue

        manifest = structure.get("file_manifest")
        if not isinstance(manifest, dict):
            continue

        snapshot: WorkspaceSnapshot = {
            "snapshot_id": str(structure.get("snapshot_id", "")),
            "prompt": str(structure.get("prompt", "")),
            "timestamp": str(structure.get("timestamp", "")),
            "root_folders": list(structure.get("root_folders", [])),
            "file_manifest": manifest,
            "summary": str(structure.get("summary", "")),
        }
        return snapshot

    return None


def select_target_files(user_prompt: str, diff: DiffResult, top_k: int = 8) -> List[str]:
    """Select likely target files using diff signals plus semantic retrieval metadata."""
    # Start with deterministic structural signals.
    targets = set(diff["modified_paths"] + diff["added_paths"])

    try:
        # Add semantically related files from document retrieval metadata.
        retrieved = semantic_search(user_prompt, top_k=top_k)
        for row in retrieved:
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                source = metadata.get("source")
                if isinstance(source, str) and source:
                    targets.add(source.replace("\\", "/"))
    except Exception:
        # Retrieval is best-effort and should not block the workflow.
        pass

    return sorted(targets)


def default_plan_builder(context: Dict[str, Any]) -> List[Operation]:
    """Default planner returning no-op plan when no external planner is provided."""
    # Context is unused in no-op mode but kept for signature compatibility.
    _ = context
    return []


def validate_operation_plan(plan: ChangePlan, workspace_root: Path) -> None:
    """Validate operation contract with safety checks."""
    # Parameters:
    # 'plan: ChangePlan' - the operation plan to validate before execution.
    # 'workspace_root: Path' - the boundary we must not escape (prevents ../../../etc/passwd attacks).
    # Passed together so we can validate all operations stay within root.
    # .resolve() converts to absolute path and follows symlinks to get true location.
    # Prevents trick where symlink-based relative paths escape workspace.
    root_resolved = workspace_root.resolve()

    for op in plan.get("operations", []):
        op_type = _normalize_operation_type(op.get("type", ""))

        if op_type in {
            OperationType.MKDIR,
            OperationType.ADD_FILE,
            OperationType.REMOVE_FILE,
            OperationType.PATCH_FILE,
        }:
            path = str(op.get("path", ""))
            _assert_path_in_workspace(path, root_resolved)

        if op_type == OperationType.MOVE_FILE:
            from_path = str(op.get("from_path", ""))
            to_path = str(op.get("to_path", ""))
            _assert_path_in_workspace(from_path, root_resolved)
            _assert_path_in_workspace(to_path, root_resolved)

        if op_type == OperationType.PATCH_FILE:
            hunks = op.get("hunks", [])
            if not isinstance(hunks, list) or not hunks:
                raise ValueError("patch_file operation requires non-empty hunks")
            for hunk in hunks:
                if not isinstance(hunk, dict):
                    raise ValueError("Patch hunk must be an object")
                before_context = hunk.get("before_context", [])
                if not isinstance(before_context, list) or not before_context:
                    raise ValueError("Patch hunk must include non-empty before_context")


def _assert_path_in_workspace(relative_path: str, workspace_root: Path) -> None:
    # Empty path is always invalid.
    if not relative_path:
        raise ValueError("Operation path cannot be empty")
    # Resolve candidate path and verify it is under workspace root.
    target = (workspace_root / relative_path).resolve()
    if workspace_root not in target.parents and target != workspace_root:
        raise ValueError(f"Path escapes workspace root: {relative_path}")


def apply_operations(plan: ChangePlan, workspace_root: str | Path) -> ApplySummary:
    """Apply operations in safe order with context-aware patching."""
    # Resolve root to avoid cwd-dependent behavior.
    root = Path(workspace_root).resolve()
    operations = plan.get("operations", [])

    # Enforce deterministic execution order for safety and predictable effects.
    ordered_types = [
        OperationType.MKDIR,
        OperationType.ADD_FILE,
        OperationType.PATCH_FILE,
        OperationType.REMOVE_FILE,
        OperationType.MOVE_FILE,
    ]
    ordered_ops: List[Operation] = []
    for op_type in ordered_types:
        ordered_ops.extend(
            op for op in operations if _normalize_operation_type(op.get("type", "")) == op_type
        )

    applied: List[Operation] = []
    failed: List[Dict[str, Any]] = []

    for op in ordered_ops:
        try:
            # Apply one operation and accumulate successful entries.
            _apply_single_operation(op, root)
            applied.append(op)
        except Exception as exc:
            # Continue applying remaining ops while recording per-op failures.
            failed.append({"operation": op, "error": str(exc)})

    return {"applied": applied, "failed": failed}


def _apply_single_operation(op: Operation, workspace_root: Path) -> None:
    # Parameters:
    # 'op: Operation' - the single file-system operation to perform (mkdir, add_file, patch, etc.)
    # 'workspace_root: Path' - the root directory where all operations execute within.
    # Passed as separate parameters to isolate each operation execution from others.
    # Dispatch to concrete file system action by operation type.
    op_type = _normalize_operation_type(op["type"])

    if op_type == OperationType.MKDIR:
        path = workspace_root / op["path"]
        # .mkdir(parents=True, exist_ok=True):
        # parents=True - create parent directories if they don't exist (like mkdir -p).
        # Example: Path("/a/b/c").mkdir(parents=True) creates /a, /a/b, /a/b/c as needed.
        # exist_ok=True - don't raise error if directory already exists (idempotent).
        # This allows safe re-execution without failure if directory already present.
        path.mkdir(parents=True, exist_ok=True)
        return

    if op_type == OperationType.ADD_FILE:
        path = workspace_root / op["path"]
        # .parent - returns Path object for the directory containing this file.
        # Example: Path("/a/b/c.txt").parent -> Path("/a/b").
        # Ensure parent tree exists before writing file.
        path.parent.mkdir(parents=True, exist_ok=True)
        # .write_text(content, encoding="utf-8") - writes string content to file with UTF-8 encoding.
        # Automatically opens, writes, and closes file in one call.
        # encoding="utf-8" ensures consistent text representation for cross-platform compatibility.
        # op.get("content", "") - retrieves content from operation dict, defaults to empty string if missing.
        path.write_text(op.get("content", ""), encoding="utf-8")
        return

    if op_type == OperationType.REMOVE_FILE:
        path = workspace_root / op["path"]
        # .exists() - checks if path exists (file or directory), returns boolean.
        # .is_file() - checks if path is a regular file (not directory or symlink), returns boolean.
        # Both checks prevent accidentally unlinking directories or non-existent paths.
        # Example: Path("/a/b.txt").exists() -> True if file exists, False if deleted.
        # Remove only regular files; silently ignore missing files.
        if path.exists() and path.is_file():
            # .unlink() - deletes the file (like rm command).
            # Raises FileNotFoundError if path doesn't exist, hence the guard checks above.
            path.unlink()
        return

    if op_type == OperationType.MOVE_FILE:
        from_path = workspace_root / op["from_path"]
        to_path = workspace_root / op["to_path"]
        # Fail fast when source is absent.
        if not from_path.exists():
            raise FileNotFoundError(f"Source file not found: {op['from_path']}")
        # Destination parent may not exist yet.
        to_path.parent.mkdir(parents=True, exist_ok=True)
        # .replace(target) - moves file from one location to another (like mv command).
        # Overwrites target if it exists, and works across filesystems.
        # Example: Path("/a/file.txt").replace(Path("/b/file.txt")) moves file from /a to /b.
        from_path.replace(to_path)
        return

    if op_type == OperationType.PATCH_FILE:
        target_path = workspace_root / op["path"]
        # Patching requires existing target file.
        if not target_path.exists():
            raise FileNotFoundError(f"Patch target file not found: {op['path']}")
        # .read_text(encoding="utf-8") - reads entire file content as string with UTF-8 decoding.
        # Returns string which we'll modify with patch hunks.
        # Example: Path("/file.txt").read_text() -> "line 1\nline 2\n".
        original = target_path.read_text(encoding="utf-8")
        patched = _apply_patch_hunks(original, op.get("hunks", []))
        # .write_text(patched, encoding="utf-8") - replaces entire file content with patched version.
        # Ensures atomic write of modified content back to file.
        target_path.write_text(patched, encoding="utf-8")
        return

    raise ValueError(f"Unsupported operation type: {op_type}")


def _apply_patch_hunks(content: str, hunks: List[PatchHunk]) -> str:
    # Parameters:
    # 'content: str' - the original file content to patch (immutable string).
    # 'hunks: List[PatchHunk]' - list of patch hunks to apply sequentially.
    # Passed as separate params to keep pure function: input content not modified, new patched version returned.
    # Apply hunks incrementally; each hunk sees content updates from prior hunks.
    updated = content
    for hunk in hunks:
        # Convert line arrays into newline-joined text blocks.
        # .join(list) - concatenates list of strings with separator ("\n" creates line-by-line text).
        # Example: "\n".join(["line 1", "line 2"]) -> "line 1\nline 2".
        # .get(key, default) - retrieves hunk field with default empty list if missing.
        before_context = "\n".join(hunk.get("before_context", []))
        remove_block = "\n".join(hunk.get("remove_lines", []))
        add_block = "\n".join(hunk.get("add_lines", []))

        if not before_context:
            raise ValueError("Patch hunk missing before_context")

        # .find(substring) - returns index of first occurrence of substring, or -1 if not found.
        # Example: "hello world".find("world") -> 6, "hello world".find("xyz") -> -1.
        idx = updated.find(before_context)
        if idx < 0:
            raise ValueError("Patch context not found in target file")

        replace_anchor = before_context
        if remove_block:
            # Replacement mode: remove matched block and add new block.
            # .find(substring, start_index) - searches for substring starting from start_index.
            # Example: "hello world".find("o", 5) -> 7 (finds second 'o' after position 5).
            remove_idx = updated.find(remove_block, idx)
            if remove_idx < 0:
                raise ValueError("Patch remove_lines block not found after context")
            start = remove_idx
            # len(remove_block) - gets string length to know where removed block ends.
            end = remove_idx + len(remove_block)
            # String slicing: updated[:start] + add_block + updated[end:]
            # Removes [start:end] range and inserts add_block in place.
            # Example: "abc def ghi"[:4] + "XXX" + "abc def ghi"[7:] -> "abc XXXghi".
            updated = updated[:start] + add_block + updated[end:]
        else:
            # Insertion mode: append new block right after context anchor.
            insert_at = idx + len(replace_anchor)
            prefix = updated[:insert_at]  # Everything up to and including context.
            suffix = updated[insert_at:]  # Everything after context.
            # .endswith(substring) - returns True if string ends with substring.
            # Example: "hello\n".endswith("\n") -> True, "hello".endswith("\n") -> False.
            # Add newline glue if add_block exists but prefix doesn't end with newline (formatting).
            glue = "\n" if (add_block and not prefix.endswith("\n")) else ""
            updated = prefix + glue + add_block + suffix

    return updated


def persist_snapshot(
    prompt: str,
    workspace_root: str | Path,
    manifest: Dict[str, FileManifestEntry],
    summary: str,
) -> str:
    """Persist latest snapshot into the workspace_structures collection."""
    # Parameters passed for snapshot creation:
    # 'prompt: str' - the user request that led to this workspace state.
    # 'workspace_root: str | Path' - the workspace being captured (converted to absolute path).
    # 'manifest: Dict[str, FileManifestEntry]' - the file hash/metadata map for this state.
    # 'summary: str' - human-readable summary of what changed in this update.
    # All passed together to fully capture workspace state context.
    # Capture snapshot payload in schema expected by workspace_structures DB.
    root = Path(workspace_root).resolve()
    snapshot: WorkspaceSnapshot = {
        # .uuid4() - generates a random UUID (Universally Unique IDentifier).
        # Returns 128-bit random value as UUID object. str() converts to standard UUID string format.
        # Example: uuid.uuid4() -> UUID('550e8400-e29b-41d4-a716-446655440000')
        # str(uuid.uuid4()) -> '550e8400-e29b-41d4-a716-446655440000'.
        "snapshot_id": str(uuid4()),
        "prompt": prompt,
        "timestamp": _now_iso(),
        "root_folders": [root.name],
        "file_manifest": manifest,
        "summary": summary,
    }
    store_workspace_structure(snapshot, request_text=prompt)
    # Return local generated id for caller tracking.
    return snapshot["snapshot_id"]


def self_monitoring_update(
    user_prompt: str,
    workspace_root: str | Path,
    planner: Optional[PlannerFn] = None,
    checks: Optional[ChecksFn] = None,
    similarity_threshold: float = 0.3,
) -> WorkflowResult:
    """Run the workspace-aware minimal-change update workflow.

    The planner callback must return explicit operation objects matching the
    contract in this file. If planner is not supplied, a no-op plan is used.
    """
    # Unique execution id for correlating logs, plan, and snapshot summary.
    request_id = str(uuid4())
    root = Path(workspace_root).resolve()

    # Step 1-3: retrieve baseline, scan, and compute structural diff.
    base_snapshot = retrieve_base_snapshot(user_prompt, similarity_threshold=similarity_threshold)
    current_manifest = scan_workspace_manifest(root)

    base_manifest = base_snapshot.get("file_manifest", {}) if base_snapshot else {}
    diff = compare_manifests(base_manifest, current_manifest)
    # Step 4: choose candidate files for focused updates.
    targets = select_target_files(user_prompt, diff)

    # Build planner context so external planner can decide operations.
    planning_context: Dict[str, Any] = {
        "request_id": request_id,
        "prompt": user_prompt,
        "workspace_root": root.as_posix(),
        "base_snapshot": base_snapshot,
        "diff": diff,
        "targets": targets,
    }

    # Step 5: generate operation plan.
    plan_builder = planner or default_plan_builder
    operations = plan_builder(planning_context)

    plan: ChangePlan = {
        "request_id": request_id,
        "snapshot_base_id": base_snapshot.get("snapshot_id", "") if base_snapshot else "",
        "prompt": user_prompt,
        "operations": operations,
        "validation_status": "pending",
    }

    # Step 6: validate operation plan before touching disk.
    validate_operation_plan(plan, root)
    plan["validation_status"] = "passed"

    # Step 7: apply operations in safe order.
    apply_summary = apply_operations(plan, root)
    if apply_summary["failed"]:
        return {
            "status": "failed",
            "request_id": request_id,
            "base_snapshot_id": plan["snapshot_base_id"],
            "targets": targets,
            "diff": diff,
            "plan": plan,
            "apply_summary": apply_summary,
            "reason": "One or more operations failed during apply phase",
        }

    # Step 8: run optional post-apply checks.
    check_result = checks(root) if checks else CheckResult(ok=True, reason="no checks configured")
    if not check_result.ok:
        return {
            "status": "failed",
            "request_id": request_id,
            "base_snapshot_id": plan["snapshot_base_id"],
            "targets": targets,
            "diff": diff,
            "plan": plan,
            "apply_summary": apply_summary,
            "check_result": {"ok": check_result.ok, "reason": check_result.reason},
            "reason": check_result.reason or "Validation checks failed",
        }

    # Step 9: persist new snapshot after successful apply/check.
    refreshed_manifest = scan_workspace_manifest(root)
    new_snapshot_id = persist_snapshot(
        prompt=user_prompt,
        workspace_root=root,
        manifest=refreshed_manifest,
        summary=(
            f"request={request_id} base_snapshot={plan['snapshot_base_id']} "
            f"ops={len(plan.get('operations', []))} targets={len(targets)}"
        ),
    )

    return {
        "status": "ok",
        "request_id": request_id,
        "base_snapshot_id": plan["snapshot_base_id"],
        "new_snapshot_id": new_snapshot_id,
        "targets": targets,
        "diff": diff,
        "plan": plan,
        "apply_summary": apply_summary,
        "check_result": {"ok": check_result.ok, "reason": check_result.reason},
    }


def plan_from_json(plan_json: str) -> List[Operation]:
    """Helper to convert a JSON operation payload into typed operations."""
    # Accept either {"operations": [...]} or direct list payload.
    data = json.loads(plan_json)
    ops = data.get("operations", data)
    if not isinstance(ops, list):
        raise ValueError("Expected list of operations in JSON plan")
    typed_ops: List[Operation] = []
    for op in ops:
        if not isinstance(op, dict):
            raise ValueError("Each operation must be an object")
        typed_ops.append(op)
    return typed_ops


class WorkspaceAgent:
    """Convenience wrapper exposing the workflow as a named agent object."""

    # Canonical workflow name requested by user.
    name = "workspace_agent"

    def run(
        self,
        user_prompt: str,
        workspace_root: str | Path,
        planner: Optional[PlannerFn] = None,
        checks: Optional[ChecksFn] = None,
        similarity_threshold: float = 0.3,
    ) -> WorkflowResult:
        return self_monitoring_update(
            user_prompt=user_prompt,
            workspace_root=workspace_root,
            planner=planner,
            checks=checks,
            similarity_threshold=similarity_threshold,
        )


def demo_workspace_agent_create_then_delete_in_testing(
    workspace_root: str | Path = ".",
) -> Dict[str, WorkflowResult]:
    """Demo helper: create a file in testing workspace, then delete it.

    This function creates/uses a sub-workspace named "testing" under the
    provided workspace_root and runs WorkspaceAgent twice:
    1) add_file operation for a demo file
    2) remove_file operation for the same file
    """
    root = Path(workspace_root).resolve()
    testing_root = root / "testing"
    testing_root.mkdir(parents=True, exist_ok=True)

    agent = WorkspaceAgent()
    demo_rel_path = "workspace_agent_demo.txt"

    def create_planner(_: Dict[str, Any]) -> List[Operation]:
        return [
            {
                "type": OperationType.ADD_FILE,
                "path": demo_rel_path,
                "content": "This file was created by WorkspaceAgent demo.\n",
                "reason": "Demo create in testing workspace",
            }
        ]

    def delete_planner(_: Dict[str, Any]) -> List[Operation]:
        return [
            {
                "type": OperationType.REMOVE_FILE,
                "path": demo_rel_path,
                "reason": "Demo cleanup in testing workspace",
            }
        ]

    create_result = agent.run(
        user_prompt="Demo: create a file in testing workspace",
        workspace_root=testing_root,
        planner=create_planner,
    )

    if create_result.get("status") != "ok":
        return {"create": create_result}

    delete_result = agent.run(
        user_prompt="Demo: delete the created file in testing workspace",
        workspace_root=testing_root,
        planner=delete_planner,
    )

    return {
        "create": create_result,
        "delete": delete_result,
    }
