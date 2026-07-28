"""Generic prompting and exact scoring for compiled constraint workloads."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


COMPILED_WORKLOAD_SCHEMA_VERSION = 4
COMPILED_WORKLOAD_KIND = "compiled_keyword_dataset"
COMPILED_CONSTRAINT_SCHEMA_VERSION = 2
COMPILED_CONSTRAINT_KIND = "compiled_token_partition_nfa"
LENGTH_CONTRACT_SCHEMA_VERSION = 1
EXECUTION_SELECTION_SCHEMA_VERSION = 1


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _integer_list(
    value: Any,
    name: str,
    *,
    nonempty: bool = False,
    nonnegative: bool = False,
) -> List[int]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}integer list")
    result = [_integer(item, f"{name} item") for item in value]
    if nonnegative and any(item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    return result


def terminal_eos_length_contract(n_low: int, n: int) -> Dict[str, Any]:
    """Return the method-neutral content/terminal length contract."""

    if not 1 <= n_low <= n:
        raise ValueError("compiled constraint length bounds are invalid")
    return {
        "schema_version": LENGTH_CONTRACT_SCHEMA_VERSION,
        "content_token_interval": [n_low - 1, n - 1],
        "total_token_interval_including_eos": [n_low, n],
        "terminal_eos_tokens": 1,
        "eos_counts_toward_total": True,
        "eos_counts_toward_content": False,
    }


def _execution_rows(
    workload: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Authenticate and apply an optional run-specific workload selection."""

    selection = workload.get("execution_selection")
    if selection is None:
        return list(jobs)
    if not isinstance(selection, Mapping):
        raise ValueError("workload execution_selection must be an object")
    body = dict(selection)
    digest = body.pop("sha256", None)
    if digest != sha256_value(body):
        raise ValueError("workload execution_selection digest mismatch")
    replicate_indices = body.get("replicate_indices")
    if replicate_indices is not None:
        replicate_indices = _integer_list(
            replicate_indices,
            "execution_selection replicate_indices",
            nonempty=True,
            nonnegative=True,
        )
        if len(replicate_indices) != len(set(replicate_indices)):
            raise ValueError(
                "execution_selection replicate_indices must be distinct"
            )
    max_jobs = body.get("max_jobs")
    if max_jobs is not None:
        max_jobs = _integer(max_jobs, "execution_selection max_jobs")
        if max_jobs <= 0:
            raise ValueError("execution_selection max_jobs must be positive")
    selected = list(jobs)
    if replicate_indices is not None:
        selected = [
            row
            for row in selected
            if row.get("replicate_index") in set(replicate_indices)
        ]
    after_replicate_filter = len(selected)
    if max_jobs is not None:
        selected = selected[:max_jobs]
    selected_ids = [str(row["job_id"]) for row in selected]
    expected = {
        "schema_version": EXECUTION_SELECTION_SCHEMA_VERSION,
        "selection_order": "canonical_workload_order",
        "operation_order": ["replicate_filter", "max_jobs_prefix"],
        "replicate_indices": replicate_indices,
        "max_jobs": max_jobs,
        "source_instances": len(jobs),
        "instances_after_replicate_filter": after_replicate_filter,
        "selected_instances": len(selected),
        "selected_family_counts": dict(
            Counter(str(row["constraint"]) for row in selected)
        ),
        "selected_job_ids": selected_ids,
        "selected_job_ids_sha256": sha256_value(selected_ids),
    }
    if body != expected:
        raise ValueError("workload execution_selection differs from its jobs")
    if not selected:
        raise ValueError("workload execution_selection contains no jobs")
    return selected


def _validate_tokenizer_fingerprint(
    fingerprint: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if not isinstance(fingerprint, Mapping):
        raise ValueError(f"{name} must be an object")
    vocab_size = _integer(fingerprint.get("vocab_size"), f"{name} vocab_size")
    if vocab_size < 1:
        raise ValueError(f"{name} vocab_size must be positive")
    for field in ("eos_token_id", "pad_token_id"):
        value = fingerprint.get(field)
        if value is not None:
            _integer(value, f"{name} {field}")
    _integer_list(
        fingerprint.get("all_special_ids"),
        f"{name} all_special_ids",
        nonnegative=True,
    )
    files = fingerprint.get("files")
    if (
        not isinstance(files, Mapping)
        or not files
        or any(
            not isinstance(filename, str)
            or not filename
            or not isinstance(digest, str)
            or len(digest) != 64
            for filename, digest in files.items()
        )
    ):
        raise ValueError(f"{name} files must map filenames to SHA-256 digests")
    combined = fingerprint.get("combined_sha256")
    if not isinstance(combined, str) or len(combined) != 64:
        raise ValueError(f"{name} combined_sha256 must be a SHA-256 digest")


def require_tokenizer_compatibility(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    """Require runtime token IDs to match the artifact's tokenizer exactly."""

    _validate_tokenizer_fingerprint(expected, name="compiled tokenizer fingerprint")
    _validate_tokenizer_fingerprint(actual, name="runtime tokenizer fingerprint")
    for field in (
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "all_special_ids",
        "files",
        "combined_sha256",
    ):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"runtime tokenizer fingerprint differs: {field}")


def validate_compiled_constraint_artifact(
    artifact: Mapping[str, Any],
) -> None:
    """Validate an authenticated schema-v2 token-partition NFA."""

    if not isinstance(artifact, Mapping):
        raise ValueError("compiled_constraint must be an object")
    body = dict(artifact)
    digest = body.pop("sha256", None)
    if body.get("schema_version") != COMPILED_CONSTRAINT_SCHEMA_VERSION:
        raise ValueError("compiled constraint schema-version mismatch")
    if body.get("kind") != COMPILED_CONSTRAINT_KIND:
        raise ValueError("compiled constraint kind mismatch")
    if digest != sha256_value(body):
        raise ValueError("compiled constraint digest mismatch")

    nfa = body.get("nfa")
    classes = body.get("token_classes")
    if not isinstance(nfa, Mapping) or not isinstance(classes, list):
        raise ValueError("compiled constraint automaton or token partition is missing")
    state_count = _integer(nfa.get("num_states"), "NFA num_states")
    alphabet_size = _integer(nfa.get("alphabet_size"), "NFA alphabet_size")
    if state_count < 1 or alphabet_size != len(classes) + 2:
        raise ValueError("compiled constraint NFA dimensions are invalid")
    if body.get("other_symbol_id") != len(classes):
        raise ValueError("compiled constraint default symbol is invalid")
    if body.get("stop_symbol_id") != alphabet_size - 1:
        raise ValueError("compiled constraint stop symbol is invalid")

    observed_names: set[str] = set()
    observed_symbols: set[int] = set()
    observed_tokens: set[int] = set()
    for token_class in classes:
        if not isinstance(token_class, Mapping):
            raise ValueError("compiled constraint token class is invalid")
        name = token_class.get("name")
        if not isinstance(name, str) or not name or name in observed_names:
            raise ValueError("compiled constraint token class name is invalid")
        symbol_id = _integer(
            token_class.get("symbol_id"),
            f"token class {name!r} symbol_id",
        )
        token_ids = _integer_list(
            token_class.get("token_ids"),
            f"token class {name!r} token_ids",
            nonempty=True,
            nonnegative=True,
        )
        if (
            symbol_id in observed_symbols
            or not 0 <= symbol_id < len(classes)
            or len(token_ids) != len(set(token_ids))
            or observed_tokens.intersection(token_ids)
        ):
            raise ValueError("compiled constraint token class is invalid")
        observed_names.add(name)
        observed_symbols.add(symbol_id)
        observed_tokens.update(token_ids)
    if observed_symbols != set(range(len(classes))):
        raise ValueError("compiled constraint token class symbols are not contiguous")

    n_low = _integer(body.get("n_low"), "compiled constraint n_low")
    n = _integer(body.get("n"), "compiled constraint n")
    if body.get("length_contract") != terminal_eos_length_contract(n_low, n):
        raise ValueError("compiled constraint terminal-EOS length contract differs")
    _integer_list(
        body.get("prompt_token_ids"),
        "compiled constraint prompt_token_ids",
        nonempty=True,
        nonnegative=True,
    )

    valid_states = set(range(state_count))
    initials = _integer_list(nfa.get("initials"), "NFA initials", nonempty=True)
    finals = _integer_list(nfa.get("finals"), "NFA finals", nonempty=True)
    if not set(initials) <= valid_states or not set(finals) <= valid_states:
        raise ValueError("compiled constraint initial or final state is invalid")
    transitions = nfa.get("transitions")
    wildcards = nfa.get("wildcard_transitions")
    if not isinstance(transitions, list) or not isinstance(wildcards, list):
        raise ValueError("compiled constraint transition lists are missing")
    for edge in transitions:
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError("compiled constraint explicit transition is invalid")
        src, symbol, dst = (
            _integer(edge[0], "transition source"),
            _integer(edge[1], "transition symbol"),
            _integer(edge[2], "transition destination"),
        )
        if (
            src not in valid_states
            or dst not in valid_states
            or not 0 <= symbol < alphabet_size
        ):
            raise ValueError("compiled constraint explicit transition is invalid")
    for edge in wildcards:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("compiled constraint wildcard transition is invalid")
        src, dst = (
            _integer(edge[0], "wildcard source"),
            _integer(edge[1], "wildcard destination"),
        )
        if src not in valid_states or dst not in valid_states:
            raise ValueError("compiled constraint wildcard transition is invalid")

    instruction = body.get("instruction")
    if not isinstance(instruction, Mapping) or any(
        not isinstance(instruction.get(field), str)
        or not str(instruction[field]).strip()
        for field in ("family", "label", "rule")
    ):
        raise ValueError("compiled constraint instruction is invalid")
    _validate_tokenizer_fingerprint(
        body.get("tokenizer_fingerprint"),
        name="compiled tokenizer fingerprint",
    )
    if not isinstance(body.get("provenance"), Mapping):
        raise ValueError("compiled constraint provenance is invalid")


def _validate_row(row: Mapping[str, Any]) -> None:
    family = row.get("constraint")
    if not isinstance(family, str) or not family:
        raise ValueError("constraint must be a non-empty string")
    artifact = row.get("compiled_constraint")
    validate_compiled_constraint_artifact(artifact)

    if row.get("n_low") != artifact["n_low"] or row.get("n") != artifact["n"]:
        raise ValueError("compiled constraint row length contract mismatch")
    if row.get("prompt_token_ids") != artifact["prompt_token_ids"]:
        raise ValueError("compiled constraint row prompt contract mismatch")
    if artifact["instruction"]["family"] != family:
        raise ValueError("compiled constraint row instruction contract mismatch")
    if (
        "nfa_states" in row
        and row.get("nfa_states") != artifact["nfa"]["num_states"]
    ):
        raise ValueError("compiled constraint row NFA-state contract mismatch")

    patterns = row.get("tracked_keyword_token_ids")
    t = _integer(row.get("t"), "t")
    if not isinstance(patterns, list) or len(patterns) != t:
        raise ValueError("tracked_keyword_token_ids must contain exactly t phrases")
    normalized = [
        tuple(
            _integer_list(
                pattern,
                "tracked keyword token pattern",
                nonempty=True,
                nonnegative=True,
            )
        )
        for pattern in patterns
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("tracked keyword token patterns must be distinct")
    separator = _integer(row.get("separator_token_id"), "separator_token_id")
    class_tokens = {
        token_id
        for token_class in artifact["token_classes"]
        for token_id in token_class["token_ids"]
    }
    if separator not in class_tokens or any(
        not set(pattern) <= class_tokens for pattern in normalized
    ):
        raise ValueError("row token metadata disagrees with compiled token classes")


def family_order(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return constraint labels in their first-occurrence workload order."""

    return list(dict.fromkeys(str(row["constraint"]) for row in rows))


def load_workload(
    path: Path,
    *,
    expected_dataset: str | None = None,
) -> Dict[str, Any]:
    """Load and authenticate a current compiled dataset workload."""

    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workload must be a JSON object")
    if raw.get("kind") != COMPILED_WORKLOAD_KIND:
        raise ValueError(
            f"workload kind must be {COMPILED_WORKLOAD_KIND!r}"
        )
    if raw.get("schema_version") != COMPILED_WORKLOAD_SCHEMA_VERSION:
        raise ValueError(
            "baseline requires compiled_keyword_dataset "
            f"schema_version {COMPILED_WORKLOAD_SCHEMA_VERSION} with per-job "
            "compiled_constraint artifacts"
        )
    jobs = raw.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("workload jobs must be a non-empty list")
    if raw.get("total_instances") != len(jobs):
        raise ValueError("workload total_instances does not match jobs")
    dataset = raw.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("workload dataset must be a non-empty string")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"workload dataset {dataset!r} does not match {expected_dataset!r}"
        )
    if raw.get("jobs_sha256") != sha256_value(jobs):
        raise ValueError("workload jobs_sha256 mismatch")
    bounds = {
        (int(row["n_low"]), int(row["n"]))
        for row in jobs
        if isinstance(row, Mapping)
        and isinstance(row.get("n_low"), int)
        and not isinstance(row.get("n_low"), bool)
        and isinstance(row.get("n"), int)
        and not isinstance(row.get("n"), bool)
    }
    if len(bounds) != 1:
        raise ValueError("workload jobs do not share one length interval")
    n_low, n = next(iter(bounds))
    if (
        raw.get("length_interval_including_eos") != [n_low, n]
        or raw.get("length_contract") != terminal_eos_length_contract(n_low, n)
    ):
        raise ValueError("workload terminal-EOS length contract differs")

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for index, row in enumerate(jobs):
        if not isinstance(row, dict):
            raise ValueError(f"workload job {index} is not an object")
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"workload job {index} has no job_id")
        design_index = _integer(row.get("design_index"), "design_index")
        if job_id in seen_ids:
            raise ValueError(f"duplicate workload job_id: {job_id}")
        if design_index in seen_indices:
            raise ValueError(f"duplicate workload design_index: {design_index}")
        if row.get("dataset") != dataset:
            raise ValueError(
                f"workload row {job_id} dataset disagrees with workload dataset"
            )
        seen_ids.add(job_id)
        seen_indices.add(design_index)
        _validate_row(row)

    counts = dict(Counter(str(row["constraint"]) for row in jobs))
    if raw.get("family_counts") != counts:
        raise ValueError("workload family_counts disagrees with jobs")
    raw["_execution_jobs"] = _execution_rows(raw, jobs)
    raw["_family_order"] = family_order(jobs)
    raw["_workload_path"] = str(path)
    raw["_workload_file_sha256"] = sha256_file(path)
    return raw


def keyword_surfaces(row: Mapping[str, Any]) -> List[str]:
    selected = row.get("selected_keywords")
    if isinstance(selected, list) and len(selected) == int(row["t"]):
        surfaces = []
        for entry in selected:
            if not isinstance(entry, Mapping):
                break
            surface = entry.get("surface")
            if not isinstance(surface, str) or not surface.strip():
                break
            surfaces.append(surface.strip())
        if len(surfaces) == int(row["t"]):
            return surfaces
    texts = row.get("tracked_token_texts")
    if isinstance(texts, list) and len(texts) == int(row["t"]):
        return [str(text).strip() for text in texts]
    return [f"keyword_{index + 1}" for index in range(int(row["t"]))]


def _verified_separator_surface(row: Mapping[str, Any], tokenizer: Any) -> str:
    """Return the authenticated separator text after checking the tokenizer."""

    separator_text = row.get("separator_text")
    if not isinstance(separator_text, str) or not separator_text:
        raise ValueError("separator_text must be a non-empty string")
    separator_id = _integer(row.get("separator_token_id"), "separator_token_id")
    try:
        decoded = tokenizer.decode(
            [separator_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode([separator_id], skip_special_tokens=False)
    if str(decoded) != separator_text:
        raise ValueError(
            "runtime tokenizer separator text differs from authenticated workload"
        )
    surface = separator_text.strip()
    if not surface:
        raise ValueError("separator_text has no visible surface")
    return surface


def render_constraint_instruction(
    row: Mapping[str, Any],
    tokenizer: Any | None = None,
) -> str:
    """Render the authenticated instruction rule without family knowledge."""

    _validate_row(row)
    artifact = row["compiled_constraint"]
    min_content = int(artifact["n_low"]) - 1
    max_content = int(artifact["n"]) - 1
    rule = str(artifact["instruction"]["rule"])
    lines = ["Output constraint:", f"- {rule}"]
    if "separator" in rule.casefold():
        if tokenizer is None:
            raise ValueError(
                "separator-aware instruction rendering requires the runtime tokenizer"
            )
        separator = _verified_separator_surface(row, tokenizer)
        lines.append(
            f'- In this rule, "separator" means the exact model-token phrase '
            f"{separator!r}."
        )
    lines.extend(
        (
            (
                "- Matching uses the exact, case-sensitive model-token phrases "
                "compiled into this workload; overlapping occurrences count."
            ),
            (
                f"- Generate between {min_content} and {max_content} content model "
                "tokens, then stop normally."
            ),
        )
    )
    return "\n".join(lines)


def compose_user_content(
    row: Mapping[str, Any],
    tokenizer: Any | None = None,
) -> str:
    """Combine the dataset task and compiled instruction in one user turn."""

    task_prompt_text = row.get("task_prompt_text")
    if not isinstance(task_prompt_text, str) or not task_prompt_text:
        raise ValueError("task_prompt_text must be a non-empty string")
    prompt_text = row.get("prompt_text")
    if prompt_text != task_prompt_text:
        raise ValueError("prompt_text must equal the authenticated task_prompt_text")
    return task_prompt_text + "\n\n" + render_constraint_instruction(row, tokenizer)


def render_prompt_ids(tokenizer: Any, row: Mapping[str, Any]) -> List[int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer must provide a chat template")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": compose_user_content(row, tokenizer)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], (list, tuple)):
        if len(rendered) != 1:
            raise ValueError("chat template unexpectedly returned a batch")
        rendered = rendered[0]
    if not isinstance(rendered, (list, tuple)) or not rendered:
        raise ValueError("chat template returned no prompt token IDs")
    return [_integer(token, "prompt token ID") for token in rendered]


def overlapping_positions(
    token_ids: Sequence[int],
    pattern: Sequence[int],
) -> List[int]:
    content = tuple(int(token) for token in token_ids)
    needle = tuple(int(token) for token in pattern)
    if not needle:
        raise ValueError("pattern must be non-empty")
    width = len(needle)
    return [
        start
        for start in range(len(content) - width + 1)
        if content[start : start + width] == needle
    ]


def nfa_accepts_content(
    row: Mapping[str, Any],
    content_ids: Sequence[int],
) -> bool:
    """Interpret the row's compact NFA over its authenticated token partition."""

    _validate_row(row)
    artifact = row["compiled_constraint"]
    nfa = artifact["nfa"]
    token_to_symbol = {
        int(token_id): int(token_class["symbol_id"])
        for token_class in artifact["token_classes"]
        for token_id in token_class["token_ids"]
    }
    other_symbol = int(artifact["other_symbol_id"])
    explicit: Dict[tuple[int, int], set[int]] = {}
    for src, symbol, dst in nfa["transitions"]:
        explicit.setdefault((int(src), int(symbol)), set()).add(int(dst))
    wildcards: Dict[int, set[int]] = {}
    for src, dst in nfa["wildcard_transitions"]:
        wildcards.setdefault(int(src), set()).add(int(dst))

    def advance(active: set[int], symbol: int) -> set[int]:
        destinations: set[int] = set()
        for state in active:
            destinations.update(explicit.get((state, symbol), ()))
            destinations.update(wildcards.get(state, ()))
        return destinations

    active = {int(state) for state in nfa["initials"]}
    for token_id in content_ids:
        symbol = token_to_symbol.get(int(token_id), other_symbol)
        active = advance(active, symbol)
        if not active:
            return False
    active = advance(active, int(artifact["stop_symbol_id"]))
    return bool(active.intersection(int(state) for state in nfa["finals"]))


def _occurrence_diagnostics(
    row: Mapping[str, Any],
    content_ids: Sequence[int],
) -> Dict[str, Any]:
    patterns = [
        [int(token) for token in pattern]
        for pattern in row["tracked_keyword_token_ids"]
    ]
    positions = [overlapping_positions(content_ids, pattern) for pattern in patterns]
    separator_id = int(row["separator_token_id"])
    separator_positions = [
        index for index, token in enumerate(content_ids) if int(token) == separator_id
    ]
    first_separator = separator_positions[0] if separator_positions else None
    left_positions: List[List[int]] = []
    right_positions: List[List[int]] = []
    if first_separator is not None:
        left = list(content_ids[:first_separator])
        right = list(content_ids[first_separator + 1 :])
        for pattern in patterns:
            left_positions.append(overlapping_positions(left, pattern))
            right_positions.append(
                [
                    first_separator + 1 + start
                    for start in overlapping_positions(right, pattern)
                ]
            )
    return {
        "keyword_surfaces": keyword_surfaces(row),
        "occurrence_positions": positions,
        "occurrence_counts": [len(value) for value in positions],
        "separator_positions": separator_positions,
        "first_separator_position": first_separator,
        "left_occurrence_positions": left_positions,
        "right_occurrence_positions": right_positions,
        "left_occurrence_counts": [len(value) for value in left_positions],
        "right_occurrence_counts": [len(value) for value in right_positions],
    }


def evaluate_generated_ids(
    row: Mapping[str, Any],
    generated_ids: Sequence[int],
    tokenizer: Any,
    *,
    terminal_token_ids: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Score raw newly generated IDs against an embedded compiled automaton."""

    _validate_row(row)
    if tokenizer.eos_token_id is None and terminal_token_ids is None:
        raise ValueError("tokenizer must define eos_token_id")
    terminals = (
        [int(tokenizer.eos_token_id)]
        if terminal_token_ids is None
        else [int(token_id) for token_id in terminal_token_ids]
    )
    terminals = list(dict.fromkeys(terminals))
    if not terminals:
        raise ValueError("terminal_token_ids must be nonempty")
    terminal_set = set(terminals)
    generated = [int(token) for token in generated_ids]
    terminal_positions = [
        index for index, token in enumerate(generated) if token in terminal_set
    ]
    terminal_eos_ok = terminal_positions == [len(generated) - 1]
    actual_terminal_id = generated[-1] if terminal_eos_ok else None
    content = (
        generated[:-1]
        if terminal_eos_ok
        else [token for token in generated if token not in terminal_set]
    )
    all_special = {
        int(token) for token in getattr(tokenizer, "all_special_ids", ())
    }
    invalid_content_special_ids = [token for token in content if token in all_special]
    valid_content_tokens = not invalid_content_special_ids
    artifact = row["compiled_constraint"]
    raw_token_budget_ok = (
        int(artifact["n_low"]) <= len(generated) <= int(artifact["n"])
    )
    length_ok = bool(terminal_eos_ok and raw_token_budget_ok)
    semantic_accepts = (
        valid_content_tokens and nfa_accepts_content(row, content)
    )
    strict_accepts = bool(length_ok and valid_content_tokens and semantic_accepts)
    return {
        "strict_accepts": strict_accepts,
        "semantic_constraint_accepts": bool(semantic_accepts),
        "length_ok": length_ok,
        "raw_token_budget_ok": bool(raw_token_budget_ok),
        "terminal_eos_ok": bool(terminal_eos_ok),
        "valid_content_tokens": bool(valid_content_tokens),
        "invalid_content_special_ids": invalid_content_special_ids,
        "eos_positions": terminal_positions,
        "terminal_token_ids": terminals,
        "actual_terminal_token_id": actual_terminal_id,
        "generated_total_token_count": len(generated),
        "generated_content_token_count": len(content),
        "content_token_ids": content,
        **_occurrence_diagnostics(row, content),
    }


__all__ = [
    "COMPILED_CONSTRAINT_KIND",
    "COMPILED_CONSTRAINT_SCHEMA_VERSION",
    "COMPILED_WORKLOAD_KIND",
    "COMPILED_WORKLOAD_SCHEMA_VERSION",
    "compose_user_content",
    "evaluate_generated_ids",
    "family_order",
    "keyword_surfaces",
    "load_workload",
    "nfa_accepts_content",
    "overlapping_positions",
    "render_constraint_instruction",
    "render_prompt_ids",
    "require_tokenizer_compatibility",
    "sha256_file",
    "sha256_value",
    "terminal_eos_length_contract",
    "validate_compiled_constraint_artifact",
]
