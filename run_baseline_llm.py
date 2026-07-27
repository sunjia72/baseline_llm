#!/usr/bin/env python3
"""Run a local instruction model as an unconstrained generation baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from constraint_baseline import (
    compose_user_content,
    evaluate_generated_ids,
    family_order,
    keyword_surfaces,
    load_workload,
    render_constraint_instruction,
    render_prompt_ids,
    require_tokenizer_compatibility,
    sha256_file,
    sha256_value,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/project/aip-ksmeel/sunjia72/models/Qwen3.5-2B")
BASELINE_METHOD = "baseline_llm"
RESULT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "baseline_llm_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("common_gen", "coauthor", "both"),
        default="both",
    )
    parser.add_argument(
        "--common_gen_workload",
        type=Path,
        help="Required for CommonGen: compiled_keyword_dataset schema v3 JSON.",
    )
    parser.add_argument(
        "--coauthor_workload",
        type=Path,
        help="Required for CoAuthor: compiled_keyword_dataset schema v3 JSON.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model_label",
        default="",
        help="Optional readable model label; defaults to the local model directory name.",
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--selection",
        choices=("midpoint_per_family", "all"),
        default="midpoint_per_family",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated design indices; overrides --selection.",
    )
    parser.add_argument("--samples_per_job", type=int, default=1)
    parser.add_argument("--generation_mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible partial run in an existing --output_dir.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and save rendered instructions without loading the model.",
    )
    parser.add_argument(
        "--slurm_nodes",
        default="",
        help=(
            "Run one persistent worker per node/GPU pair. Use 'allocation' for "
            "$SLURM_JOB_NODELIST or provide comma-separated node names."
        ),
    )
    parser.add_argument(
        "--slurm_gpus",
        default="0,1,2,3",
        help="Comma-separated physical GPU IDs to use on every --slurm_nodes node.",
    )
    parser.add_argument(
        "--cpus_per_worker",
        type=int,
        default=8,
        help="CPUs assigned to each persistent Slurm GPU worker.",
    )
    parser.add_argument("--worker_plan", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _parse_indices(raw: str) -> List[int]:
    if not raw.strip():
        return []
    result: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending index range: {piece}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(piece))
    if len(result) != len(set(result)):
        raise ValueError("--indices contains duplicates")
    return result


def select_jobs(
    jobs: Sequence[Mapping[str, Any]],
    selection: str,
    indices: Sequence[int],
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in jobs]
    if indices:
        by_index = {int(row["design_index"]): row for row in rows}
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise ValueError(f"design indices are absent from workload: {missing}")
        return [by_index[index] for index in indices]
    if selection == "all":
        return rows
    if selection != "midpoint_per_family":
        raise ValueError(f"unsupported selection: {selection}")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["constraint"])].append(row)
    selected = []
    for family in family_order(rows):
        family_rows = sorted(grouped[family], key=lambda row: int(row["family_index"]))
        if not family_rows:
            raise ValueError(f"workload has no rows for {family}")
        selected.append(family_rows[len(family_rows) // 2])
    return selected


def _datasets(args: argparse.Namespace) -> List[str]:
    return ["common_gen", "coauthor"] if args.dataset == "both" else [args.dataset]


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid partial result JSON at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object result at {path}:{line_number}")
        rows.append(value)
    return rows


def _sample_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        return (
            str(row["dataset"]),
            str(row["job_id"]),
            int(row["sample_index"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sample key fields: {row}") from error


def _sample_seed(base_seed: int, row: Mapping[str, Any], sample_index: int) -> int:
    return (
        int(base_seed) + int(row["seed"]) * 1009 + int(sample_index)
    ) % (2**31 - 1)


def _planned_tasks(
    selected: Sequence[Mapping[str, Any]],
    samples_per_job: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for row in selected:
        for sample_index in range(samples_per_job):
            tasks.append(
                {
                    "ordinal": len(tasks),
                    "row": dict(row),
                    "sample_index": sample_index,
                    "sample_seed": _sample_seed(base_seed, row, sample_index),
                }
            )
    return tasks


def _balanced_assignments(
    tasks: Sequence[Mapping[str, Any]],
    slot_count: int,
) -> List[List[Dict[str, Any]]]:
    if slot_count < 1:
        raise ValueError("slot_count must be positive")
    return [
        [dict(task) for task in tasks[slot_index::slot_count]]
        for slot_index in range(slot_count)
    ]


def _resolve_slurm_nodes(raw: str) -> List[str]:
    value = raw.strip()
    if not value:
        return []
    if value == "allocation":
        value = os.environ.get("SLURM_JOB_NODELIST", "").strip()
        if not value:
            raise ValueError(
                "--slurm_nodes=allocation requires $SLURM_JOB_NODELIST"
            )
        command = ["scontrol", "show", "hostnames", value]
        nodes = subprocess.check_output(command, text=True).splitlines()
    elif "[" in value:
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", value],
            text=True,
        ).splitlines()
    else:
        nodes = value.split(",")
    nodes = [node.strip() for node in nodes if node.strip()]
    if not nodes:
        raise ValueError("--slurm_nodes resolved to no hosts")
    if len(nodes) != len(set(nodes)):
        raise ValueError("--slurm_nodes contains duplicate hosts")
    return nodes


def _parse_gpu_ids(raw: str) -> List[int]:
    try:
        gpu_ids = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    except ValueError as error:
        raise ValueError("--slurm_gpus must be comma-separated integers") from error
    if not gpu_ids:
        raise ValueError("--slurm_gpus contains no GPU IDs")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("--slurm_gpus IDs must be nonnegative")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--slurm_gpus contains duplicate IDs")
    return gpu_ids


def _worker_slots(nodes: Sequence[str], gpu_ids: Sequence[int]) -> List[Dict[str, Any]]:
    if not nodes or not gpu_ids:
        raise ValueError("worker slots require nonempty nodes and GPU IDs")
    return [
        {
            "slot_index": len(gpu_ids) * node_index + gpu_index,
            "node": str(node),
            "physical_gpu_id": int(gpu_id),
        }
        for node_index, node in enumerate(nodes)
        for gpu_index, gpu_id in enumerate(gpu_ids)
    ]


def _model_slug(model_path: Path) -> str:
    text = model_path.name.lower()
    slug = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(piece for piece in slug.split("_") if piece) or "local_model"


def _tokenizer_artifact_hashes(model_path: Path) -> Dict[str, str]:
    candidates = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    names = [name for name in candidates if (model_path / name).is_file()]
    if not names:
        raise ValueError("local model directory contains no tokenizer artifacts")
    return {name: sha256_file(model_path / name) for name in names}


def _runtime_tokenizer_fingerprint(
    tokenizer: Any,
    model_path: Path,
) -> Dict[str, Any]:
    files = _tokenizer_artifact_hashes(model_path)
    return {
        "path": str(model_path),
        "vocab_size": int(len(tokenizer)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "all_special_ids": [
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", ())
        ],
        "files": files,
        "combined_sha256": sha256_value(files),
    }


def _load_scoring_contract(
    model_path: Path,
    *,
    local_files_only: bool,
) -> tuple[Any, Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("target tokenizer must define EOS and PAD")
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    configured_terminals = getattr(config, "eos_token_id", None)
    if configured_terminals is None:
        terminal_ids = [int(tokenizer.eos_token_id)]
        terminal_source = "tokenizer.eos_token_id"
    else:
        terminal_ids = (
            [int(configured_terminals)]
            if isinstance(configured_terminals, int)
            else [int(token_id) for token_id in configured_terminals]
        )
        terminal_ids.insert(0, int(tokenizer.eos_token_id))
        terminal_ids = list(dict.fromkeys(terminal_ids))
        terminal_source = "tokenizer EOS plus top-level model config eos_token_id"
    special_ids = {
        int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
    }
    if not terminal_ids or any(
        token_id not in special_ids for token_id in terminal_ids
    ):
        raise ValueError("model terminal IDs must be tokenizer special tokens")
    text_config = getattr(config, "text_config", None)
    tokenizer_fingerprint = _runtime_tokenizer_fingerprint(tokenizer, model_path)
    contract = {
        "model_type": str(getattr(config, "model_type", "")),
        "architectures": list(getattr(config, "architectures", ()) or ()),
        "model_max_position_embeddings": getattr(
            text_config if text_config is not None else config,
            "max_position_embeddings",
            None,
        ),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
        "terminal_token_ids": terminal_ids,
        "terminal_token_texts": [
            str(tokenizer.convert_ids_to_tokens(token_id))
            for token_id in terminal_ids
        ],
        "terminal_policy_source": terminal_source,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "prompt_chat_template_kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    }
    contract["contract_sha256"] = sha256_value(contract)
    return tokenizer, contract


def _prepare_scoring_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Verify compiled token IDs; never retokenize or rebuild a constraint."""

    prepared: List[Dict[str, Any]] = []
    artifact_identities = []
    for source in rows:
        artifact = source["compiled_constraint"]
        require_tokenizer_compatibility(
            artifact["tokenizer_fingerprint"],
            tokenizer_fingerprint,
        )
        prepared.append(dict(source))
        artifact_identities.append(
            {
                "dataset": source["dataset"],
                "job_id": source["job_id"],
                "compiled_constraint_sha256": artifact["sha256"],
            }
        )
    metadata = {
        "policy": "use authenticated compiled token IDs without retokenization",
        "jobs": len(prepared),
        "tokenizer_combined_sha256": tokenizer_fingerprint["combined_sha256"],
        "compiled_artifact_identities_sha256": sha256_value(
            artifact_identities
        ),
        "prepared_scoring_rows_sha256": sha256_value(prepared),
    }
    return prepared, metadata


def _model_artifact_hashes(model_path: Path) -> Dict[str, str]:
    if not model_path.is_dir():
        raise ValueError("--model must be a local model directory")
    names = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        *sorted(path.name for path in model_path.glob("*.safetensors")),
    ]
    names = list(dict.fromkeys(name for name in names if (model_path / name).is_file()))
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("local model directory contains no safetensors weights")
    return {name: sha256_file(model_path / name) for name in names}


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    args: argparse.Namespace,
    sample_seed: int,
) -> Dict[str, Any]:
    prompt_ids = render_prompt_ids(tokenizer, row)
    device = torch.device(args.device)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    terminal_ids = [int(token_id) for token_id in args.terminal_token_ids]
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(row["n"]),
        "do_sample": args.generation_mode == "sample",
        "eos_token_id": terminal_ids[0] if len(terminal_ids) == 1 else terminal_ids,
        "pad_token_id": int(tokenizer.pad_token_id),
        "use_cache": True,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs.update(
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
        )
    _set_seed(sample_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_s = time.perf_counter() - started
    new_ids = [
        int(token)
        for token in output[0, input_ids.shape[1] :].detach().cpu().tolist()
    ]
    evaluation = evaluate_generated_ids(
        row,
        new_ids,
        tokenizer,
        terminal_token_ids=terminal_ids,
    )
    content_ids = evaluation["content_token_ids"]
    return {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": sha256_value(prompt_ids),
        "new_token_ids": new_ids,
        "generated_text": _decode(tokenizer, content_ids),
        "generation_runtime_s": runtime_s,
        "cuda_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "evaluation": evaluation,
    }


def _build_record(
    model: Any,
    tokenizer: Any,
    task: Mapping[str, Any],
    args: argparse.Namespace,
    execution: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    row = task["row"]
    artifact = row["compiled_constraint"]
    result = _generate(
        model,
        tokenizer,
        row,
        args,
        int(task["sample_seed"]),
    )
    nfa_states = row.get("nfa_states")
    if nfa_states is None:
        nfa_states = artifact["nfa"]["num_states"]
    unrolled_states = row.get("unrolled_states")
    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "method": BASELINE_METHOD,
        "status": "success",
        "dataset": row["dataset"],
        "job_id": row["job_id"],
        "design_index": row["design_index"],
        "base_instance_id": row["base_instance_id"],
        "constraint": row["constraint"],
        "k": row.get("k"),
        "t": row["t"],
        "n_low": row["n_low"],
        "n": row["n"],
        "compiled_constraint_sha256": artifact["sha256"],
        "nfa_states": int(nfa_states),
        "unrolled_states": (
            int(unrolled_states) if unrolled_states is not None else None
        ),
        "precompute_runtime_s": None,
        "keywords": keyword_surfaces(row),
        "sample_index": int(task["sample_index"]),
        "sample_seed": int(task["sample_seed"]),
        "instruction": render_constraint_instruction(row, tokenizer),
        **result,
    }
    if execution is not None:
        record["execution"] = dict(execution)
    return record


def _load_model_runtime(args: argparse.Namespace) -> tuple[Any, Any, Dict[str, Any]]:
    model_path = Path(args.model).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("target tokenizer must define EOS and PAD")
    require_tokenizer_compatibility(
        args.scoring_contract["tokenizer_fingerprint"],
        _runtime_tokenizer_fingerprint(tokenizer, model_path),
    )
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=_dtype(args.dtype),
        local_files_only=bool(args.local_files_only),
    ).to(device).eval()
    runtime = {
        "python_torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
        "terminal_token_ids": [
            int(token_id) for token_id in args.terminal_token_ids
        ],
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_compute_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
    }
    return model, tokenizer, runtime


def _worker_main(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported worker-plan schema")
    expected_plan_hash = plan.get("plan_sha256")
    hash_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if expected_plan_hash != sha256_value(hash_payload):
        raise ValueError("worker plan hash mismatch")

    worker_dir = Path(plan["worker_dir"])
    record_dir = worker_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    generation = plan["generation"]
    worker_args = SimpleNamespace(
        model=Path(generation["model"]),
        device=str(generation["device"]),
        dtype=str(generation["dtype"]),
        local_files_only=True,
        generation_mode=str(generation["generation_mode"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        terminal_token_ids=[
            int(token_id) for token_id in generation["terminal_token_ids"]
        ],
        scoring_contract=dict(generation["scoring_contract"]),
    )
    tasks = list(plan["tasks"])
    status_path = worker_dir / "status.json"
    _atomic_json(
        status_path,
        {
            "state": "loading_model",
            "completed": 0,
            "total": len(tasks),
            "actual_node": socket.gethostname(),
        },
    )
    _, current_contract = _load_scoring_contract(
        Path(worker_args.model),
        local_files_only=True,
    )
    expected_contract = generation["scoring_contract"]
    if current_contract["contract_sha256"] != expected_contract["contract_sha256"]:
        raise ValueError("worker model/tokenizer scoring contract mismatch")
    for task in tasks:
        require_tokenizer_compatibility(
            task["row"]["compiled_constraint"]["tokenizer_fingerprint"],
            current_contract["tokenizer_fingerprint"],
        )
    model, tokenizer, runtime = _load_model_runtime(worker_args)
    metadata = {
        "schema_version": 1,
        "run_signature_sha256": plan["run_signature_sha256"],
        "dispatch_id": plan["dispatch_id"],
        "slot_index": int(plan["slot"]["slot_index"]),
        "scheduled_node": plan["slot"]["node"],
        "actual_node": socket.gethostname(),
        "physical_gpu_id": int(plan["slot"]["physical_gpu_id"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpus_on_node": os.cpu_count(),
        "runtime": runtime,
        "tasks": len(tasks),
        "scoring_contract_sha256": current_contract["contract_sha256"],
    }
    _atomic_json(worker_dir / "metadata.json", metadata)

    completed = 0
    for task in tasks:
        ordinal = int(task["ordinal"])
        record_path = record_dir / f"{ordinal:06d}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if _sample_key(record) != _sample_key(
                {
                    "dataset": task["row"]["dataset"],
                    "job_id": task["row"]["job_id"],
                    "sample_index": task["sample_index"],
                }
            ):
                raise ValueError(f"worker checkpoint key mismatch: {record_path}")
        else:
            execution = {
                "run_signature_sha256": plan["run_signature_sha256"],
                "dispatch_id": plan["dispatch_id"],
                "task_ordinal": ordinal,
                "slot_index": int(plan["slot"]["slot_index"]),
                "scheduled_node": plan["slot"]["node"],
                "actual_node": socket.gethostname(),
                "physical_gpu_id": int(plan["slot"]["physical_gpu_id"]),
            }
            record = _build_record(
                model,
                tokenizer,
                task,
                worker_args,
                execution=execution,
            )
            _atomic_json(record_path, record)
        completed += 1
        status = (
            "PASS" if record["evaluation"]["strict_accepts"] else "FAIL"
        )
        print(
            f"[{completed:02d}/{len(tasks):02d}] ordinal={ordinal:03d} "
            f"{record['job_id']} {status} "
            f"time={record['generation_runtime_s']:.2f}s",
            flush=True,
        )
        _atomic_json(
            status_path,
            {
                "state": "running" if completed < len(tasks) else "complete",
                "completed": completed,
                "total": len(tasks),
                "last_ordinal": ordinal,
                "last_key": list(_sample_key(record)),
                "actual_node": socket.gethostname(),
            },
        )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def summarize(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def group_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        strict = sum(bool(row["evaluation"]["strict_accepts"]) for row in rows)
        semantic = sum(
            bool(row["evaluation"]["semantic_constraint_accepts"]) for row in rows
        )
        eos = sum(bool(row["evaluation"]["terminal_eos_ok"]) for row in rows)
        length = sum(bool(row["evaluation"]["length_ok"]) for row in rows)
        token_budget = sum(
            bool(row["evaluation"]["raw_token_budget_ok"]) for row in rows
        )
        return {
            "samples": total,
            "strict_successes": strict,
            "strict_success_rate": _safe_rate(strict, total),
            "semantic_constraint_successes": semantic,
            "semantic_constraint_success_rate": _safe_rate(semantic, total),
            "terminal_eos_successes": eos,
            "terminal_eos_rate": _safe_rate(eos, total),
            "length_successes": length,
            "length_success_rate": _safe_rate(length, total),
            "raw_token_budget_successes": token_budget,
            "raw_token_budget_success_rate": _safe_rate(token_budget, total),
        }

    by_dataset: Dict[str, Any] = {}
    by_family: Dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in samples}):
        by_dataset[dataset] = group_summary(
            [row for row in samples if row["dataset"] == dataset]
        )
    for family in family_order(samples):
        family_rows = [row for row in samples if row["constraint"] == family]
        if family_rows:
            by_family[family] = group_summary(family_rows)
    runtimes = [float(row["generation_runtime_s"]) for row in samples]
    return {
        **group_summary(samples),
        "by_dataset": by_dataset,
        "by_constraint": by_family,
        "generation_runtime_s": {
            "total": sum(runtimes),
            "mean": sum(runtimes) / len(runtimes) if runtimes else 0.0,
            "max": max(runtimes) if runtimes else 0.0,
        },
    }


def _write_csv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "schema_version",
        "method",
        "status",
        "dataset",
        "job_id",
        "design_index",
        "base_instance_id",
        "constraint",
        "k",
        "t",
        "compiled_constraint_sha256",
        "nfa_states",
        "unrolled_states",
        "sample_index",
        "sample_seed",
        "strict_accepts",
        "semantic_constraint_accepts",
        "terminal_eos_ok",
        "length_ok",
        "raw_token_budget_ok",
        "generated_total_token_count",
        "generated_content_token_count",
        "precompute_runtime_s",
        "generation_runtime_s",
        "cuda_peak_allocated_gib",
        "generated_text",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            evaluation = row["evaluation"]
            writer.writerow(
                {
                    field: (
                        evaluation[field]
                        if field in evaluation
                        else row.get(field)
                    )
                    for field in fields
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_sample_record(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    run_signature: str,
    *,
    require_execution: bool,
) -> None:
    row = task["row"]
    expected_key = (
        str(row["dataset"]),
        str(row["job_id"]),
        int(task["sample_index"]),
    )
    if _sample_key(record) != expected_key:
        raise ValueError(
            f"sample key mismatch: expected {expected_key}, got {_sample_key(record)}"
        )
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"sample {expected_key} has wrong schema version")
    if record.get("method") != BASELINE_METHOD:
        raise ValueError(f"sample {expected_key} has wrong method")
    if record.get("status") != "success":
        raise ValueError(f"sample {expected_key} has non-success execution status")
    for field in (
        "design_index",
        "base_instance_id",
        "constraint",
        "t",
        "n_low",
        "n",
    ):
        if record.get(field) != row.get(field):
            raise ValueError(f"sample {expected_key} has mismatched {field}")
    if record.get("k") != row.get("k"):
        raise ValueError(f"sample {expected_key} has mismatched k")
    artifact = row["compiled_constraint"]
    if record.get("compiled_constraint_sha256") != artifact["sha256"]:
        raise ValueError(
            f"sample {expected_key} has mismatched compiled constraint"
        )
    expected_nfa_states = row.get("nfa_states")
    if expected_nfa_states is None:
        expected_nfa_states = artifact["nfa"]["num_states"]
    if record.get("nfa_states") != int(expected_nfa_states):
        raise ValueError(f"sample {expected_key} has mismatched nfa_states")
    expected_unrolled = row.get("unrolled_states")
    if record.get("unrolled_states") != (
        int(expected_unrolled) if expected_unrolled is not None else None
    ):
        raise ValueError(f"sample {expected_key} has mismatched unrolled_states")
    if record.get("precompute_runtime_s") is not None:
        raise ValueError(
            f"sample {expected_key} has invalid baseline precompute runtime"
        )
    if int(record.get("sample_seed", -1)) != int(task["sample_seed"]):
        raise ValueError(f"sample {expected_key} has mismatched seed")
    if not isinstance(record.get("new_token_ids"), list):
        raise ValueError(f"sample {expected_key} has no generated token list")
    if not isinstance(record.get("evaluation"), dict):
        raise ValueError(f"sample {expected_key} has no evaluation object")
    if require_execution:
        execution = record.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"distributed sample {expected_key} lacks provenance")
        if execution.get("run_signature_sha256") != run_signature:
            raise ValueError(
                f"distributed sample {expected_key} has wrong run signature"
            )
        if int(execution.get("task_ordinal", -1)) != int(task["ordinal"]):
            raise ValueError(
                f"distributed sample {expected_key} has wrong task ordinal"
            )


def _collect_samples(
    output_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    run_signature: str,
) -> tuple[List[Dict[str, Any]], Dict[tuple[str, str, int], Dict[str, Any]]]:
    expected = {
        (
            str(task["row"]["dataset"]),
            str(task["row"]["job_id"]),
            int(task["sample_index"]),
        ): task
        for task in tasks
    }
    if len(expected) != len(tasks):
        raise ValueError("run plan contains duplicate task keys")
    by_key: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    sources: Dict[tuple[str, str, int], Path] = {}

    canonical_path = output_dir / "samples.jsonl"
    for record in _load_jsonl(canonical_path):
        key = _sample_key(record)
        if key not in expected:
            raise ValueError(f"{canonical_path} contains an out-of-plan sample {key}")
        _validate_sample_record(
            record,
            expected[key],
            run_signature,
            require_execution=False,
        )
        if key in by_key:
            raise ValueError(f"duplicate sample {key} in {canonical_path}")
        by_key[key] = record
        sources[key] = canonical_path

    record_glob = output_dir / "dispatches"
    if len(by_key) < len(tasks) and record_glob.exists():
        paths = sorted(record_glob.glob("dispatch_*/slot_*/records/*.json"))
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                raise ValueError(f"invalid atomic worker record: {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"non-object atomic worker record: {path}")
            key = _sample_key(record)
            if key not in expected:
                raise ValueError(f"{path} contains an out-of-plan sample {key}")
            _validate_sample_record(
                record,
                expected[key],
                run_signature,
                require_execution=True,
            )
            if key in by_key:
                raise ValueError(
                    f"duplicate sample {key} in {sources[key]} and {path}"
                )
            by_key[key] = record
            sources[key] = path

    ordered = []
    for task in tasks:
        key = (
            str(task["row"]["dataset"]),
            str(task["row"]["job_id"]),
            int(task["sample_index"]),
        )
        if key in by_key:
            ordered.append(by_key[key])
    return ordered, by_key


def _next_dispatch_id(dispatch_root: Path) -> str:
    used = []
    if dispatch_root.exists():
        for path in dispatch_root.glob("dispatch_*"):
            try:
                used.append(int(path.name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return f"dispatch_{max(used, default=-1) + 1:03d}"


def _slurm_command(
    slot: Mapping[str, Any],
    worker_dir: Path,
    plan_path: Path,
    cpus_per_worker: int,
) -> List[str]:
    return [
        "srun",
        "--overlap",
        "--exact",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={slot['node']}",
        f"--cpus-per-task={cpus_per_worker}",
        "--cpu-bind=none",
        "--kill-on-bad-exit=1",
        "--oom-kill-step=1",
        "--input=/dev/null",
        "--unbuffered",
        "--open-mode=truncate",
        f"--output={worker_dir / 'stdout.log'}",
        f"--error={worker_dir / 'stderr.log'}",
        f"--chdir={ROOT}",
        "env",
        f"CUDA_VISIBLE_DEVICES={slot['physical_gpu_id']}",
        f"OMP_NUM_THREADS={cpus_per_worker}",
        f"MKL_NUM_THREADS={cpus_per_worker}",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONDONTWRITEBYTECODE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker_plan",
        str(plan_path),
    ]


def _dispatch_slurm(
    args: argparse.Namespace,
    output_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    completed_keys: set[tuple[str, str, int]],
    run_config: Dict[str, Any],
    config_path: Path,
) -> None:
    nodes = _resolve_slurm_nodes(args.slurm_nodes)
    gpu_ids = _parse_gpu_ids(args.slurm_gpus)
    slots = _worker_slots(nodes, gpu_ids)
    if args.cpus_per_worker < 1:
        raise ValueError("--cpus_per_worker must be positive")
    pending = [
        dict(task)
        for task in tasks
        if (
            str(task["row"]["dataset"]),
            str(task["row"]["job_id"]),
            int(task["sample_index"]),
        )
        not in completed_keys
    ]
    if not pending:
        return
    assignments = _balanced_assignments(pending, len(slots))
    dispatch_root = output_dir / "dispatches"
    dispatch_id = _next_dispatch_id(dispatch_root)
    dispatch_dir = dispatch_root / dispatch_id
    dispatch_dir.mkdir(parents=True, exist_ok=False)
    generation = {
        "model": str(Path(args.model).expanduser().resolve()),
        "device": "cuda:0",
        "dtype": args.dtype,
        "generation_mode": args.generation_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "terminal_token_ids": [
            int(token_id) for token_id in args.terminal_token_ids
        ],
        "scoring_contract": dict(args.scoring_contract),
    }
    worker_specs: List[Dict[str, Any]] = []
    for slot, assigned in zip(slots, assignments):
        if not assigned:
            continue
        worker_dir = dispatch_dir / f"slot_{int(slot['slot_index']):03d}"
        worker_dir.mkdir(parents=True, exist_ok=False)
        plan = {
            "schema_version": 1,
            "dispatch_id": dispatch_id,
            "run_signature_sha256": run_config["run_signature_sha256"],
            "worker_dir": str(worker_dir),
            "slot": dict(slot),
            "generation": generation,
            "tasks": assigned,
        }
        plan["plan_sha256"] = sha256_value(plan)
        plan_path = worker_dir / "plan.json"
        _atomic_json(plan_path, plan)
        command = _slurm_command(
            slot,
            worker_dir,
            plan_path,
            args.cpus_per_worker,
        )
        _atomic_json(worker_dir / "command.json", command)
        worker_specs.append(
            {
                "slot": dict(slot),
                "tasks": len(assigned),
                "worker_dir": str(worker_dir),
                "plan_sha256": plan["plan_sha256"],
                "command": command,
            }
        )

    dispatch_metadata: Dict[str, Any] = {
        "dispatch_id": dispatch_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "nodes": nodes,
        "physical_gpu_ids": gpu_ids,
        "cpus_per_worker": int(args.cpus_per_worker),
        "slots_available": len(slots),
        "workers_launched": len(worker_specs),
        "pending_tasks": len(pending),
        "worker_task_counts": [spec["tasks"] for spec in worker_specs],
        "workers": [
            {key: value for key, value in spec.items() if key != "command"}
            for spec in worker_specs
        ],
        "state": "launching",
    }
    execution = run_config.setdefault("execution", {})
    execution.setdefault("dispatch_history", []).append(dispatch_metadata)
    _atomic_json(config_path, run_config)

    processes: Dict[int, subprocess.Popen[Any]] = {}
    try:
        for spec in worker_specs:
            slot_index = int(spec["slot"]["slot_index"])
            processes[slot_index] = subprocess.Popen(spec["command"])
    except Exception:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise

    dispatch_metadata["state"] = "running"
    _atomic_json(config_path, run_config)
    task_counts = dispatch_metadata["worker_task_counts"]
    print(
        f"Dispatched {len(pending)} pending tasks across "
        f"{len(worker_specs)} persistent workers on {len(nodes)} nodes; "
        f"tasks/worker={min(task_counts)}..{max(task_counts)}",
        flush=True,
    )
    active = dict(processes)
    exit_codes: Dict[int, int] = {}
    last_checkpoint_count = -1
    while active:
        checkpoint_count = sum(
            1 for _ in dispatch_dir.glob("slot_*/records/*.json")
        )
        if checkpoint_count != last_checkpoint_count:
            print(
                f"[distributed progress] {checkpoint_count}/{len(pending)} "
                "atomic records complete",
                flush=True,
            )
            last_checkpoint_count = checkpoint_count
        for slot_index, process in list(active.items()):
            return_code = process.poll()
            if return_code is not None:
                exit_codes[slot_index] = int(return_code)
                del active[slot_index]
                print(
                    f"[worker slot {slot_index:02d}] exited with code "
                    f"{return_code}; {len(active)} workers remain",
                    flush=True,
                )
        if active:
            time.sleep(1.0)

    dispatch_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    dispatch_metadata["exit_codes"] = {
        str(slot_index): exit_codes[slot_index]
        for slot_index in sorted(exit_codes)
    }
    dispatch_metadata["worker_metadata"] = []
    for spec in worker_specs:
        metadata_path = Path(spec["worker_dir"]) / "metadata.json"
        if metadata_path.is_file():
            dispatch_metadata["worker_metadata"].append(
                json.loads(metadata_path.read_text(encoding="utf-8"))
            )
    failed = {
        slot_index: code
        for slot_index, code in exit_codes.items()
        if code != 0
    }
    dispatch_metadata["state"] = "failed" if failed else "complete"
    _atomic_json(config_path, run_config)
    if failed:
        details = ", ".join(
            f"slot {slot_index}: exit {code}"
            for slot_index, code in sorted(failed.items())
        )
        raise RuntimeError(
            f"distributed workers failed ({details}); resume the same output "
            f"directory after inspecting {dispatch_dir}"
        )


def _finalize_results(
    output_dir: Path,
    model_path: Path,
    config_path: Path,
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    sample_path = output_dir / "samples.jsonl"
    result_path = output_dir / "results.jsonl"
    sample_csv_path = output_dir / "samples.csv"
    result_csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    _atomic_jsonl(sample_path, samples)
    _atomic_jsonl(result_path, samples)
    _write_csv(sample_csv_path, samples)
    _write_csv(result_csv_path, samples)
    summary = summarize(samples)
    summary.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "method": BASELINE_METHOD,
            "model": str(model_path),
            "output_dir": str(output_dir),
            "config": str(config_path),
            "samples_jsonl": str(sample_path),
            "samples_csv": str(sample_csv_path),
            "results_jsonl": str(result_path),
            "results_csv": str(result_csv_path),
            "manifest": str(manifest_path),
            "precompute_runtime_s": None,
        }
    )
    _atomic_json(summary_path, summary)
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "method": BASELINE_METHOD,
        "status": "complete",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_signature_sha256": run_config.get("run_signature_sha256"),
        "model": str(model_path),
        "datasets": list(run_config.get("datasets", ())),
        "workloads": dict(run_config.get("workloads", {})),
        "records": len(samples),
        "precompute_runtime_s": None,
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in (
                ("config", config_path),
                ("results_jsonl", result_path),
                ("results_csv", result_csv_path),
                ("samples_jsonl", sample_path),
                ("samples_csv", sample_csv_path),
                ("summary", summary_path),
            )
        },
    }
    _atomic_json(manifest_path, manifest)
    return summary


def main() -> None:
    args = parse_args()
    if args.worker_plan is not None:
        _worker_main(args.worker_plan.expanduser().resolve())
        return
    if args.samples_per_job < 1:
        raise ValueError("--samples_per_job must be positive")
    if args.generation_mode == "greedy" and args.samples_per_job != 1:
        raise ValueError(
            "greedy decoding is deterministic; use exactly one sample per job"
        )
    if not 0 < args.temperature or not math.isfinite(args.temperature):
        raise ValueError("--temperature must be positive and finite")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top_k must be nonnegative")

    dataset_names = _datasets(args)
    workload_paths = {
        "common_gen": args.common_gen_workload,
        "coauthor": args.coauthor_workload,
    }
    missing_workloads = [
        f"--{dataset}_workload"
        for dataset in dataset_names
        if workload_paths[dataset] is None
    ]
    if missing_workloads:
        raise ValueError(
            "current schema-v3 compiled workloads are required: "
            + ", ".join(missing_workloads)
        )
    workloads = {
        dataset: load_workload(
            workload_paths[dataset],
            expected_dataset=dataset,
        )
        for dataset in dataset_names
    }
    indices = _parse_indices(args.indices)
    source_selected: List[Dict[str, Any]] = []
    for dataset in dataset_names:
        source_selected.extend(
            select_jobs(
                workloads[dataset]["jobs"],
                args.selection,
                indices,
            )
        )
    model_path = args.model.expanduser().resolve()
    scoring_tokenizer, scoring_contract = _load_scoring_contract(
        model_path,
        local_files_only=bool(args.local_files_only),
    )
    selected, tokenization_metadata = _prepare_scoring_rows(
        source_selected,
        scoring_contract["tokenizer_fingerprint"],
    )
    args.terminal_token_ids = list(scoring_contract["terminal_token_ids"])
    args.scoring_contract = scoring_contract
    model_label = args.model_label.strip() or model_path.name
    model_slug = _model_slug(model_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ROOT
        / "results"
        / f"{model_slug}_{args.selection}_{len(selected)}jobs_{timestamp}"
    )
    if args.resume:
        if args.output_dir is None:
            raise ValueError("--resume requires an explicit --output_dir")
        if not output_dir.is_dir():
            raise ValueError("--resume output directory does not exist")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "schema_version": 1,
        "method": BASELINE_METHOD,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_label": model_label,
        "model_slug": model_slug,
        "model_artifact_sha256": _model_artifact_hashes(model_path),
        "tokenizer_file_sha256": _tokenizer_artifact_hashes(model_path),
        "scoring_contract": scoring_contract,
        "compiled_constraint_scoring": tokenization_metadata,
        "scorer_file_sha256": {
            str(ROOT / "constraint_baseline.py"): sha256_file(
                ROOT / "constraint_baseline.py"
            ),
            str(ROOT / "run_baseline_llm.py"): sha256_file(
                ROOT / "run_baseline_llm.py"
            ),
        },
        "datasets": dataset_names,
        "selection": args.selection,
        "indices": indices,
        "samples_per_job": args.samples_per_job,
        "generation_mode": args.generation_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "local_files_only": bool(args.local_files_only),
        "strict_evaluation": {
            "score_new_tokens_only": True,
            "terminal_eos_required": True,
            "accepted_terminal_token_ids": list(
                scoring_contract["terminal_token_ids"]
            ),
            "accepted_terminal_token_texts": list(
                scoring_contract["terminal_token_texts"]
            ),
            "length_intervals_including_terminal": [
                list(bounds)
                for bounds in sorted(
                    {
                        (
                            int(row["compiled_constraint"]["n_low"]),
                            int(row["compiled_constraint"]["n"]),
                        )
                        for row in selected
                    }
                )
            ],
            "constraint_evaluator": "compiled_token_partition_nfa_schema_v1",
            "authoritative_scorer": str(ROOT / "constraint_baseline.py"),
        },
        "workloads": {
            dataset: {
                "path": workloads[dataset]["_workload_path"],
                "file_sha256": workloads[dataset]["_workload_file_sha256"],
                "jobs_sha256": workloads[dataset]["jobs_sha256"],
                "kind": workloads[dataset]["kind"],
                "schema_version": workloads[dataset]["schema_version"],
            }
            for dataset in dataset_names
        },
        "selected_jobs": [
            {
                "dataset": row["dataset"],
                "job_id": row["job_id"],
                "design_index": row["design_index"],
                "constraint": row["constraint"],
                "k": row.get("k"),
                "t": row["t"],
                "keywords": keyword_surfaces(row),
                "tracked_keyword_token_ids": row["tracked_keyword_token_ids"],
                "separator_token_id": row["separator_token_id"],
                "compiled_constraint_sha256": row["compiled_constraint"]["sha256"],
                "instruction": render_constraint_instruction(
                    row,
                    scoring_tokenizer,
                ),
                "user_content": compose_user_content(
                    row,
                    scoring_tokenizer,
                ),
            }
            for row in selected
        ],
        "dry_run": bool(args.dry_run),
    }
    signature_payload = {
        key: value
        for key, value in run_config.items()
        if key not in {"created_at_utc", "runtime", "execution"}
    }
    run_config["run_signature_sha256"] = sha256_value(signature_payload)
    config_path = output_dir / "config.json"
    if args.resume:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            existing_config.get("run_signature_sha256")
            != run_config["run_signature_sha256"]
        ):
            raise ValueError("resume configuration does not match existing run")
        run_config = existing_config
    else:
        _atomic_json(config_path, run_config)
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "jobs": len(selected)}, indent=2))
        return

    tasks = _planned_tasks(selected, args.samples_per_job, args.seed)
    samples, by_key = _collect_samples(
        output_dir,
        tasks,
        run_config["run_signature_sha256"],
    )
    if len(by_key) == len(tasks):
        summary = _finalize_results(
            output_dir,
            model_path,
            config_path,
            samples,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.slurm_nodes:
        _dispatch_slurm(
            args,
            output_dir,
            tasks,
            set(by_key),
            run_config,
            config_path,
        )
        samples, by_key = _collect_samples(
            output_dir,
            tasks,
            run_config["run_signature_sha256"],
        )
        if len(by_key) != len(tasks):
            missing = len(tasks) - len(by_key)
            raise RuntimeError(
                f"distributed run ended with {missing} missing samples; "
                "resume the same output directory"
            )
        summary = _finalize_results(
            output_dir,
            model_path,
            config_path,
            samples,
        )
        print(json.dumps(summary, indent=2))
        return

    model, tokenizer, runtime = _load_model_runtime(args)
    run_config["runtime"] = runtime
    _atomic_json(config_path, run_config)

    sample_path = output_dir / "samples.jsonl"
    total = len(tasks)
    completed = len(samples)
    for task in tasks:
        row = task["row"]
        key = (
            str(row["dataset"]),
            str(row["job_id"]),
            int(task["sample_index"]),
        )
        if key in by_key:
            continue
        record = _build_record(model, tokenizer, task, args)
        samples.append(record)
        by_key[key] = record
        _append_jsonl(sample_path, record)
        completed += 1
        status = "PASS" if record["evaluation"]["strict_accepts"] else "FAIL"
        print(
            f"[{completed:03d}/{total:03d}] {row['job_id']} "
            f"sample={task['sample_index']} {status} "
            f"tokens={record['evaluation']['generated_total_token_count']} "
            f"time={record['generation_runtime_s']:.2f}s",
            flush=True,
        )

    samples, by_key = _collect_samples(
        output_dir,
        tasks,
        run_config["run_signature_sha256"],
    )
    summary = _finalize_results(
        output_dir,
        model_path,
        config_path,
        samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
