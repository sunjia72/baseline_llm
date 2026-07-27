# Local instruction-model constraint baseline

This folder measures whether an unconstrained local instruction model follows
compiled token-level constraints when each constraint is described in natural
language. Generation itself is ordinary greedy or sampled decoding: there is
no logit mask, guided decoding, rejection sampling, or witness.

## Boundary with the benchmark compiler

The baseline is a consumer, not a constraint compiler. It accepts only current
`compiled_keyword_dataset` workload schema v3 files whose jobs embed
`compiled_token_partition_nfa` schema v1 artifacts. Each artifact contains:

- the compact NFA transitions and wildcard transitions;
- the token-ID partition, default symbol, and terminal STOP symbol;
- generation length bounds;
- the tokenizer fingerprint;
- the human-facing instruction rule; and
- a canonical SHA-256 digest.

`constraint_baseline.py` validates and interprets that generic schema directly.
It does not import the NFA-FPRAS project, rebuild an automaton, retokenize a
keyword, or contain executable knowledge of benchmark family names.

The runtime tokenizer must match the compiled fingerprint (vocabulary,
special-token IDs, and tokenizer-file hashes). A workload compiled for Qwen
therefore cannot silently be scored with Gemma or another tokenizer; compile a
new workload for that tokenizer first.

When an authenticated rule refers to a separator, the prompt also names its
visible phrase. The renderer obtains the separator token ID and text from the
workload, verifies the text by decoding that ID with the fingerprint-matched
runtime tokenizer, and adds the clarification without changing the workload or
compiled-constraint hashes. This is generic artifact rendering; it does not
dispatch on constraint-family names.

## Evaluation contract

- The ordered workload job list and every embedded automaton are authenticated.
- The artifact's instruction rule is added to the original dataset prompt in
  one user turn, with thinking disabled.
- Only newly generated token IDs are scored; prompt tokens never count.
- A strict success ends in exactly one accepted model-native terminal token,
  has a total length within the artifact bounds, contains no other special
  token, and is accepted by the embedded NFA.
- Phrase occurrence locations remain diagnostic metadata. The authoritative
  decision is the compiled NFA, including explicit and wildcard transitions.
- Semantic acceptance, terminal validity, length validity, and strict
  acceptance are reported separately.

## Inputs and usage

Workload paths are intentionally required. Historical schema-v1 result
workloads do not embed authenticated compiled artifacts and are rejected with
a migration error. Use schema-v3 CommonGen and CoAuthor workloads produced by
the current NFA benchmark preparation step.

For the default balanced smoke run (one midpoint job per family and dataset):

```bash
cd /project/aip-ksmeel/sunjia72/constraint_decoding/baseline_llm
conda activate nfa

python run_baseline_llm.py \
  --dataset both \
  --common_gen_workload /path/to/common_gen_compiled_v3/workload.json \
  --coauthor_workload /path/to/coauthor_compiled_v3/workload.json \
  --selection midpoint_per_family
```

To validate inputs and save rendered instructions without loading the model:

```bash
python run_baseline_llm.py \
  --dataset common_gen \
  --common_gen_workload /path/to/common_gen_compiled_v3/workload.json \
  --dry_run
```

For all jobs with stochastic decoding:

```bash
python run_baseline_llm.py \
  --dataset both \
  --common_gen_workload /path/to/common_gen_compiled_v3/workload.json \
  --coauthor_workload /path/to/coauthor_compiled_v3/workload.json \
  --selection all \
  --samples_per_job 3 \
  --generation_mode sample \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 20
```

An existing Slurm allocation can run one persistent worker per node/GPU pair:

```bash
python -u run_baseline_llm.py \
  --dataset both \
  --common_gen_workload /path/to/common_gen_compiled_v3/workload.json \
  --coauthor_workload /path/to/coauthor_compiled_v3/workload.json \
  --selection all \
  --slurm_nodes allocation \
  --slurm_gpus 0,1,2,3 \
  --cpus_per_worker 8 \
  --output_dir results/qwen35_all_compiled_v3
```

Canonical completed-run outputs are:

- `manifest.json`: run identity, completion status, workload references, and
  SHA-256 identities for the result artifacts;
- `results.jsonl`: one schema-v1 record per successful model invocation;
- `results.csv`: the compact tabular form of those records;
- `summary.json`: aggregate, per-dataset, and per-constraint statistics; and
- `config.json`: the authenticated run plan and provenance.

Every result identifies `method=baseline_llm`, its compiled constraint, design
index, compact and length-unrolled NFA sizes, and execution
`status=success`. That status means model invocation completed; whether its text
satisfies the constraint remains the independent
`evaluation.strict_accepts` field. An unconstrained LLM has no precomputation
phase, so `precompute_runtime_s` is JSON `null` (and blank in CSV), never zero.

`samples.jsonl` and `samples.csv` are retained as byte-identical compatibility
aliases. During a run, `samples.jsonl` also remains the resumable checkpoint:
resume an interrupted explicit output directory with the same arguments plus
`--resume`, and completed sample keys are skipped.

## Tests

```bash
conda activate nfa
python -m pytest -q test_constraint_baseline.py
```
