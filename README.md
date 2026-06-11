# claw-swe-bench

A unified adapter framework for evaluating agent harnesses ("claws") on
[SWE-bench](https://www.swebench.com/). One command runs any supported claw on
SWE-bench Verified or SWE-bench Multilingual, with identical prompting, patch
collection, and evaluation across all of them — so the harness becomes a
controlled variable and results are comparable harness-to-harness.

This repository is the reference implementation for the technical report
[Claw-SWE-Bench: A Benchmark for Evaluating OpenClaw-Style Agent Harnesses on
Coding Tasks](https://arxiv.org/abs/2606.12344). The full benchmark is 350 GitHub issue-resolution
instances across 8 languages (300 from SWE-bench Multilingual + 50 from
SWE-bench Verified-Mini); the paper also defines an 80-instance **Lite** subset
for low-cost iteration (see *Instance lists* in Notes).

Supported claws:

| Claw | Runtime | How it gets into the container | Model selection |
|---|---|---|---|
| `openclaw` | Node.js CLI | bind-mount node + module + `~/.openclaw` | `--model` (per-agent) |
| `hermes` | Python venv | bind-mount standalone Python + venv | `--model` + `claw_configs/hermes/config.yaml` providers |
| `nanobot` | Python venv | bind-mount standalone Python + venv | `claw_configs/nanobot/config.json` |
| `zeroclaw` | single Rust binary | bind-mount binary | `claw_configs/zeroclaw/config.toml` |
| `generic` | Python repo ([lsdefine/GenericAgent](https://github.com/lsdefine/GenericAgent)) | bind-mount repo + venv | `--llm_no` index into `claw_configs/generic/mykey.py` |

## Design

```
run_infer.py ──► orchestrator ──► SWEBenchWorkspace (Docker)  ── claw-agnostic
                     │                    │
                     └────────► BaseClawAdapter hooks ──────── claw-specific
                                (claw_swebench/claws/)
run_eval.py  ──► official SWE-bench harness (separate venv)
```

- **Claw-agnostic core** (`claw_swebench/`): dataset loading, container
  lifecycle, repo preparation, prompt rendering, runner-side patch
  collection/cleaning, predictions/state persistence, harness evaluation.
- **Claw adapters** (`claw_swebench/claws/`): each adapter implements
  `BaseClawAdapter` — extra `docker run` args (mounts), post-start
  provisioning, agent lifecycle, task launch, session backup, usage
  collection. Adding a claw = one new file + one registry entry.

Key fairness/contamination properties, enforced for every claw:

- **Same prompt.** All claws use `prompts/default.txt` (a phase-by-phase
  long prompt). The only allowed override is tool-name guidance
  (`prompts/generic.txt` adds 3 lines for GenericAgent's tool names and
  bans its web tools; everything else is identical).
- **No network answers.** The prompt forbids network use; OpenClaw
  additionally gets a 13-tool deny list (web/memory/session/cron tools);
  NanoBot's web tools are disabled in config; ZeroClaw's traffic goes
  through a tool-filtering proxy.
- **Future-commit stripping.** The official Multilingual images retain the
  fix commit in git history (`git log --all` leaks the gold patch). Every
  workspace strips future tags/commits, expires reflogs, and GCs before the
  agent starts, then asserts zero future commits remain.
- **Runner-side patch collection.** The patch is always `git diff` taken by
  the runner after the agent exits — never agent-reported. Setup/lock-file
  and binary diffs are stripped (`patch.py`).
- **Per-instance isolation.** One fresh container per instance (pids/memory
  limited); OpenClaw additionally gets a throwaway agent per instance.

## Setup

### 1. Host requirements

- Docker with prebuilt SWE-bench instance images
  (`sweb.eval.x86_64.<instance_id>:latest`, or SWE-agent's
  `swebench/sweb.eval.x86_64.<id>` naming — both are auto-detected).
- Python 3.10+ with `pip install -r requirements.txt`.
- The official SWE-bench harness installed in its own venv for evaluation
  (default `/data/swe-bench-env`, override with `SWEBENCH_VENV`).

### 2. Install the claw(s) you want to run

Each claw's runtime lives on the host and is bind-mounted read-only into the
eval containers. Defaults (all overridable via env vars, see
`claw_swebench/config.py`):

| Env var | Default | Used by |
|---|---|---|
| `CLAW_PYTHON_HOME` | standalone Python 3.12 home (e.g. the `uv python install 3.12` location) | hermes, nanobot, generic |
| `OPENCLAW_NODE_BIN` / `OPENCLAW_MODULE_DIR` / `OPENCLAW_STATE_DIR` | `/usr/bin/node` / `/usr/lib/node_modules/openclaw` / `~/.openclaw` | openclaw |
| `HERMES_ENV_PATH` | `/opt/hermes-env` | hermes |
| `NANOBOT_ENV_PATH` | `/opt/nanobot-env` | nanobot |
| `ZEROCLAW_BIN` | `/usr/local/bin/zeroclaw` | zeroclaw |
| `GA_REPO_PATH` / `GA_ENV_PATH` | `/opt/genericagent` / `/opt/genericagent-env` | generic |

The standalone Python (`uv python install 3.12`) is required because the
SWE-bench images don't ship a usable Python 3.12; the venvs must be created
with that interpreter so they run inside any container.

### 3. Configure the claw

Copy the example config and fill in your API keys (real config files are
gitignored):

```bash
cp claw_configs/hermes/config.yaml.example   claw_configs/hermes/config.yaml
cp claw_configs/nanobot/config.json.example  claw_configs/nanobot/config.json
cp claw_configs/zeroclaw/config.toml.example claw_configs/zeroclaw/config.toml
cp claw_configs/generic/mykey.py.example     claw_configs/generic/mykey.py
```

`hermes` and `generic` also read API keys from the host environment
(`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`,
`DASHSCOPE_API_KEY`, …) — these are forwarded into the container
automatically. OpenClaw uses its own credential store (`~/.openclaw`).

## Run

Inference (patch generation):

```bash
python3 run_infer.py \
    --claw openclaw \
    --dataset multilingual \
    --run_id openclaw-multi-1 \
    --instance_file config/multilingual_300_instances.txt \
    --timeout 3600
```

`--timeout 3600` is the standard per-instance budget used in our
evaluations (also the built-in default — spelled out here so runs are
reproducible even if defaults change).

- `--claw {openclaw,hermes,nanobot,zeroclaw,generic}` — which harness.
- `--dataset {verified,multilingual}` — loads `config/<dataset>.yaml`.
- `--model`, `--timeout`, `--max_turns` — override per-claw defaults
  (`CLAW_DEFAULTS` in `config.py`). For nanobot/zeroclaw the model lives in
  the claw's own config file; `--model` is recorded as metadata.
- `--llm_no N` — generic only: selects the Nth provider in `mykey.py`.
- `--workers N` — parallel instances (each in its own container).
- Re-running the same `--run_id` resumes (skips completed instances);
  `--no_resume` disables that.

Artifacts land in `artifacts/<run_id>/`: per-instance `prompt.txt`,
`agent_stdout.log` / `agent_stderr.log`, session logs, `git.patch`,
`metadata.json` (incl. token usage where available), plus shared
`predictions.jsonl` and `state.jsonl`.

Evaluation (official harness):

```bash
python3 run_eval.py \
    --predictions artifacts/openclaw-multi-1/predictions.jsonl \
    --dataset_name SWE-bench/SWE-bench_Multilingual \
    --run_id openclaw-multi-1
```

Use a distinct `--run_id` per claw/run so harness logs don't collide.

## Adding a new claw

1. Create `claw_swebench/claws/<name>.py` implementing `BaseClawAdapter`:
   - `container_run_args(instance_id)` — bind mounts for your runtime;
   - `send_task(...)` — launch the agent in the container, return `AgentResult`;
   - optionally `post_container_start`, `create_agent`/`delete_agent`,
     `backup_session`, `collect_usage`, `prompt_template`.
2. Register the class in `claw_swebench/claws/__init__.py` and add defaults
   to `CLAW_DEFAULTS` in `config.py`.
3. Keep the default prompt. Only add tool-name guidance if your claw's tools
   genuinely need it, and keep the rest of the template byte-identical.

## Notes

- **Resource limits**: every container runs with `--pids-limit 300
  --memory 8g` (override via `CLAW_PIDS_LIMIT` / `CLAW_CONTAINER_MEMORY`).
- **Proxies** (`proxies/`, `claw_configs/zeroclaw/tool_filter_proxy.py`):
  optional host-side HTTP proxies for accurate cache/usage accounting on
  providers that under-report it (e.g. DashScope `cached_tokens`). See the
  file headers for details.
- **Instance lists**: `config/multilingual_300_instances.txt` and
  `config/verified_mini_50.txt` together form the 350-instance full set. The
  80-instance Lite subset is selected by the cost-aware, rank-aware procedure
  described in the paper.
