# Design: explore_source_code → free-form bash command

Date: 2026-09-07
Branch: `feature/context-reducer-0.37`

## Problem

The `explore_tool` ("explore_source_code") injection currently uses a rigid
`{path, focus_question, start_line, end_line}` schema. The proxy *synthesizes* a
`sed -n 'start,endp' path` command on the way out to the agent:

- OpenAI Responses (Codex): rewritten to `exec_command` with a generated `sed`.
- Anthropic Messages (Claude Code): rewritten to the `Read` tool with a derived
  `offset`/`limit` window.

This forces the model into a line-window mental model and hides the actual shell
semantics behind proxy-generated commands. We want the model to author its own
read-only bash command, pass it straight through to the agent, and be told (and
be able) to use this tool **only when reading code files**.

## Goal

Replace the forced `sed` rewrite (and the `path`/`start_line`/`end_line`
line-window) with a **free-form bash `command`** authored by the model, plus a
required `focus_question`. The command is forwarded verbatim to the agent's
shell. The injected instructions make clear the tool is for reading code files
only.

## Design

### 1. Tool schema (`headroom/proxy/explore_pruner/focus.py`)

`explore_source_code` parameters become:

- `command: string` (required) — a read-only bash command that prints source
  text (e.g. `sed -n '40,120p' src/app.py`, `rg -n "def auth" -A 30 src/`,
  `head -n 50 foo.py`).
- `focus_question: string` (required) — unchanged semantics: a complete
  interrogative question stating what the model is trying to understand.

Drop `path`, `start_line`, `end_line`. Keep `additionalProperties: false`,
`strict: true`, and the Anthropic `input_schema` mirror.

### 2. Outbound rewrite — pass command through verbatim

- `rewrite_explore_call_to_exec` (OpenAI/Codex): emit `exec_command` with
  `{"cmd": command}` using the model's command **as-is**. No `sed` synthesis, no
  window clamping. Still returns `(item, focus, source)`; focus is the
  `focus_question` (truncated by `focus_max_chars`). Invalid args (missing
  command/focus) return the item unchanged as before.
- Rename `rewrite_explore_call_to_read` → `rewrite_explore_call_to_bash`
  (Anthropic/Claude Code): emit a `Bash` tool_use with
  `{"command": command}` instead of `Read`. Claude Code's Bash tool input key is
  `command` (already confirmed in `headroom/learn/models.py`).

### 3. Inbound restore (`service.py` + `focus.py`)

Restore client history back to `explore_source_code` so the model sees a
consistent tool name, keyed off the store:

- OpenAI: `_restore_explore_call_item` matches `exec_command`; restore via a new
  `restore_explore_function_call(command, focus_question)`.
- Anthropic: `_restore_explore_anthropic_block` switches its match from `Read`
  → `Bash`; restore via a new `restore_explore_tool_use(command, focus_question)`.

Restore payload is `{command, focus_question}`.

### 4. Store (`store.py`)

Drop `explore_path`, `explore_start_line`, `explore_end_line` from
`ExplorePrunerRecord`. The record carries `commands: list[str]` (the verbatim
command) and `focus_question: str | None`. Remove the keep-fields merge logic
for the dropped fields.

### 5. Instructions (`focus.py`)

Rewrite `EXPLORE_TOOL_INSTRUCTIONS`:

1. **Scope gate (new, emphasized):** use `explore_source_code` **only to read /
   inspect code files**. Never for edits, tests, installs, git, or general shell
   work — those go through the normal shell tool.
2. **Command authoring:** provide a read-only bash command that prints source
   text; the gateway forwards it verbatim to the shell. Prefer locating lines
   with `rg`/`grep` first, then read a bounded window.
3. **Bounded output:** keep output to a few hundred lines to avoid truncation;
   re-call with a narrower window if truncation markers appear.
4. **`focus_question`:** unchanged — real interrogative, no keep-lists, no line
   numbers, no symbol names.

### 6. Pruning eligibility (unchanged, best-effort)

`has_supported_source_file_scope(commands)` keeps parsing the command string for
source-file suffixes; a read command targeting `.py/.ts/.go/…` remains prunable,
non-file commands are simply not pruned (fail-safe). `strip_shell_line_numbers`
still strips `cat -n`/`nl` prefixes before reduction.

### 7. Streaming rewrite (`stream_rewrite.py`)

- `_looks_like_explore_args` now matches `command` + `focus_question` (was
  `path` + `focus_question` + line fields).
- `ExploreAnthropicStreamRewriter` renames `explore_source_code` blocks to
  `Bash` (was `Read`) and emits `{"command": ...}` input JSON.

### 8. Config

- `focus_max_chars`: kept (truncates `focus_question`).
- `explore_max_lines` + `clamp_explore_window` + `HEADROOM_EXPLORE_MAX_LINES`:
  kept as a **deprecated no-op** to avoid breaking existing configs. The service
  constructor drops the `explore_max_lines` parameter (default `400` remains in
  the config model but is no longer consumed).

## Files touched

- `headroom/proxy/explore_pruner/focus.py` — schema, instructions, rewrite/
  restore/parse helpers.
- `headroom/proxy/explore_pruner/service.py` — outbound/inbound orchestration.
- `headroom/proxy/explore_pruner/store.py` — record fields.
- `headroom/proxy/explore_pruner/stream_rewrite.py` — streaming renames.
- `headroom/proxy/explore_pruner/__init__.py` — export updates.
- `headroom/proxy/explore_pruner/factory.py` — stop passing `explore_max_lines`.
- `tests/test_explore_pruner.py` — update rewrite/restore/store assertions.

## Error handling / fail-open

- Missing/invalid `command` or `focus_question` → leave the call untouched
  (same as today's invalid-args path), log `invalid_args`.
- Non-file command → not pruned (reducer never called), output forwarded
  verbatim.
- Reducer failure → existing `fail_open` path unchanged.

## Testing

- Unit: rewrite `explore_source_code` → `exec_command {"cmd": <command>}` /
  `Bash {"command": <command>}`; restore round-trips; store no longer carries
  path/line fields.
- Unit: scope gate — instructions text asserts "only … read code files".
- Unit: `_looks_like_explore_args` matches the new arg shape.
- Regression: existing prune/anthropic tests updated to the new schema.
