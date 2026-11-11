# Upstream Integration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge Aratako/main while preserving the fork's cancellation, model selection, Windows launchers, device/precision settings, and browser launch behavior.

**Architecture:** Keep upstream's batched inference structure. Carry the optional `cancel_check` callback through single/batch synthesis, samplers, decoding, and every Length Check pass. A callback cancels its entire synthesis call, including all requests in that batch; Gradio continues submitting one request per call.

**Tech Stack:** Python, PyTorch, Gradio, unittest.

**Spec:** User-approved integration direction in this chat: retain fork functionality and incorporate upstream updates.

## Global Constraints

- Keep dependencies and Windows launchers unchanged unless integration requires otherwise.
- Preserve session-local cooperative cancellation and model reuse after stopping.
- Keep upstream defaults: rare-token splitting enabled, Length Check disabled.
- Work on `integrate/upstream-20261111`, validate, then integrate locally into main. Do not push.

## Review Focus

- Cancellation during each Length Check pass must escape without retrying and remove attention hooks.
- A cancelled batch must release the inference lock and leave the runtime reusable.
- Sequential decoding must stop before decoding subsequent candidates.
- Both Gradio UIs must retain Stop, model choices, and runtime options.
- Existing calls without a cancellation callback must retain upstream behavior.

## Task 1: Merge and preserve cancellation

**Files:** gradio_app.py, irodori_tts/{inference_runtime,rf,meanflow,length_check}.py; tests/test_generation_cancellation.py; tests/test_upstream_integration.py.

**Interfaces:** Optional `cancel_check: Callable[[], None] | None = None` on runtime synthesis and samplers; optional callback on Length Check and its Gradio wrapper.

- [x] Verify the existing test suite on the initial merged scaffold (the sandbox baseline process stalled; normal-permission execution passed all 5 existing tests).
- [x] Merge upstream without committing; retain both sampler argument additions and both Gradio inputs, and use upstream runtime structure.
- [x] Add regression tests for cancellation through batch synthesis, decoding, and Length Check; run to demonstrate the missing integration.
- [x] Thread cancellation checks through the new runtime and all Length Check passes; preserve sampler step checks.
- [x] Run unittest discovery, relevant lint/compile checks, and inspect the diff for preserved fork features.
- [x] Review the final integration and prepare the verified merge for local main.

## Progress

- Latest upstream fetched; integration branch created from ff31558.
- RED: the extended Gradio test failed with Length Check enabled; runtime/Length Check regression cases reproduced missing cancellation interfaces.
- GREEN: all 10 tests passed after callback integration, including single/batch RF and MeanFlow, sequential decoding, all Length Check passes, hook cleanup, runtime reuse, and both Gradio UIs.
- Validation: compileall passed; infer.py and both Gradio entry points load their help successfully; Ruff passed for all modified implementation files and the new regression test.
- Existing tests/test_generation_cancellation.py has 11 pre-existing Ruff findings (confirmed against ff31558); no new findings were introduced there.
- Preserved files were compared against ff31558: model choices, Windows launchers, VoiceDesign UI, cancellation helpers, runtime option helpers, pyproject.toml and uv.lock are unchanged.
- Independent final review: no actionable issues found. No real-checkpoint/GPU audio quality validation was performed.
- Integration choice: batch cancellation stops the entire synthesis call; Gradio continues making one-request calls, preserving session-local Stop behavior.
- Local completion: commit the merge and fast-forward main; leave origin untouched.
