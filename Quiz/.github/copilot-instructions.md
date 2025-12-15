Purpose
=======

This repository is a small, single-file PyTorch quiz/exercise. The goal of this document is to give AI coding agents immediate, actionable context so they can be productive editing or improving the code without guessing project conventions.

Key facts
---------
- **Main entry:** `pytorch_quiz.py` — a short educational script that demonstrates a gradient-descent-style optimization.
- **Project type:** single-script learning exercise (no package layout, no tests, no CI configs present).
- **Dependencies:** PyTorch (imported as `torch` in the script). There is no `requirements.txt` in the repo, so assume `pip install torch` (or the appropriate CUDA wheel) is required to run.

Big picture / intent
--------------------
- This is a pedagogical file: placeholders and questions are intentionally present (comments like "Research this!"). Changes should be minimal and explicit so a human student or grader can follow them.
- The script computes and prints a final line intended for automated/manual grading on Moodle: `print(f"Minimum at x = {x.item():.4f}")`. Preserve that exact output format unless instructed otherwise.

Project-specific patterns and conventions
----------------------------------------
- Placeholders: incomplete method calls are indicated with comments and underscores (e.g. `# y.___________`, `# optimizer._________`). Replace these with the appropriate PyTorch API calls rather than refactoring surrounding code.
- Minimal edits: prefer completing the placeholders and leaving surrounding instructional comments intact. This repository is used for learning; large refactors are unexpected.
- Logging / output: progress is printed inside the training loop every 10 steps. Keep prints readable and numeric formatting consistent with existing `f"...:.4f"` usage.

Examples from the codebase
--------------------------
- Missing optimizer zeroing: the script shows `optimizer.zero` — the correct API is `optimizer.zero_grad()`.
- Missing backward call: the line `# y.___________` expects `y.backward()` (after `y` is defined).
- Missing optimizer step: the line `# optimizer._________` expects `optimizer.step()`.

Developer workflows (how to run & debug)
--------------------------------------
- Quick run (CPU):

  ```bash
  python3 pytorch_quiz.py
  ```

- Recommended reproducible env (local dev):

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch
  python3 pytorch_quiz.py
  ```

- If you add dependencies, also add a `requirements.txt` with pinned packages (e.g. `torch`) so other contributors can reproduce the environment quickly.

What an AI agent should do first
--------------------------------
1. Open `pytorch_quiz.py` and locate commented placeholders named with underscores or obvious typos (e.g. `optimizer.zero`).
2. Complete placeholders using the canonical PyTorch calls: define `y` (one of the commented functions), call `y.backward()`, call `optimizer.step()`, and ensure gradients are zeroed with `optimizer.zero_grad()`.
3. Run the script locally to verify output and ensure `Minimum at x = ...` prints as expected.

When to be conservative vs. proactive
-------------------------------------
- Be conservative: avoid restructuring the file, renaming the final print, or removing instructional comments unless the user explicitly asks for a rewrite.
- Be proactive: fix obvious API mistakes (typos or missing method calls) and add a `requirements.txt` if you add functionality that requires extra packages.

Files to reference
------------------
- `pytorch_quiz.py` — core script, contains the placeholders and the expected Moodle output.

Questions / clarifications to request from a human
-------------------------------------------------
- Should the final printed format remain exactly `Minimum at x = {x:.4f}` for grading? (Default: yes.)
- If adding dependencies (e.g. `requirements.txt`), should a CPU-only `torch` wheel be used or should we document CUDA variants?

If anything in this file is unclear, ask the repo owner for the grading expectations and allowed edits before making large changes.
