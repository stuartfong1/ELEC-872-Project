# Execution evidence

Machine-readable actual commands, inputs/parameters, UTC timestamps, exit codes,
wall time, and log destinations are in `reports/command_runs.jsonl`. These are
written only by executed steps, not inferred from source code. `logs` are local
artifacts excluded from Git, and remain in this workspace.

Successful steps are distinguished from unsuccessful/setup/interrupted attempts.
Initial independent `.venv` installation failed once due to an old pip resolver
and the CPU index's typing-extensions metadata, then was retried. That retry was
stopped during large-package installation because of setup time and memory/I/O
pressure. `.venv-runtime` setup was also interrupted; `.venv-run` is the actual
verified environment and does not modify global packages. Dependency resolution
was checked and corrected locally; see `logs/dependency-fix*.log`.

The full-decode inventory attempt was interrupted before completion; its partial
work is **not** included in the coverage count. The completed inventory provides
full file-list counts. `decode-audit` reports the actual bounded audit count, and
`prepare-*` reports the actual source-image coverage of each experiment. These
sets can overlap and must not be blindly added.

Initial reference preparation was visually reviewed; excluded sources and reasons
are in `configs/excluded_sources.json`. Final smoke preparation and later model
execution use the reviewed manifest. `reports/reference_review.json` separates
visual crop review from independent identity-ground-truth verification.

Recommended exact verification entry point:

```powershell
.\.venv-run\Scripts\python.exe scripts/execute_verified.py
```

Read `README.md` for acquisition, setup, individual CLI commands and resume.

The first shared stage logger recorded its own original script name even when
called from the feature/pilot entrypoints. `scripts/correct_execution_ledger.py`
corrects those commands to the observed launch invocations, preserving the
original values and correction reasons. No result, exit code, parameter or
timing is changed. Corrections are separately listed in
`reports/execution_ledger_corrections.json`. New stage records use actual argv.
