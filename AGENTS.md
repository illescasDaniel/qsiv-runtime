# qsiv-runtime

Stdlib-only client, shared GPU service, `setup` and `doctor` for the QSIV model. Source in `src/qsiv_runtime/`
(src layout), tests in `tests/` (fakes only: no GPU, no network, no weights).

## Quality gate (all must pass)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -q
```

## Rules

- The client modules (`client`, `_protocol`, `_preflight`, `paths`, `errors`, `_ports`, `service`, `setup`,
  `doctor`, `cli`) import only the standard library: the service is started through `PYTHONPATH` inside another
  virtualenv. `torch` and `qsiv` are imported lazily inside `service.load_engine`.
- Tests are named `test_given_..._when_..._then_...` with `# given / # when / # then`; scenario ids (S1..S31,
  from the aiscript SDD page `sdd/qsiv-provider.html`) are in the docstrings.
- Never download weights or touch the network in tests; `setup()` is only tested with a fake runner.
- Do not commit, tag or publish without an explicit go-ahead (release is a manual workflow, see `.github/`).
