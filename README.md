# qsiv-runtime

A small local decision model ([QSIV](https://huggingface.co/djinn-san/QSIV)) that answers yes/no questions about a
piece of text with a probability each. This package runs it for you: one resident GPU service shared by every tool on
the machine, a stdlib-only client for your program, and `setup` / `doctor` commands. No key, no network at inference.

## Requirements

- Linux x86_64
- An NVIDIA GPU with compute capability 8.9 or higher (FP8); the `reference` runtime needs 8.0 or higher
- [uv](https://docs.astral.sh/uv/) and Python 3.12 for the isolated GPU environment that `setup` creates
- Several GB of disk and network access for `setup` only (never at inference)
- The client itself needs only Python 3.10 or newer and the standard library

## Install

```bash
uv tool install qsiv-runtime        # or: pip install qsiv-runtime
qsiv-runtime setup --dry-run        # print what would be created and downloaded
qsiv-runtime setup                  # create the GPU environment, download the pinned weights, verify the checksum
qsiv-runtime doctor                 # one line per check; exit code 0 only when everything is ready
```

`setup` is the only code that downloads anything. It refuses before any download on an unsupported machine, verifies
the pinned SHA-256 of the weights, and writes an install marker only when everything succeeded. Default locations are
`~/.local/share/qsiv-runtime/model` and `~/.local/share/qsiv-runtime/venv` (an existing agemem install under
`~/.local/share/agemem/` is used when the defaults do not exist).

## Python API

```python
from qsiv_runtime import QSIVClient, QSIVUnavailable

questions = [
    "Does the script escalate privileges?",
    "Does the script modify files outside its own directory?",
]
try:
    with QSIVClient() as client:
        probabilities = client.decide('{"script": "rm -rf /var/lib/app"}', questions)
except QSIVUnavailable as exc:
    # exc.message is user-readable; exc.diagnostic.cause is one of qsiv_runtime.Cause
    print(exc.message, exc.diagnostic.hint)
else:
    flagged = [q for q, p in zip(questions, probabilities) if p >= 0.5]
```

`decide(state, questions)` returns one probability in [0, 1] per question (at most 8), in order. The client is lazy:
nothing happens until the first call, then it checks the machine, connects to the shared service and starts it if
absent. Every failure is a `QSIVUnavailable` (never a traceback) with a `diagnostic.cause`: `no_gpu`,
`not_installed`, `startup_failed`, `service_exited`, `model_in_use`, `invalid_answer`, `unsafe_directory`,
`bad_checksum`, `too_large`, `timeout`. Treat any of them as "not reviewed" (fail closed).

Timeouts: service startup 120 s, idle exit 300 s, one request 60 s. A service that exits mid-request is restarted
once; nothing else is retried.

Environment overrides: `QSIV_RUNTIME_MODEL_DIRECTORY`, `QSIV_RUNTIME_PYTHON`, `QSIV_RUNTIME_RUNTIME` (`fp8` or
`reference`), `QSIV_RUNTIME_SOCKET_DIRECTORY`, `QSIV_RUNTIME_IDLE_SECONDS`. Precedence: constructor arguments, then
environment, then the package defaults, then the legacy agemem location.

## Privacy and trust

- The text you send goes only to a local process over a Unix socket in `/tmp/qsiv-runtime-<uid>` (mode 0700, owned by
  you, never a symlink). Nothing is sent to a network service and inference never downloads.
- The service is started with Hugging Face offline mode set and without these environment variables
  (case-insensitive): any name containing `API_KEY`, `APIKEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`,
  `CREDENTIAL`, `PRIVATE_KEY` or `ACCESS_KEY`, or with a whole `_`-, `-`- or `.`-separated segment equal to `AUTH`
  (`SSH_AUTH_SOCK`; `AUTHOR` stays). Everything else (`PATH`, `HOME`, ...) is passed on, except `PYTHONPATH`: the service gets only a directory holding a symlink to this package, never the caller's `PYTHONPATH`.
- The GPU environment is an ordinary virtualenv; it is not sandboxed. Inputs are never truncated. QSIV 1.0 takes at
  most 65,536 tokens per request (`qsiv.constants.MAX_CONTEXT`: the state plus every templated question block,
  counted with the model's own tokenizer); a longer request is rejected with `too_large` before inference. On a GPU
  with little free memory a long request can fail earlier with `startup_failed` ("QSIV service failed", CUDA out of
  memory in the service log): on an 8 GB laptop GPU a 49,143-token state answered and a 49,175-token one ran out of
  memory (it depends on the free GPU memory). A decision-model signal can dilute for very long inputs (remote-code probability 0.95 at
  ~300 tokens, 0.13 at ~47K), so keep your own static checks and never treat a safe answer as clearance. See [docs/manual-gpu-check.md](https://github.com/illescasDaniel/qsiv-runtime/blob/main/docs/manual-gpu-check.md).
- `runtime: reference` (`QSIV_RUNTIME_RUNTIME=reference`) relies on a model directory you place manually; `setup`
  does not install it and its weights are not pinned or checksummed.
- The client validates every answer as a whole (a list of the right length of real numbers in [0, 1]; booleans,
  strings and NaN are rejected). Callers should still treat the result as untrusted data.

## Relation to agemem and aiscript

The service code is adapted from [agemem](https://github.com/illescasDaniel/agemem), which will later depend on this
package instead of its own copy; until then both share one GPU model by taking the same model lock.
[aiscript](https://github.com/illescasDaniel/aiscript) can use it as a local safety reviewer
(`reviewer_provider: qsiv`).

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest -q
```

Tests use fakes only (no GPU, no weights, no network). The live check on a real GPU is manual:
[docs/manual-gpu-check.md](https://github.com/illescasDaniel/qsiv-runtime/blob/main/docs/manual-gpu-check.md).

## License

MIT. QSIV (Apache-2.0) is downloaded by `setup`, not bundled; see [THIRD_PARTY.md](https://github.com/illescasDaniel/qsiv-runtime/blob/main/THIRD_PARTY.md).
