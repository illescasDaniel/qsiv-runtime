# Changelog

## [0.1.0] - 2026-11-13

First release.

- `QSIVClient.decide(state, questions)`: one validated probability per question from a shared local QSIV service
  (stdlib-only client, JSON lines over a private Unix socket, config handshake, protocol 1).
- `QSIVUnavailable` with ten causes: `no_gpu`, `not_installed`, `startup_failed`, `service_exited`,
  `model_in_use`, `invalid_answer`, `unsafe_directory`, `bad_checksum`, `too_large`, `timeout`.
- `qsiv-runtime setup` and `qsiv-runtime doctor`; `python -m qsiv_runtime.service`.
- Reads an existing agemem install as a fallback.
- Input over QSIV's 65,536-token request limit (counted with the model's own renderer and tokenizer) is rejected
  as `too_large` before inference; checked on a real GPU (see `docs/manual-gpu-check.md`).
- The service environment drops names containing `API_KEY`, `APIKEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`,
  `CREDENTIAL`, `PRIVATE_KEY` or `ACCESS_KEY`, or with a whole segment `AUTH` (`SSH_AUTH_SOCK`; not `AUTHOR`). The caller's `PYTHONPATH` is not passed on.
- On an 8 GB GPU a state above about 49K tokens can run out of CUDA memory below the limit; the client reports it as
  `startup_failed`. A very long script dilutes the QSIV signal (remote code 0.95 at ~300 tokens, 0.13 at ~47K), so
  keep your own static checks.
