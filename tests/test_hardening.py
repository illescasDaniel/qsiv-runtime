"""Fix round 2: size limit before inference, launch cwd, preflight order, doctor, error mapping."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from qsiv_runtime import Cause, QSIVUnavailable, doctor, setup
from qsiv_runtime._ports import SubprocessLauncher, SubprocessRunner
from qsiv_runtime.paths import ENV_RUNTIME, ENV_SOCKET_DIRECTORY, INSTALL_MARKER
from qsiv_runtime.service import check_input_size, native_engine
from tests.conftest import Install
from tests.fakes.fake_machine import FakeMachine
from tests.fakes.fake_service import FakeEngine, Generation, ScriptedLauncher, ThreadServiceLauncher
from tests.helpers import QUESTIONS, make_client
from tests.test_setup_doctor import FakeRunner

PLANT = "import sys\nopen(sys.argv[1] + '.planted', 'w').write('x')\n"
PROBE = "import json, sys\nopen(sys.argv[1], 'w').write(json.__file__)\n"


class WordCountingEngine(FakeEngine):
    """A fake tokenizer: one token per word of the state and of each templated question block."""

    TEMPLATE = "Question: {} Answer yes or no. Answer:"

    def __init__(self, max_input_tokens: int) -> None:
        super().__init__()
        self.max_input_tokens = max_input_tokens

    def count_tokens(self, request: Mapping[str, object]) -> int:
        state, questions = request["state"], request["questions"]
        assert isinstance(state, str) and isinstance(questions, list)
        blocks = [self.TEMPLATE.format(q["text"]) for q in questions]
        return len(state.split()) + sum(len(block.split()) for block in blocks)


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class FakeQSIVService:
    """The loaded `QSIVService` as far as the engine sees it: a tokenizer and decide()."""

    def __init__(self) -> None:
        self.tok = FakeTokenizer()
        self.requests: list[dict[str, object]] = []

    def decide(self, request: dict[str, object]) -> list[dict[str, object]]:
        self.requests.append(request)
        return [{"type": "noul", "p_true": 0.25}]


def fake_encode_request(
    tok: FakeTokenizer, state: str, questions: list[dict[str, str]]
) -> dict[str, object]:
    """`qsiv.render.encode_request`: the state, then one templated block per question."""
    template = WordCountingEngine.TEMPLATE
    return {
        "state_ids": tok.encode(state),
        "questions": [{"ids": tok.encode(template.format(q["text"]))} for q in questions],
    }


QSIV_RENDER = SimpleNamespace(encode_request=fake_encode_request)


def test_given_raw_texts_within_the_limit_but_the_rendered_request_over_it_when_checking_then_too_large() -> (  # noqa: E501
    None
):
    """F9: the limit applies to the request as the model renders it (question template included)."""
    # given: 4 state words + 1 question word fit in 10; with the 6-word template they do not
    engine = WordCountingEngine(max_input_tokens=10)
    request = {"state": "a b c d", "questions": [{"type": "noul", "text": "why"}]}
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        check_input_size(engine, request)
    # then
    assert caught.value.diagnostic.cause is Cause.TOO_LARGE


def test_given_the_loaded_qsiv_package_when_building_the_engine_then_its_limit_rejects_before_inference() -> (  # noqa: E501
    None
):
    """F9 on the real engine: the limit is the package's MAX_CONTEXT, counted with its own
    renderer and the service's tokenizer, and checked before decide() runs."""
    # given
    service = FakeQSIVService()
    engine = native_engine(service, SimpleNamespace(MAX_CONTEXT=12), QSIV_RENDER)
    small = {"state": "a b c", "questions": [{"type": "noul", "text": "why"}]}
    at_limit = {"state": "a b c d e", "questions": [{"type": "noul", "text": "why"}]}
    over = {"state": "a b c d e f", "questions": [{"type": "noul", "text": "why"}]}
    # when
    check_input_size(engine, small)  # 3 + 7 = 10 tokens
    check_input_size(engine, at_limit)  # 5 + 7 = 12 tokens: exactly the limit is accepted
    answers = engine.decide(small)
    with pytest.raises(QSIVUnavailable) as caught:
        check_input_size(engine, over)  # 13 tokens
    # then
    assert engine.max_input_tokens == 12
    assert engine.count_tokens is not None and engine.count_tokens(at_limit) == 12
    assert caught.value.diagnostic.cause is Cause.TOO_LARGE
    assert answers == [{"type": "noul", "p_true": 0.25}]
    assert service.requests == [small]  # only decide() reaches the model


def test_given_a_qsiv_package_without_a_documented_limit_when_building_the_engine_then_nothing_is_rejected() -> (  # noqa: E501
    None
):
    # given
    engine = native_engine(FakeQSIVService(), SimpleNamespace(), QSIV_RENDER)
    # when / then
    assert engine.max_input_tokens is None
    check_input_size(engine, {"state": "word " * 100000, "questions": []})


def test_given_input_over_the_known_limit_when_deciding_then_too_large_before_inference(
    install: Install, sock_dir: Path
) -> None:
    # given
    engine = WordCountingEngine(max_input_tokens=10)
    launcher = ThreadServiceLauncher(engine)
    client = make_client(install, sock_dir, launcher)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("word " * 50, QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.TOO_LARGE
    assert engine.requests == []  # rejected before the model ran
    client.close()
    launcher.stop()


def test_given_input_within_the_known_limit_when_deciding_then_it_is_answered(
    install: Install, sock_dir: Path
) -> None:
    # given
    engine = WordCountingEngine(max_input_tokens=1000)
    launcher = ThreadServiceLauncher(engine)
    client = make_client(install, sock_dir, launcher)
    # when
    answers = client.decide("a small script", QUESTIONS)
    # then
    assert len(answers) == 4 and len(engine.requests) == 1
    client.close()
    launcher.stop()


class SharedTokenizerEngine(WordCountingEngine):
    """Counting and inference share one tokenizer, as in qsiv (its caches and pending slot are
    not thread-safe); records how many calls used it at the same time."""

    def __init__(self) -> None:
        super().__init__(max_input_tokens=1000)
        self._mutex = threading.Lock()
        self.active = 0
        self.most_active = 0

    def _use_tokenizer(self) -> None:
        with self._mutex:
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        time.sleep(0.02)
        with self._mutex:
            self.active -= 1

    def count_tokens(self, request: Mapping[str, object]) -> int:
        self._use_tokenizer()
        return super().count_tokens(request)

    def decide(self, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
        self._use_tokenizer()
        return super().decide(request)


def test_given_two_clients_when_their_input_is_size_checked_then_never_alongside_inference(
    install: Install, sock_dir: Path
) -> None:
    """F9 check on the real engine: counting tokenizes with the model's tokenizer, so it is
    serialized with inference like decide() itself."""
    # given
    engine = SharedTokenizerEngine()
    launcher = ThreadServiceLauncher(engine)
    clients = [make_client(install, sock_dir, launcher) for _ in range(2)]
    barrier = threading.Barrier(2)

    def run(index: int) -> None:
        barrier.wait()
        for _ in range(5):
            clients[index].decide("a small script", QUESTIONS)

    # when
    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    # then
    assert len(engine.requests) == 10
    assert engine.most_active == 1
    for client in clients:
        client.close()
    launcher.stop()


def test_given_an_engine_without_a_limit_when_checking_then_nothing_is_rejected() -> None:
    # given
    engine = FakeEngine()
    # when / then
    check_input_size(engine, {"state": "word " * 100000, "questions": []})


def test_given_a_launch_from_a_directory_with_a_planted_json_when_starting_then_it_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    planted = tmp_path / "caller"
    planted.mkdir()
    (planted / "json.py").write_text(PLANT)
    private = tmp_path / "private"
    private.mkdir()
    out = tmp_path / "out.txt"
    monkeypatch.chdir(planted)
    env = {"PATH": "/usr/bin", "PYTHONSAFEPATH": "1"}
    # when
    process = SubprocessLauncher().start(
        [sys.executable, "-c", PROBE, str(out)], env, private / "service.log"
    )
    process.wait()
    # then
    assert "caller" not in out.read_text()
    assert not (tmp_path / "out.txt.planted").exists()


def test_given_a_planted_json_in_the_caller_cwd_when_running_a_command_then_it_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    planted = tmp_path / "caller"
    planted.mkdir()
    (planted / "json.py").write_text(PLANT)
    out = tmp_path / "out.txt"
    monkeypatch.chdir(planted)
    # when
    SubprocessRunner().run([sys.executable, "-c", PROBE, str(out)])
    # then
    assert "caller" not in out.read_text()


def test_given_a_start_then_the_service_env_sets_pythonsafepath_and_the_cwd_is_private(
    install: Install, sock_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    seen: dict[str, object] = {}
    real_popen = subprocess.Popen

    def fake_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        seen.update(kwargs)
        return real_popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    client = make_client(install, sock_dir, SubprocessLauncher())
    # when
    with pytest.raises(QSIVUnavailable):  # the fake process never listens
        client.decide("x", QUESTIONS)
    # then
    assert seen["cwd"] == sock_dir
    env = seen["env"]
    assert isinstance(env, dict) and env["PYTHONSAFEPATH"] == "1"


def test_given_a_non_linux_machine_when_deciding_then_f1_is_raised_before_importing_fcntl(
    install: Install, sock_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    monkeypatch.setitem(sys.modules, "fcntl", None)  # importing it now raises ImportError
    client = make_client(
        install, sock_dir, ThreadServiceLauncher(), machine=FakeMachine(system="Windows")
    )
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.NO_GPU


def test_given_an_invalid_runtime_when_running_doctor_then_a_failed_check_not_an_exception(
    install: Install, sock_dir: Path
) -> None:
    # given / when
    report = doctor(
        install.model,
        install.python,
        machine=FakeMachine(),
        environ={ENV_RUNTIME: "bogus", ENV_SOCKET_DIRECTORY: str(sock_dir)},
    )
    # then
    assert not report.ok
    assert report.checks[0].name == "runtime" and "fp8 or reference" in report.checks[0].detail


@pytest.mark.parametrize(
    ("reply", "cause", "text"),
    [
        ({"error": "x", "diagnostic": {"cause": "no_gpu"}}, Cause.NO_GPU, "(no_gpu; see "),
        (
            {"error": "Local QSIV inference failed"},
            Cause.STARTUP_FAILED,
            "QSIV service failed (see ",
        ),
        ({"error": "x", "diagnostic": {"cause": "weird"}}, Cause.STARTUP_FAILED, "failed (see "),
    ],
)
def test_given_a_service_reported_failure_when_deciding_then_a_message_that_is_not_invalid_answer(
    install: Install, sock_dir: Path, reply: dict[str, object], cause: Cause, text: str
) -> None:
    # given
    launcher = ScriptedLauncher([Generation(replies=[reply])])
    client = make_client(install, sock_dir, launcher)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is cause
    assert text in caught.value.message and "invalid answer" not in caught.value.message
    client.close()
    launcher.stop()


def test_given_a_failed_rerun_when_running_setup_then_the_old_install_marker_is_gone(
    tmp_path: Path,
) -> None:
    # given
    model = tmp_path / "model"
    model.mkdir()
    (model / INSTALL_MARKER).write_text("{}")

    class Failing(FakeRunner):
        def run(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> None:
            raise subprocess.CalledProcessError(1, list(command))

    # when
    with pytest.raises(subprocess.CalledProcessError):
        setup(model, tmp_path / "venv", machine=FakeMachine(), runner=Failing())
    # then
    assert not (model / INSTALL_MARKER).exists()
