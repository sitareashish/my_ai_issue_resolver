"""Docker sandbox manager — isolated patch application and test execution."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import docker
from docker.errors import DockerException
from docker.models.containers import Container

from src.config import settings
from src.observability.tracer import sandbox_active
from src.schema import TestResult, TestVerdict
from src.utils.logger import get_logger

log = get_logger(__name__)

RUNNER_DETECTION: list[tuple[str, str]] = [
    ("pytest.ini",     "python -m pytest -v --tb=short 2>&1"),
    ("pyproject.toml", "python -m pytest -v --tb=short 2>&1"),
    ("setup.py",       "python -m pytest -v --tb=short 2>&1"),
    ("package.json",   "npm test -- --no-coverage 2>&1"),
    ("Cargo.toml",     "cargo test 2>&1"),
    ("go.mod",         "go test ./... 2>&1"),
    ("Makefile",       "make test 2>&1"),
]


@dataclass
class CmdResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


class SandboxManager:
    def __init__(self) -> None:
        try:
            self._client = docker.from_env()
            self._client.ping()
        except DockerException as exc:
            log.error("docker_unavailable", error=str(exc))
            raise
        self._pool: ContainerPool | None = None
        if settings.sandbox_pool_size > 0:
            self._pool = ContainerPool(self._client, settings.sandbox_pool_size)

    def apply_patch_and_test(
        self,
        repo_local_path: str,
        unified_diff: str,
        base_image: str = "python:3.11-slim",
    ) -> TestResult:
        container: Container | None = None
        with sandbox_active():
            try:
                container = self._get_container(base_image, repo_local_path)
                log.info("sandbox_started", container_id=container.short_id)

                # Install deps
                self._install_deps(container, repo_local_path)

                # Apply the patch
                escaped = unified_diff.replace("'", "'\\''")
                apply = self._exec(container, f"echo '{escaped}' | patch -p1", timeout=30)
                if apply.exit_code != 0:
                    log.warning("patch_apply_failed", exit_code=apply.exit_code)
                    return TestResult(
                        verdict=TestVerdict.ERROR, failed_tests=[],
                        stdout=apply.stdout,
                        stderr=f"patch -p1 failed:\n{apply.stderr}",
                        duration_ms=apply.duration_ms, exit_code=apply.exit_code,
                    )

                # Run tests
                cmd = self._detect_runner(repo_local_path)
                log.info("running_tests", cmd=cmd)
                test = self._exec(container, cmd, timeout=settings.sandbox_timeout_seconds)
                return self._parse(test, cmd)

            finally:
                if container is not None:
                    try:
                        container.stop(timeout=5)
                        container.remove(force=True)
                    except Exception as e:
                        log.warning("container_cleanup_failed", error=str(e))

    # ── Container management ───────────────────────────────────────────────────

    def _get_container(self, image: str, repo_path: str) -> Container:
        try:
            self._client.images.get(image)
        except docker.errors.ImageNotFound:
            log.info("pulling_image", image=image)
            self._client.images.pull(image)

        return self._client.containers.run(
            image=image,
            command="sleep infinity",
            detach=True,
            network_mode="none",                     # no outbound network
            mem_limit=settings.sandbox_memory_limit,
            nano_cpus=int(settings.sandbox_cpu_count * 1e9),
            working_dir="/workspace",
            volumes={os.path.abspath(repo_path): {"bind": "/workspace", "mode": "rw"}},
            security_opt=["no-new-privileges"],
            read_only=False,                          # workspace needs writes
        )

    def _exec(self, container: Container, cmd: str, timeout: int) -> CmdResult:
        t0 = time.monotonic()
        try:
            exit_code, output = container.exec_run(
                f"/bin/sh -c '{cmd}'", stream=False, demux=False, workdir="/workspace"
            )
            ms = int((time.monotonic() - t0) * 1000)
            raw = (output or b"").decode("utf-8", errors="replace")
            return CmdResult(stdout=raw, stderr="", exit_code=exit_code or 0, duration_ms=ms)
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            return CmdResult(stdout="", stderr=str(exc), exit_code=1, duration_ms=ms)

    def _install_deps(self, container: Container, repo_path: str) -> None:
        r = Path(repo_path)
        if (r / "requirements.txt").exists():
            self._exec(container, "pip install -r requirements.txt -q 2>&1 || true", 120)
        elif (r / "pyproject.toml").exists():
            self._exec(container, "pip install -e . -q 2>&1 || true", 120)
        elif (r / "package.json").exists():
            self._exec(container, "npm ci --silent 2>&1 || true", 120)

    @staticmethod
    def _detect_runner(repo_path: str) -> str:
        for fname, cmd in RUNNER_DETECTION:
            if (Path(repo_path) / fname).exists():
                return cmd
        return "python -m pytest -v --tb=short 2>&1 || true"

    @staticmethod
    def _parse(result: CmdResult, runner: str) -> TestResult:
        output = result.stdout + result.stderr
        failed: list[str] = []
        for line in output.splitlines():
            s = line.strip()
            if s.startswith("FAILED "):
                failed.append(s.removeprefix("FAILED ").split(" ")[0])

        verdict = (
            TestVerdict.PASS if result.exit_code == 0
            else TestVerdict.FAIL if failed
            else TestVerdict.ERROR
        )
        return TestResult(
            verdict=verdict, failed_tests=failed,
            stdout=result.stdout, stderr=result.stderr,
            duration_ms=result.duration_ms, exit_code=result.exit_code,
            test_runner=runner.split()[0] if runner else "unknown",
        )


class ContainerPool:
    """Pre-warms N containers to reduce cold-start latency."""
    def __init__(self, client: docker.DockerClient, size: int) -> None:
        self._client = client
        self._q: Queue[Container] = Queue()
        for _ in range(size):
            t = Thread(target=self._warm, daemon=True)
            t.start()

    def _warm(self) -> None:
        try:
            c = self._client.containers.run(
                "python:3.11-slim", "sleep infinity", detach=True,
                network_mode="none",
                mem_limit=settings.sandbox_memory_limit,
                nano_cpus=int(settings.sandbox_cpu_count * 1e9),
            )
            self._q.put(c)
            log.debug("container_pre_warmed", id=c.short_id)
        except Exception as e:
            log.warning("container_warm_failed", error=str(e))

    def get(self, timeout: float = 5.0) -> Container | None:
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None
