"""Launch a local cluster (N workers + router) as subprocesses.

Used both by the CLI (``python -m llmserve.cluster configs/cluster_3x.yaml``)
and by the experiment scripts as a context manager::

    with Cluster(workers=[WorkerSpec(port=8001), WorkerSpec(port=8002)],
                 router_policy="gpu_aware") as c:
        ... c.router_url ...
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkerSpec:
    port: int = 8001
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    precision: str = "fp16"
    policy: str = "continuous"
    max_batch: int = 32
    window_ms: float = 20.0
    kv_budget_tokens: int = 65536
    max_prefill_tokens: int = 4096
    name: str | None = None
    device: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        a = [sys.executable, "-m", "llmserve.worker", "--model", self.model, "--precision", self.precision,
             "--policy", self.policy, "--max-batch", str(self.max_batch), "--window-ms", str(self.window_ms),
             "--kv-budget-tokens", str(self.kv_budget_tokens), "--max-prefill-tokens", str(self.max_prefill_tokens),
             "--port", str(self.port)]
        if self.name:
            a += ["--name", self.name]
        if self.device:
            a += ["--device", self.device]
        return a + list(self.extra_args)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class Cluster:
    def __init__(self, workers: list[WorkerSpec], router_policy: str | None = "gpu_aware", router_port: int = 8000,
                 log_dir: str | Path = ROOT / "results" / "logs", poll_interval: float = 0.1,
                 tokenizer: str = "Qwen/Qwen2.5-0.5B-Instruct", reuse_workers: dict | None = None):
        self.specs = workers
        self.router_policy = router_policy
        self.router_port = router_port
        self.log_dir = Path(log_dir)
        self.poll_interval = poll_interval
        self.tokenizer = tokenizer
        self.procs: dict[str, subprocess.Popen] = {}
        self.router_proc: subprocess.Popen | None = None

    @property
    def router_url(self) -> str:
        return f"http://127.0.0.1:{self.router_port}"

    @property
    def worker_urls(self) -> list[str]:
        return [w.url for w in self.specs]

    def _spawn(self, key: str, argv: list[str]) -> subprocess.Popen:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"{key}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)

    def start_workers(self, timeout: float = 300) -> None:
        for w in self.specs:
            self.procs[w.url] = self._spawn(f"worker_{w.port}", w.argv())
        for w in self.specs:
            _wait_healthy(w.url, timeout, self.procs[w.url])

    def start_router(self, policy: str | None = None, timeout: float = 120) -> None:
        self.stop_router()
        policy = policy or self.router_policy
        argv = [sys.executable, "-m", "llmserve.router", "--workers", ",".join(self.worker_urls), "--policy",
                policy, "--port", str(self.router_port), "--poll-interval", str(self.poll_interval),
                "--tokenizer", self.tokenizer]
        self.router_proc = self._spawn(f"router_{policy}", argv)
        _wait_healthy(self.router_url, timeout, self.router_proc)

    def stop_router(self) -> None:
        if self.router_proc is not None:
            _kill(self.router_proc)
            self.router_proc = None

    def kill_worker(self, url: str) -> None:
        _kill(self.procs[url])

    def stop(self) -> None:
        self.stop_router()
        for p in self.procs.values():
            _kill(p)
        self.procs.clear()

    def __enter__(self) -> "Cluster":
        try:
            self.start_workers()
            if self.router_policy:
                self.start_router()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _wait_healthy(url: str, timeout: float, proc: subprocess.Popen | None = None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process for {url} exited with code {proc.returncode}; see results/logs")
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} not healthy after {timeout}s")


def _kill(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=15)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def from_yaml(path: str | Path) -> Cluster:
    cfg = yaml.safe_load(Path(path).read_text())
    defaults = cfg.get("defaults", {})
    workers = [WorkerSpec(**{**defaults, **w}) for w in cfg["workers"]]
    r = cfg.get("router", {})
    return Cluster(workers, router_policy=r.get("policy", "gpu_aware"), router_port=r.get("port", 8000),
                   poll_interval=r.get("poll_interval", 0.1), tokenizer=r.get("tokenizer", workers[0].model))


def main() -> None:
    ap = argparse.ArgumentParser(description="Start a local llmserve cluster from a YAML file")
    ap.add_argument("config")
    a = ap.parse_args()
    c = from_yaml(a.config)
    with c:
        print(f"router: {c.router_url}  workers: {', '.join(c.worker_urls)}  (Ctrl-C to stop)", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
