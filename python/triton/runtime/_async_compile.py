# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import multiprocessing
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

from .errors import CompileTimeoutError

active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)


class CompileConfigBlacklist:
    """Thread-safe and optionally JSON-persisted blacklist for timed-out or broken configs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[Tuple[str, str], str] = {}

    def add(self, kernel_name: str, config_sig: str, reason: str) -> None:
        with self._lock:
            self._entries[(str(kernel_name), str(config_sig))] = str(reason)

    def is_blacklisted(self, kernel_name: str, config_sig: str) -> bool:
        with self._lock:
            return (str(kernel_name), str(config_sig)) in self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return {f"{k}::{c}": r for (k, c), r in self._entries.items()}

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")

    def load_json(self, path: str | Path) -> int:
        target = Path(path)
        if not target.exists():
            return 0
        raw = json.loads(target.read_text(encoding="utf-8"))
        loaded = 0
        with self._lock:
            for composite, reason in raw.items():
                if "::" in composite:
                    k_name, c_sig = composite.split("::", 1)
                    self._entries[(k_name, c_sig)] = str(reason)
                    loaded += 1
        return loaded


GLOBAL_COMPILE_BLACKLIST = CompileConfigBlacklist()


def _subprocess_entry(fn: Callable[[], Any], conn) -> None:
    try:
        res = fn()
        conn.send(("ok", res))
    except BaseException as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def run_with_compile_timeout(
    compile_fn: Callable[[], Any],
    timeout_sec: Optional[float],
    *,
    kernel_name: str = "",
    config_desc: str = "",
    use_subprocess: bool = False,
) -> Any:
    """Execute ``compile_fn`` bounded by ``timeout_sec``, raising ``CompileTimeoutError`` on hang."""
    if timeout_sec is None or timeout_sec <= 0:
        return compile_fn()

    limit = float(timeout_sec)
    if use_subprocess:
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_subprocess_entry, args=(compile_fn, child_conn), daemon=True)
        proc.start()
        child_conn.close()
        proc.join(timeout=limit)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
            parent_conn.close()
            if kernel_name and config_desc:
                GLOBAL_COMPILE_BLACKLIST.add(kernel_name, config_desc, f"timeout>{limit}s")
            raise CompileTimeoutError(limit, kernel_name=kernel_name, config_desc=config_desc)
        if parent_conn.poll():
            status, payload = parent_conn.recv()
            parent_conn.close()
            if status == "ok":
                return payload
            raise RuntimeError(f"Subprocess compile failed: {payload}")
        parent_conn.close()
        raise RuntimeError("Subprocess compile exited without returning a result")

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(compile_fn)
    try:
        return fut.result(timeout=limit)
    except FuturesTimeoutError as exc:
        fut.cancel()
        if kernel_name and config_desc:
            GLOBAL_COMPILE_BLACKLIST.add(kernel_name, config_desc, f"timeout>{limit}s")
        raise CompileTimeoutError(limit, kernel_name=kernel_name, config_desc=config_desc) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future, key: str = ""):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future
        self.key = key

    def result(self, ignore_errors: bool = False, timeout_sec: Optional[float] = None):
        if self.kernel is not None:
            return self.kernel

        try:
            if timeout_sec is not None and timeout_sec > 0:
                kernel = self.future.result(timeout=float(timeout_sec))
            else:
                kernel = self.future.result()
        except FuturesTimeoutError as exc:
            self.future.cancel()
            if ignore_errors:
                return
            raise CompileTimeoutError(float(timeout_sec or 0.0), kernel_name=str(self.key)) from exc
        except Exception:
            if ignore_errors:
                return
            else:
                raise
        self.finalize_compile(kernel)
        self.kernel = kernel
        return kernel


class AsyncCompileMode:

    def __init__(self, executor: Executor, *, ignore_errors=False, timeout_sec: Optional[float] = None):
        self.executor = executor
        self.ignore_errors = ignore_errors
        self.timeout_sec = timeout_sec
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(finalize_fn, future, key=str(key))
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for future in as_completed(self.raw_futures, timeout=self.timeout_sec):
            self.future_kernels[future._key].result(self.ignore_errors, timeout_sec=self.timeout_sec)
        active_mode.set(None)
