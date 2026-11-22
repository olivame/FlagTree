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

from contextlib import contextmanager
import json
import os
import shutil
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional
try:
    import fcntl
except ImportError:
    fcntl = None
import base64
import hashlib
import functools
import sysconfig

from triton import __version__, knobs

_CACHE_LOCK_FILENAME = ".flagtree_cache.lock"
_CACHE_THREAD_LOCKS_GUARD = threading.Lock()
_CACHE_THREAD_LOCKS: Dict[str, threading.RLock] = {}


def _get_cache_thread_lock(cache_root: str) -> threading.RLock:
    key = os.path.realpath(cache_root)
    with _CACHE_THREAD_LOCKS_GUARD:
        lock = _CACHE_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _CACHE_THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _cache_root_lock(cache_root: Optional[str], *, exclusive: bool = True) -> Iterator[None]:
    """Inter-process POSIX ``fcntl.flock`` lock on ``cache_root`` with thread-lock fallback.

    Warm cache hit reads (``has_file`` / ``get_file`` / ``get_group``) never call this
    context manager and remain 100% lock-free.
    """
    if not cache_root:
        yield
        return
    os.makedirs(cache_root, exist_ok=True)
    tlock = _get_cache_thread_lock(cache_root)
    with tlock:
        if fcntl is None:
            yield
            return
        lock_path = os.path.join(cache_root, _CACHE_LOCK_FILENAME)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fd, op)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def enforce_binary_cache_budget(
    cache_root: Optional[str],
    *,
    max_entries: Optional[int] = None,
    max_bytes: Optional[int] = None,
    protect_key: Optional[str] = None,
) -> int:
    """Evict oldest cache-key directories (including .hgbin/.cubin/.xpubin/.ttir/.json)
    when directory count or total size exceeds configured budget.

    Serialized across processes via ``_cache_root_lock(exclusive=True)`` so eviction
    never races with concurrent ``FileCacheManager.put()`` writers. Never removes
    ``protect_key`` or active ``tmp.*`` staging directories.
    """
    if not cache_root or not os.path.isdir(cache_root):
        return 0
    lim_entries = int(max_entries) if (max_entries is not None and int(max_entries) > 0) else None
    lim_bytes = int(max_bytes) if (max_bytes is not None and int(max_bytes) > 0) else None
    if lim_entries is None and lim_bytes is None:
        return 0

    evicted = 0
    with _cache_root_lock(cache_root, exclusive=True):
        candidates = []
        total_size = 0
        try:
            with os.scandir(cache_root) as it:
                for entry in it:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    name = entry.name
                    if name.startswith("tmp.") or name.startswith("."):
                        continue
                    dir_size = 0
                    latest_mtime = 0.0
                    try:
                        st_dir = entry.stat(follow_symlinks=False)
                        latest_mtime = st_dir.st_mtime
                        for root_d, _dirs, files in os.walk(entry.path):
                            for fn in files:
                                fp = os.path.join(root_d, fn)
                                try:
                                    st = os.stat(fp)
                                    dir_size += st.st_size
                                    if st.st_mtime > latest_mtime:
                                        latest_mtime = st.st_mtime
                                except OSError:
                                    continue
                    except OSError:
                        continue
                    total_size += dir_size
                    if name != protect_key:
                        candidates.append((latest_mtime, dir_size, entry.path))
        except OSError:
            return 0

        protect_exists = bool(protect_key and os.path.isdir(os.path.join(cache_root, protect_key)))
        total_dirs = len(candidates) + (1 if protect_exists else 0)
        if (lim_entries is None or total_dirs <= lim_entries) and (lim_bytes is None or total_size <= lim_bytes):
            return 0

        candidates.sort(key=lambda item: item[0])
        for _mtime, dsize, dpath in candidates:
            need_entries = lim_entries is not None and total_dirs > lim_entries
            need_bytes = lim_bytes is not None and total_size > lim_bytes
            if not (need_entries or need_bytes):
                break
            try:
                shutil.rmtree(dpath, ignore_errors=False)
                evicted += 1
                total_dirs -= 1
                total_size = max(0, total_size - dsize)
            except OSError:
                continue

    if evicted:
        try:
            from .shape_bucketing import GLOBAL_BUCKET_TELEMETRY
            GLOBAL_BUCKET_TELEMETRY.record_eviction(evicted)
        except Exception:
            pass
    return evicted


class CacheManager(ABC):

    def __init__(self, key, override=False, dump=False):
        pass

    @abstractmethod
    def get_file(self, filename) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, data, filename, binary=True) -> str:
        pass

    @abstractmethod
    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def put_group(self, filename: str, group: Dict[str, str]):
        pass


class FileCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        self._override = bool(override)
        self._dump = bool(dump)
        if dump:
            self.cache_dir = knobs.cache.dump_dir
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = knobs.cache.override_dir
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = knobs.cache.dir
            if self.cache_dir:
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")

    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename) -> bool:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        grp_filename = f"__grp__{filename}"
        if not self.has_file(grp_filename):
            return None
        grp_filepath = self._make_path(grp_filename)
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)
        # Invalid group data.
        if child_paths is None:
            return None
        result = {}
        for c, p in child_paths.items():
            if os.path.exists(p):
                result[c] = p
        return result

    # Note a group of pushed files as being part of a group
    def _maybe_enforce_disk_budget(self) -> None:
        if self._override or self._dump:
            return
        max_entries = knobs.cache.max_cache_entries
        if max_entries is None:
            max_entries = getattr(knobs.autotuning, "max_bucket_artifacts", None)
        max_bytes = knobs.cache.max_cache_bytes
        if max_entries is not None or max_bytes is not None:
            cache_root = os.path.dirname(self.cache_dir) if self.cache_dir else knobs.cache.dir
            enforce_binary_cache_budget(
                cache_root,
                max_entries=max_entries,
                max_bytes=max_bytes,
                protect_key=self.key,
            )

    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        grp_contents = json.dumps({"child_paths": group})
        grp_filename = f"__grp__{filename}"
        out = self.put(grp_contents, grp_filename, binary=False)
        self._maybe_enforce_disk_budget()
        return out

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        mode = "wb" if binary else "w"
        lock_root = None if (self._override or self._dump) else os.path.dirname(self.cache_dir)

        with _cache_root_lock(lock_root, exclusive=False):
            # Re-ensure cache_dir exists in case a prior eviction removed it before this write
            os.makedirs(self.cache_dir, exist_ok=True)
            # use temp dir to be robust against program interruptions
            temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            try:
                with open(temp_path, mode) as f:
                    f.write(data)
                # Replace is guaranteed to be atomic on POSIX systems if it succeeds
                # so filepath cannot see a partial write
                os.replace(temp_path, filepath)
                try:
                    os.utime(self.cache_dir, None)
                except OSError:
                    pass
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return filepath


class RemoteCacheBackend:
    """
    A backend implementation for accessing a remote/distributed cache.
    """

    def __init__(self, key: str):
        pass

    @abstractmethod
    def get(self, filenames: List[str]) -> Dict[str, bytes]:
        pass

    @abstractmethod
    def put(self, filename: str, data: bytes):
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend):

    def __init__(self, key):
        import redis
        self._key = key
        self._key_fmt = knobs.cache.redis.key_format
        self._redis = redis.Redis(
            host=knobs.cache.redis.host,
            port=knobs.cache.redis.port,
        )

    def _get_key(self, filename: str) -> str:
        return self._key_fmt.format(key=self._key, filename=filename)

    def get(self, filenames: List[str]) -> Dict[str, str]:
        results = self._redis.mget([self._get_key(f) for f in filenames])
        return {filename: result for filename, result in zip(filenames, results) if result is not None}

    def put(self, filename: str, data: bytes) -> Dict[str, bytes]:
        self._redis.set(self._get_key(filename), data)


class RemoteCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        # Setup backend pointed too by `TRITON_REMOTE_CACHE_BACKEND`.
        remote_cache_cls = knobs.cache.remote_manager_class
        if not remote_cache_cls:
            raise RuntimeError(
                "Unable to instantiate RemoteCacheManager, TRITON_REMOTE_CACHE_BACKEND doesn't point to a valid class")
        self._backend = remote_cache_cls(key)

        self._override = override
        self._dump = dump

        # Use a `FileCacheManager` to materialize remote cache paths locally.
        self._file_cache_manager = FileCacheManager(key, override=override, dump=dump)

    def _materialize(self, filename: str, data: bytes):
        # We use a backing `FileCacheManager` to provide the materialized data.
        return self._file_cache_manager.put(data, filename, binary=True)

    def get_file(self, filename: str) -> Optional[str]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_file(filename)

        # We always check the remote cache backend -- even if our internal file-
        # based cache has the item -- to make sure LRU accounting works as
        # expected.
        results = self._backend.get([filename])
        if len(results) == 0:
            return None
        (_, data), = results.items()
        return self._materialize(filename, data)

    def put(self, data, filename: str, binary=True) -> str:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put(data, filename, binary=binary)

        if not isinstance(data, bytes):
            data = str(data).encode("utf-8")
        self._backend.put(filename, data)
        return self._materialize(filename, data)

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_group(filename)

        grp_filename = f"__grp__{filename}"
        grp_filepath = self.get_file(grp_filename)
        if grp_filepath is None:
            return None
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)

        result = None

        # Found group data.
        if child_paths is not None:
            result = {}
            for child_path, data in self._backend.get(child_paths).items():
                result[child_path] = self._materialize(child_path, data)

        return result

    def put_group(self, filename: str, group: Dict[str, str]):
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put_group(filename, group)

        grp_contents = json.dumps({"child_paths": sorted(list(group.keys()))})
        grp_filename = f"__grp__{filename}"
        return self.put(grp_contents, grp_filename)


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


def get_cache_manager(key) -> CacheManager:
    cls = knobs.cache.manager_class or FileCacheManager
    return cls(_base32(key))


def get_override_manager(key) -> CacheManager:
    cls = knobs.cache.manager_class or FileCacheManager
    return cls(_base32(key), override=True)


def get_dump_manager(key) -> CacheManager:
    cls = knobs.cache.manager_class or FileCacheManager
    return cls(_base32(key), dump=True)


def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}-{ids}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _base32(key)


@functools.lru_cache()
def triton_key():
    import pkgutil
    TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]
    # compiler
    path_prefixes = [
        (os.path.join(TRITON_PATH, "compiler"), "triton.compiler."),
        (os.path.join(TRITON_PATH, "backends"), "triton.backends."),
    ]
    for path, prefix in path_prefixes:
        for lib in pkgutil.walk_packages([path], prefix=prefix):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hashlib.sha256(f.read()).hexdigest()]

    # backend
    libtriton_hash = hashlib.sha256()
    ext = sysconfig.get_config_var("EXT_SUFFIX").split(".")[-1]
    with open(os.path.join(TRITON_PATH, "_C", f"libtriton.{ext}"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.walk_packages([language_path], prefix="triton.language."):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha256(f.read()).hexdigest()]
    return f'{__version__}' + '-'.join(contents)


def get_cache_key(src, backend, backend_options, env_vars):
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
    return key
