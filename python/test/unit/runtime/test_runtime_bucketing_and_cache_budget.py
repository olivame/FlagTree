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

import os
import tempfile
import time
import unittest

import triton
from triton.runtime.adjust_kernel_param import bucket_axes
from triton.runtime.backend_caps import BackendBucketingCapabilities
from triton.runtime.cache import FileCacheManager, enforce_binary_cache_budget
from triton.runtime.jit import JITFunction


class _FakeTensor:
    def __init__(self, shape=(4, 64)):
        self.shape = shape

    def data_ptr(self):
        return 1024


class _DummyJITRunner:
    def __init__(self):
        self._bkt_unaligned_int_buckets = set()

    _normalize_specialization_for_bucketing = JITFunction._normalize_specialization_for_bucketing


class TestRuntimeBucketingAndCacheBudget(unittest.TestCase):
    def test_jit_bucket_scoped_alignment_preserves_static_head_dim(self):
        runner = _DummyJITRunner()
        a_t = _FakeTensor()

        # S=264 (unaligned, bucket 384) marks idx=1,2 at bucket 384 while static K=64 stays "D"
        spec_264 = [("*bf16", 16), ("i32", ""), ("i32", ""), ("i32", "D")]
        r1 = runner._normalize_specialization_for_bucketing(
            {"A": a_t, "M": 264, "N": 264, "K": 64}, list(spec_264)
        )
        self.assertEqual(r1[3], ("i32", "D"))

        # Attn@V swaps N=64 (bucket 64) and K=264 (bucket 384); N=64 must NOT be polluted
        spec_av = [("*bf16", 16), ("i32", ""), ("i32", "D"), ("i32", "")]
        r2 = runner._normalize_specialization_for_bucketing(
            {"A": a_t, "M": 264, "N": 64, "K": 264}, list(spec_av)
        )
        self.assertEqual(r2[2], ("i32", "D"))

        # S=288 (%16==0) in same bucket 384 strips M & N to "", while K=64 stays "D"
        spec_288 = [("*bf16", 16), ("i32", "D"), ("i32", "D"), ("i32", "D")]
        r3 = runner._normalize_specialization_for_bucketing(
            {"A": a_t, "M": 288, "N": 288, "K": 64}, list(spec_288)
        )
        self.assertEqual(r3[1], ("i32", ""))
        self.assertEqual(r3[2], ("i32", ""))
        self.assertEqual(r3[3], ("i32", "D"))

    def test_jit_multi_tag_alignment_normalization(self):
        runner = _DummyJITRunner()
        runner._bkt_caps = BackendBucketingCapabilities(
            backend="custom", alignment_spec_tags=("D", "D32")
        )
        runner._bkt_align_tags = ("D", "D32")
        a_t = _FakeTensor()

        # Observe unaligned in bucket 384
        runner._normalize_specialization_for_bucketing(
            {"A": a_t, "M": 264}, [("*bf16", 16), ("i32", "")]
        )
        # Aligned D32 in same bucket must strip completely to ""
        res = runner._normalize_specialization_for_bucketing(
            {"A": a_t, "M": 288}, [("*bf16", 16), ("i32", "D32")]
        )
        self.assertEqual(res[1], ("i32", ""))

    def test_bucket_axes_decorator_overrides_ast_inference(self):
        @bucket_axes(custom_rows="dynamic_tile_dim", magic_mode="invariant_const", custom_ld="dynamic_stride")
        def custom_kernel(ptr, custom_rows, magic_mode, custom_ld):
            return ptr

        meta = getattr(custom_kernel, "_ft_bucket_axis_overrides", None)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["custom_rows"].value, "dynamic_tile_dim")
        self.assertEqual(meta["magic_mode"].value, "invariant_const")
        self.assertEqual(meta["custom_ld"].value, "dynamic_stride")

    def test_file_cache_manager_binary_budget_eviction(self):
        prev_dir = triton.knobs.cache.dir
        with tempfile.TemporaryDirectory() as td:
            try:
                triton.knobs.cache.dir = td
                for i in range(6):
                    mgr = FileCacheManager(f"k_hash_{i}")
                    p_bin = mgr.put(b"X" * 1024, f"kernel_{i}.hgbin", binary=True)
                    p_ir = mgr.put(b"Y" * 512, f"kernel_{i}.ttir", binary=True)
                    mgr.put_group(
                        f"group_{i}.json",
                        {f"kernel_{i}.hgbin": p_bin, f"kernel_{i}.ttir": p_ir},
                    )
                    dpath = os.path.join(td, f"k_hash_{i}")
                    ts = time.time() - (6 - i) * 10
                    for fname in os.listdir(dpath):
                        os.utime(os.path.join(dpath, fname), (ts, ts))
                    os.utime(dpath, (ts, ts))

                stats = enforce_binary_cache_budget(td, max_entries=3, max_bytes=None)
                remaining = [
                    d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))
                ]
                self.assertEqual(len(remaining), 3)
                self.assertEqual(stats, 3)
                self.assertEqual(sorted(remaining), ["k_hash_3", "k_hash_4", "k_hash_5"])

                stats2 = enforce_binary_cache_budget(td, max_entries=None, max_bytes=2000)
                remaining2 = [
                    d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))
                ]
                self.assertLessEqual(len(remaining2), 1)
                self.assertGreaterEqual(stats2, 2)
            finally:
                triton.knobs.cache.dir = prev_dir


    def test_file_cache_manager_concurrent_eviction_flock_and_lockfree_reads(self):
        import subprocess
        import sys
        import textwrap
        from triton import knobs

        worker_code = textwrap.dedent('''
            import sys, time
            from triton import knobs
            from triton.runtime.cache import FileCacheManager

            cache_dir, wid, start_at = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
            knobs.cache.dir = cache_dir
            knobs.cache.max_cache_entries = 3
            knobs.cache.max_cache_bytes = None
            wait_s = start_at - time.time()
            if wait_s > 0:
                time.sleep(wait_s)
            for i in range(15):
                key = f"W{wid}K{i % 5:02d}" if (i % 2 == 0) else f"SHR{i % 4:02d}"
                mgr = FileCacheManager(key)
                p1 = mgr.put(b"bin" * 256, "k.bin", binary=True)
                mgr.put_group("k.json", {"k.bin": p1})
                _ = mgr.get_file("k.json")
        ''')
        with tempfile.TemporaryDirectory() as td:
            start_at = time.time() + 0.3
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", worker_code, td, str(w), f"{start_at:.6f}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for w in range(4)
            ]
            for p in procs:
                _out, err = p.communicate(timeout=30)
                self.assertEqual(p.returncode, 0, msg=f"worker failed: {err}")
            remaining = [
                e.name
                for e in os.scandir(td)
                if e.is_dir(follow_symlinks=False) and not e.name.startswith("tmp.")
            ]
            self.assertLessEqual(len(remaining), 3)

        # Warm read path must stay completely lock-free (never touches .flagtree_cache.lock)
        with tempfile.TemporaryDirectory() as td_warm:
            prev_dir = knobs.cache.dir
            prev_max = knobs.cache.max_cache_entries
            try:
                knobs.cache.dir = td_warm
                knobs.cache.max_cache_entries = None
                mgr = FileCacheManager("WARMREAD01")
                fp = mgr.put(b"abc", "k.bin", binary=True)
                lock_file = os.path.join(td_warm, ".flagtree_cache.lock")
                if os.path.exists(lock_file):
                    os.unlink(lock_file)
                self.assertEqual(mgr.get_file("k.bin"), fp)
                self.assertFalse(os.path.exists(lock_file))
            finally:
                knobs.cache.dir = prev_dir
                knobs.cache.max_cache_entries = prev_max

    def test_core_runtime_vendor_dehardcoding_contracts(self):
        import inspect
        from pathlib import Path
        from triton.runtime import autotuner as at_mod
        from triton.runtime import _distributed as dist_mod
        from triton.runtime import shape_bucketing as sb_mod

        sb_src = Path(inspect.getfile(sb_mod)).read_text(encoding="utf-8")
        dist_src = Path(inspect.getfile(dist_mod)).read_text(encoding="utf-8")
        self.assertNotIn("aiu_load_mask", sb_src)
        self.assertNotIn("torch_npu", dist_src)

        sig_params = set(inspect.signature(at_mod.autotune).parameters) | set(
            inspect.signature(at_mod.Autotuner.__init__).parameters
        )
        for forbidden in ("op_affiliation", "row_sign", "col_sign", "n_elem_sign", "generate_configs"):
            self.assertNotIn(forbidden, sig_params)

if __name__ == "__main__":
    unittest.main()
