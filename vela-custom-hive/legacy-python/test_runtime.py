#!/usr/bin/env python3
"""Hardware-independent safety tests for the custom V2 runtime."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sd_miner
import sdint
import torch


class RuntimeSafetyTests(unittest.TestCase):
    def test_benchmark_task_split_has_no_overrun_or_missing_inferences(self):
        tasks = sd_miner.make_bench_tasks(total=11, batch_size=4, lane_count=3)
        self.assertEqual(tasks, [(0, 4), (1, 4), (2, 3)])
        self.assertEqual(sum(count for _, count in tasks), 11)
        self.assertEqual(sd_miner.make_bench_tasks(2, 8, 4), [(0, 2)])
        self.assertEqual(sd_miner.make_bench_tasks(0, 8, 4), [])

    def test_v2_block_serialization_includes_hash_poi_in_112_byte_header(self):
        header80 = bytes(range(80))
        hash_poi = bytes(range(32))
        coinbase_witness = b"coinbase-witness-bytes"
        transactions = [{"data": "01020304"}]
        x0 = b"x" * 32768
        block = sd_miner.serialize_v2_block(header80, hash_poi, coinbase_witness, transactions, x0)
        self.assertEqual(block[:80], header80)
        self.assertEqual(block[80:112], hash_poi)
        self.assertEqual(sd_miner.sha256d(block[:112]), sd_miner.sha256d(header80 + hash_poi))
        self.assertEqual(block[112:113], b"\x02")  # compactSize(2 transactions)
        self.assertEqual(block[113:113 + len(coinbase_witness)], coinbase_witness)
        tx_start = 113 + len(coinbase_witness)
        self.assertEqual(block[tx_start:tx_start + 4], bytes.fromhex("01020304"))
        payload_start = tx_start + 4
        self.assertEqual(block[payload_start], sdint.VERSION)
        self.assertEqual(block[payload_start + 1:payload_start + 4], b"\xfd\x00\x80")
        self.assertEqual(block[payload_start + 4:], x0)

    def test_rotating_file_logger_writes_timestamps_and_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "miner.log"
            sd_miner.setup_logging(str(path), "INFO")
            sd_miner.LOG.info("unit_test_hash=abc target=def nonce=42")
            handlers = list(sd_miner.LOG.handlers)
            for handler in handlers:
                handler.flush()
            content = path.read_text(encoding="utf-8")
            self.assertIn("INFO unit_test_hash=abc target=def nonce=42", content)
            self.assertIn("rotation=10MiB backups=5", content)
            for handler in handlers:
                handler.close()
            sd_miner.LOG.handlers.clear()

    def test_v2_block_serialization_rejects_wrong_header_or_hash_poi_length(self):
        with self.assertRaises(ValueError):
            sd_miner.serialize_v2_block(b"h" * 79, b"p" * 32, b"cb", [], b"x")
        with self.assertRaises(ValueError):
            sd_miner.serialize_v2_block(b"h" * 80, b"p" * 31, b"cb", [], b"x")

    def test_nonce_ranges_are_disjoint_for_mixed_batches_and_gpus(self):
        miners = [
            SimpleNamespace(fast=object(), streams=[object(), object()], batch=8),
            SimpleNamespace(fast=object(), streams=[object()], batch=4),
            SimpleNamespace(fast=None, streams=[], batch=1),
        ]
        specs, stride = sd_miner.build_lane_specs(miners)
        sequences = [sd_miner.lane_nonce_sequence(spec, stride, rounds=25) for spec in specs]
        flattened = [nonce for sequence in sequences for nonce in sequence]
        self.assertEqual(len(flattened), len(set(flattened)))
        # The interleaving must cover one continuous nonce range with no gaps.
        self.assertEqual(sorted(flattened), list(range(len(flattened))))

    def test_nonce_ranges_remain_disjoint_when_batches_differ(self):
        miners = [
            SimpleNamespace(fast=object(), streams=[object(), object()], batch=3),
            SimpleNamespace(fast=object(), streams=[object(), object()], batch=5),
        ]
        specs, stride = sd_miner.build_lane_specs(miners)
        values = [sd_miner.lane_nonce_sequence(spec, stride, rounds=100) for spec in specs]
        flattened = [nonce for sequence in values for nonce in sequence]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(sorted(flattened), list(range(len(flattened))))

    def test_old_cached_tip_does_not_mark_new_template_stale(self):
        watcher = sd_miner.TipWatcher(lambda method: "unused", interval=0.5)
        watcher._tip = "old-template-tip"
        self.assertTrue(watcher.set_expected_tip("new-template-tip"))
        self.assertFalse(watcher.is_stale())

    def test_in_flight_old_tip_poll_cannot_mark_new_job_stale(self):
        import threading
        first_call_started = threading.Event()
        release_first_call = threading.Event()
        calls = {"n": 0}

        def fake_rpc(method):
            calls["n"] += 1
            if calls["n"] == 1:
                first_call_started.set()
                release_first_call.wait(timeout=2)
                return "old-tip"
            return "new-tip"

        watcher = sd_miner.TipWatcher(fake_rpc, interval=0.25)
        watcher.start()
        try:
            self.assertTrue(first_call_started.wait(timeout=1))
            watcher.set_expected_tip("new-tip")
            release_first_call.set()
            # The first response belongs to the generation before the new job;
            # the second poll confirms the active parent.
            deadline = __import__("time").monotonic() + 2
            while calls["n"] < 2 and __import__("time").monotonic() < deadline:
                __import__("time").sleep(0.02)
            while watcher._tip != "new-tip" and __import__("time").monotonic() < deadline:
                __import__("time").sleep(0.02)
            self.assertEqual(watcher._tip, "new-tip")
            self.assertFalse(watcher.is_stale())
        finally:
            release_first_call.set()
            watcher.stop()

    def test_tip_poll_marks_job_stale_and_can_clear_after_matching_tip(self):
        watcher = sd_miner.TipWatcher(lambda method: "unused", interval=0.5)
        watcher.set_expected_tip("expected")
        with watcher._lock:
            watcher._tip = "unexpected"
            watcher._stale = watcher._tip != watcher._expected
        self.assertTrue(watcher.is_stale())
        with watcher._lock:
            watcher._tip = "expected"
            watcher._stale = watcher._tip != watcher._expected
        self.assertFalse(watcher.is_stale())


    def test_noise_batch_is_bit_exact_with_legacy_single_noise(self):
        # Small synthetic LUT is enough to verify byte-for-byte ChaCha indexing
        # and tensor layout without downloading the 1.19 GB production model.
        net = SimpleNamespace(
            dev="cpu",
            t={"lut.gauss": torch.arange(65536, dtype=torch.int64)},
        )
        seeds = [hashlib.sha256(f"noise-test-{i}".encode()).digest() for i in range(3)]
        expected = torch.stack([sdint.noise(net, seed) for seed in seeds])
        actual = sdint.noise_batch(net, seeds)
        self.assertTrue(torch.equal(expected, actual))

    def test_explicit_cookie_overrides_placeholder_rpc_url_credentials(self):
        import base64
        from poi_miner import RPC
        with tempfile.TemporaryDirectory() as tmp:
            cookie = Path(tmp) / ".cookie"
            cookie.write_text("cookie-user:cookie-secret", encoding="utf-8")
            rpc = RPC("http://placeholder:placeholder@127.0.0.1:19732", str(cookie))
            decoded = base64.b64decode(rpc.auth).decode()
            self.assertEqual(decoded, "cookie-user:cookie-secret")

    def test_model_file_hash_verifier_accepts_exact_expected_digest(self):
        payload = b"small test model placeholder, not a Vela model"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-model.bin"
            path.write_bytes(payload)
            self.assertEqual(sd_miner.verify_v2_model_file(path, expected), expected)

    def test_model_file_hash_verifier_rejects_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-model.bin"
            path.write_bytes(b"wrong bytes")
            with self.assertRaises(SystemExit):
                sd_miner.verify_v2_model_file(path, "0" * 64)

    def test_auto_device_selection_falls_back_to_reference_on_cpu_only_torch(self):
        if sd_miner.torch.cuda.is_available():
            self.skipTest("CUDA is available; CPU-only fallback test does not apply")
        args = SimpleNamespace(device=None, devices="auto", engine="fast")
        self.assertEqual(sd_miner.resolve_devices(args), ["cpu"])

    def test_explicit_cuda_selection_fails_in_cpu_only_torch(self):
        if sd_miner.torch.cuda.is_available():
            self.skipTest("CUDA is available; CPU-only failure test does not apply")
        args = SimpleNamespace(device=None, devices="0", engine="fast")
        with self.assertRaises(SystemExit):
            sd_miner.resolve_devices(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
