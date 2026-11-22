#!/usr/bin/env python3
"""Offline protocol tests for ExerMiner's independent Vela Stratum client."""
import json
import logging
import socketserver
import threading
import time
import unittest

from stratum_client import (StratumClient, build_stratum_header76, parse_endpoint,
                             target_from_difficulty)

_PREVIOUS_LOG_DISABLE = logging.root.manager.disable

def setUpModule():
    # Suppress protocol-test logs without mutating logging state for other modules.
    logging.disable(logging.CRITICAL)

def tearDownModule():
    logging.disable(_PREVIOUS_LOG_DISABLE)


class _StratumHandler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        requests = []
        for _ in range(2):
            line = self.rfile.readline()
            if not line:
                return
            requests.append(json.loads(line.decode("utf-8")))
        if [r.get("method") for r in requests] != ["mining.subscribe", "mining.authorize"]:
            server.failure = "subscribe/authorize request order or method is wrong"
            return
        server.subscribe_params = requests[0].get("params")
        server.authorize_params = requests[1].get("params")

        messages = [
            {"id": 1, "result": {"extranonce1": "a1b2c3d4", "model": {"model_hash": "ab" * 32}}, "error": None},
            {"id": 2, "result": True, "error": None},
            {"id": None, "method": "mining.set_difficulty", "params": {"difficulty": 2048}},
            {"id": None, "method": "mining.notify", "params": {
                "job_id": "job-001",
                "prev_hash": "00" * 32,
                "coinbase1": "0102",
                "coinbase2": "0304",
                "version": "20000000",
                "bits": "1d00ffff",
                "ntime": "6553f100",
                "merkle_branch": ["11" * 32],
                "prompt_tokens": [320, 736, 3240],
                "height": 1234,
                "poi_version": 2,
                "clean_jobs": True,
            }},
        ]
        self.wfile.write(b"".join((json.dumps(m, separators=(",", ":")) + "\n").encode() for m in messages))
        self.wfile.flush()
        server.ready_sent.set()

        while True:
            line = self.rfile.readline()
            if not line:
                return
            request = json.loads(line.decode("utf-8"))
            if request.get("method") == "mining.submit":
                server.submitted_request = request
                server.submit_received.set()
                response = {"id": request["id"], "result": True, "error": None}
                self.wfile.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
                self.wfile.flush()
                if server.advance_job.wait(3):
                    notify = {"id": None, "method": "mining.notify", "params": {
                        "job_id": "job-002", "prev_hash": "44" * 32,
                        "coinbase1": "0102", "coinbase2": "0304",
                        "version": "20000000", "bits": "1d00ffff", "ntime": "6553f101",
                        "merkle_branch": [], "prompt_tokens": [320, 736, 3240],
                        "height": 1235, "poi_version": 2, "clean_jobs": True,
                    }}
                    self.wfile.write((json.dumps(notify, separators=(",", ":")) + "\n").encode())
                    self.wfile.flush()
                    server.next_job_sent.set()
                while self.rfile.readline():
                    pass
                return


class _StratumServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _StratumHandler)
        self.ready_sent = threading.Event()
        self.submit_received = threading.Event()
        self.advance_job = threading.Event()
        self.next_job_sent = threading.Event()
        self.submitted_request = None
        self.subscribe_params = None
        self.authorize_params = None
        self.failure = None


class StratumClientTests(unittest.TestCase):
    def test_endpoint_and_target(self):
        self.assertEqual(parse_endpoint("vela.suprnova.cc:7601"), ("vela.suprnova.cc", 7601, False))
        self.assertEqual(parse_endpoint("stratum+ssl://example.net:7604"), ("example.net", 7604, True))
        self.assertEqual(target_from_difficulty(1), (1 << 256) - 1)
        self.assertEqual(target_from_difficulty(2), ((1 << 256) - 1) // 2)
        with self.assertRaises(ValueError):
            target_from_difficulty(0)

    def test_pool_job_header_reconstruction(self):
        import hashlib
        import struct

        job = {
            "coinbase1": "0102",
            "extranonce1": "a1b2",
            "extranonce2": "d00df00d",
            "coinbase2": "0304",
            "merkle_branch": ["11" * 32, "22" * 32],
            "version": "20000000",
            "prev_hash": "33" * 32,
            "ntime": "6553f100",
            "bits": "1d00ffff",
            "target": target_from_difficulty(2048),
        }
        def sha256d(data):
            return hashlib.sha256(hashlib.sha256(data).digest()).digest()
        expected_coinbase = bytes.fromhex("0102a1b2d00df00d0304")
        expected_root = sha256d(expected_coinbase)
        expected_root = sha256d(expected_root + bytes.fromhex("11" * 32))
        expected_root = sha256d(expected_root + bytes.fromhex("22" * 32))
        expected_header = (struct.pack("<I", 0x20000000)
                           + bytes.fromhex("33" * 32)[::-1]
                           + expected_root
                           + struct.pack("<I", 0x6553f100)
                           + struct.pack("<I", 0x1d00ffff))
        header, root, target = build_stratum_header76(job)
        self.assertEqual(header, expected_header)
        self.assertEqual(root, expected_root)
        self.assertEqual(target, job["target"])
        self.assertEqual(len(header), 76)

    def test_named_params_handshake_job_and_submit(self):
        server = _StratumServer()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"127.0.0.1:{server.server_address[1]}"
        client = StratumClient(endpoint, "vela1example.worker", "x", user_agent="ExerMiner-test")
        try:
            client.start()
            self.assertTrue(server.ready_sent.wait(3), "mock server did not send a job")
            job = client.wait_for_job(timeout=3)
            self.assertIsNotNone(job, "client failed to become ready")
            self.assertEqual(server.subscribe_params, {"user_agent": "ExerMiner-test"})
            self.assertEqual(server.authorize_params, {"wallet": "vela1example.worker", "password": "x"})
            self.assertEqual(job["job_id"], "job-001")
            self.assertEqual(job["extranonce1"], "a1b2c3d4")
            self.assertEqual(job["model_hash"], "ab" * 32)
            self.assertEqual(job["difficulty"], 2048.0)
            self.assertEqual(job["prompt_tokens"], [320, 736, 3240])
            self.assertEqual(job["extranonce2"], client.extranonce2)

            self.assertTrue(client.submit_share(job, "0000002a", "aa" * 32768, "bb" * 32))
            self.assertTrue(server.submit_received.wait(3), "mock server did not receive mining.submit")
            sent = server.submitted_request
            self.assertEqual(sent["method"], "mining.submit")
            self.assertEqual(sent["params"]["worker"], "vela1example.worker")
            self.assertEqual(sent["params"]["job_id"], "job-001")
            self.assertEqual(sent["params"]["extranonce2"], job["extranonce2"])
            deadline = time.monotonic() + 2
            while client.accepted != 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(client.accepted, 1)
            self.assertEqual(client.submitted, 1)
            self.assertIsNone(server.failure)

            server.advance_job.set()
            self.assertTrue(server.next_job_sent.wait(3), "mock server did not send the next job")
            deadline = time.monotonic() + 2
            next_job = None
            while time.monotonic() < deadline:
                next_job = client.current_job()
                if next_job and next_job["job_id"] == "job-002":
                    break
                time.sleep(0.01)
            self.assertIsNotNone(next_job)
            self.assertEqual(next_job["job_id"], "job-002")
            self.assertFalse(client.submit_share(job, "0000002b", "aa" * 32768, "bb" * 32))
            self.assertEqual(client.discarded, 1)
            self.assertEqual(client.submitted, 1)
        finally:
            client.stop()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
