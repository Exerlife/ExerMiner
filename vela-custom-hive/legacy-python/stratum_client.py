#!/usr/bin/env python3
"""Vela named-parameter Stratum client.

This module implements the pool wire protocol independently of the GPU engine.
It owns the TCP/TLS session, job lifecycle, vardiff target and share submission.
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
import math
import secrets
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

LOG = logging.getLogger("vela-miner")

# Vela PoI v2 uses CLIP: prompt body <= 75 IDs, each ID < BOS (49406).
# BOS/EOS are inserted by sdint.context_ids(); they are not sent in prompt_tokens.
MAX_PROMPT_TOKENS = 75
PROMPT_TOKEN_LIMIT = 49406


def validate_prompt_tokens(tokens):
    """Validate the pool's prompt body using the Vela v2 consensus limits."""
    if (not isinstance(tokens, list)
            or not 1 <= len(tokens) <= MAX_PROMPT_TOKENS
            or any(not isinstance(x, int) or isinstance(x, bool)
                   or not 0 <= x < PROMPT_TOKEN_LIMIT for x in tokens)):
        raise ValueError(
            "mining.notify prompt_tokens must contain 1..75 integer token IDs in [0, 49405]"
        )
    return tokens


def parse_endpoint(value: str):
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty Stratum endpoint")
    if "://" not in raw:
        raw = "stratum://" + raw
    parsed = urlsplit(raw)
    if parsed.scheme not in ("stratum", "stratum+tcp", "stratum+ssl", "stratum+tls"):
        raise ValueError("Stratum URL scheme must be stratum:// or stratum+ssl://")
    if not parsed.hostname or not parsed.port:
        raise ValueError("Stratum endpoint must contain host and port")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("put wallet/password in their separate fields; endpoint must be host:port")
    tls = parsed.scheme in ("stratum+ssl", "stratum+tls") or parsed.port == 7604
    return parsed.hostname, parsed.port, tls


def target_from_difficulty(difficulty: float) -> int:
    """Vela pool share target; one difficulty unit represents one inference."""
    try:
        value = float(difficulty)
    except (TypeError, ValueError):
        raise ValueError("pool sent a non-numeric share difficulty")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid share difficulty: {difficulty!r}")
    divisor = max(1, math.ceil(value))
    return ((1 << 256) - 1) // divisor


def build_stratum_header76(job):
    """Rebuild the 76-byte Vela block-header prefix from a pool-issued job."""
    def sha256d(data):
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    coinbase = (bytes.fromhex(job["coinbase1"])
                + bytes.fromhex(job["extranonce1"])
                + bytes.fromhex(job["extranonce2"])
                + bytes.fromhex(job["coinbase2"]))
    merkle = sha256d(coinbase)
    for branch_hex in job["merkle_branch"]:
        merkle = sha256d(merkle + bytes.fromhex(branch_hex))
    header76 = (struct.pack("<I", int(job["version"], 16))
                + bytes.fromhex(job["prev_hash"])[::-1]
                + merkle
                + struct.pack("<I", int(job["ntime"], 16))
                + struct.pack("<I", int(job["bits"], 16)))
    if len(header76) != 76:
        raise ValueError(f"constructed Stratum header prefix must be 76 bytes, got {len(header76)}")
    target = int(job["target"])
    if not 0 <= target < (1 << 256):
        raise ValueError("Stratum share target is outside uint256 range")
    return header76, merkle, target


class StratumClient:
    def __init__(self, endpoint: str, wallet: str, password: str = "x", user_agent: str = "ExerMiner/0.2.2"):
        if not wallet.startswith("vela1"):
            raise ValueError("Stratum username must start with a mainnet VELA address (vela1...)")
        self.host, self.port, self.tls = parse_endpoint(endpoint)
        self.wallet = wallet
        self.password = password or "x"
        self.user_agent = user_agent

        self._state_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._condition = threading.Condition(self._state_lock)
        self._socket = None
        self._reader = None
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._generation = 0
        self._connection_id = 0
        self._job = None
        self._extranonce1 = ""
        self._model_hash = ""
        self._difficulty = 2048.0
        self._subscribed = False
        self._authorized = False
        self._last_error = None
        self._next_id = 10
        self._pending = {}
        self._extranonce2 = secrets.token_hex(4)  # pool dialect expects a four-byte extranonce2
        self.accepted = 0
        self.rejected = 0
        self.stale = 0
        self.submitted = 0
        self.discarded = 0

    @property
    def extranonce2(self):
        return self._extranonce2

    @property
    def ready(self):
        return self._ready.is_set()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vela-stratum", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._send_lock:
            sock = self._socket
            reader = self._reader
            self._socket = None
            self._reader = None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if reader is not None:
                try:
                    reader.close()
                except OSError:
                    pass
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        with self._condition:
            self._ready.clear()
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def current_job(self):
        with self._state_lock:
            if not self._ready.is_set() or self._job is None:
                return None
            return dict(self._job)

    def wait_for_job(self, after_generation=None, timeout=1.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop.is_set():
                if self._ready.is_set() and self._job is not None:
                    if after_generation is None or self._job["generation"] != after_generation:
                        return dict(self._job)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.5))
        return None

    def wait_ready(self, timeout=120):
        if self._ready.wait(timeout):
            return True
        with self._state_lock:
            detail = self._last_error
        raise TimeoutError(f"timed out waiting for Stratum authorization/job: {detail or 'no job received'}")

    def _send(self, message):
        wire = (json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        with self._send_lock:
            return self._send_wire_locked(wire)

    def _send_wire_locked(self, wire):
        """Send while holding _send_lock so reconnect/close cannot swap the socket."""
        sock = self._socket
        if sock is None:
            return False
        try:
            sock.sendall(wire)
            return True
        except OSError as exc:
            LOG.warning("stratum_send_failed error=%s", exc)
            return False

    def _run(self):
        backoff = 2
        while not self._stop.is_set():
            sock = None
            try:
                LOG.info("stratum_connecting host=%s port=%d tls=%s", self.host, self.port, self.tls)
                sock = socket.create_connection((self.host, self.port), timeout=15)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.tls:
                    context = ssl.create_default_context()
                    sock = context.wrap_socket(sock, server_hostname=self.host)
                sock.settimeout(10.0)
                reader = None
                with self._send_lock:
                    self._socket = sock
                    self._reader = None
                with self._condition:
                    self._connection_id += 1
                    self._subscribed = False
                    self._authorized = False
                    self._extranonce1 = ""
                    self._model_hash = ""
                    self._job = None
                    self._pending.clear()
                    self._ready.clear()
                    self._last_error = None
                    self._condition.notify_all()

                if not self._send({"id": 1, "method": "mining.subscribe", "params": {"user_agent": self.user_agent}}):
                    raise ConnectionError("could not send mining.subscribe")
                if not self._send({"id": 2, "method": "mining.authorize", "params": {"wallet": self.wallet, "password": self.password}}):
                    raise ConnectionError("could not send mining.authorize")

                LOG.info("stratum_socket_connected awaiting_subscribe_and_authorize")
                backoff = 2
                receive_buffer = bytearray()
                last_rx = time.monotonic()
                last_ping = last_rx
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(65536)
                    except socket.timeout:
                        now = time.monotonic()
                        if now - last_rx >= 300:
                            raise ConnectionError("no data from pool for five minutes")
                        if now - last_rx >= 120 and now - last_ping >= 120:
                            with self._condition:
                                ping_id = self._next_id
                                self._next_id += 1
                            if not self._send({"id": ping_id, "method": "mining.ping", "params": {}}):
                                raise ConnectionError("could not send Stratum keepalive ping")
                            last_ping = now
                        continue
                    if not chunk:
                        raise ConnectionError("pool closed the connection")
                    last_rx = time.monotonic()
                    receive_buffer.extend(chunk)
                    if len(receive_buffer) > 2 * 1024 * 1024 and b"\n" not in receive_buffer:
                        raise ValueError("Stratum message exceeds 2 MiB")
                    while True:
                        newline = receive_buffer.find(b"\n")
                        if newline < 0:
                            break
                        line = bytes(receive_buffer[:newline]).strip()
                        del receive_buffer[:newline + 1]
                        if line:
                            self._handle(json.loads(line.decode("utf-8", errors="strict")))
            except Exception as exc:
                if not self._stop.is_set():
                    with self._state_lock:
                        self._last_error = str(exc)
                    LOG.warning("stratum_connection_error error=%s retry_seconds=%d", exc, backoff)
            finally:
                with self._send_lock:
                    old_sock = self._socket
                    old_reader = self._reader
                    self._socket = None
                    self._reader = None
                if old_reader is not None:
                    try:
                        old_reader.close()
                    except OSError:
                        pass
                if old_sock is not None:
                    try:
                        old_sock.close()
                    except OSError:
                        pass
                elif sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                with self._condition:
                    self._ready.clear()
                    self._job = None
                    self._subscribed = False
                    self._authorized = False
                    self._extranonce1 = ""
                    self._model_hash = ""
                    self._condition.notify_all()
            if not self._stop.is_set():
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    def _set_ready_locked(self):
        if self._subscribed and self._authorized and self._job is not None:
            self._ready.set()
            self._condition.notify_all()

    def _handle(self, message):
        if not isinstance(message, dict):
            raise ValueError("Stratum message must be a JSON object")
        method = message.get("method")
        if method:
            params = message.get("params", {})
            if method == "mining.set_difficulty":
                if isinstance(params, dict):
                    difficulty = params.get("difficulty")
                elif isinstance(params, list) and params:
                    difficulty = params[0]
                else:
                    raise ValueError("mining.set_difficulty missing difficulty")
                target_from_difficulty(difficulty)
                with self._condition:
                    self._difficulty = float(difficulty)
                    LOG.info("stratum_difficulty=%s", self._difficulty)
                    # Vela pools issue a corresponding notify with a new job id/target.
            elif method == "mining.notify":
                if isinstance(params, list):
                    if len(params) != 1 or not isinstance(params[0], dict):
                        raise ValueError("mining.notify must contain one named-parameter object")
                    params = params[0]
                if not isinstance(params, dict):
                    raise ValueError("mining.notify params must be an object")
                required = ("job_id", "prev_hash", "coinbase1", "coinbase2", "version", "bits", "ntime", "merkle_branch", "prompt_tokens")
                missing = [field for field in required if field not in params]
                if missing:
                    raise ValueError("mining.notify missing fields: " + ", ".join(missing))
                job_id = str(params["job_id"])
                prev_hash = str(params["prev_hash"]).lower()
                if len(prev_hash) != 64 or any(ch not in "0123456789abcdef" for ch in prev_hash):
                    raise ValueError("mining.notify prev_hash must be a 32-byte hex hash")
                for name in ("coinbase1", "coinbase2", "version", "bits", "ntime"):
                    raw = str(params[name])
                    if len(raw) % 2 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
                        raise ValueError(f"mining.notify {name} is not valid hex")
                branches = params["merkle_branch"]
                tokens = validate_prompt_tokens(params["prompt_tokens"])
                if (not isinstance(branches, list)
                        or any(len(str(x)) != 64
                               or any(ch not in "0123456789abcdefABCDEF" for ch in str(x))
                               for x in branches)):
                    raise ValueError("mining.notify merkle_branch must contain 32-byte hex hashes")
                if len(str(params["version"])) != 8 or len(str(params["bits"])) != 8 or len(str(params["ntime"])) != 8:
                    raise ValueError("mining.notify version/bits/ntime must each be 4-byte hex")
                notify_model_hash = str(params.get("model_hash") or "").lower()
                if notify_model_hash and (len(notify_model_hash) != 64
                                           or any(ch not in "0123456789abcdef" for ch in notify_model_hash)):
                    raise ValueError("mining.notify model_hash must be empty or 32-byte hex")
                with self._condition:
                    if notify_model_hash:
                        self._model_hash = notify_model_hash
                    self._generation += 1
                    job = {
                        "job_id": job_id,
                        "prev_hash": prev_hash,
                        "coinbase1": str(params["coinbase1"]).lower(),
                        "coinbase2": str(params["coinbase2"]).lower(),
                        "version": str(params["version"]).lower(),
                        "bits": str(params["bits"]).lower(),
                        "ntime": str(params["ntime"]).lower(),
                        "merkle_branch": [str(x).lower() for x in branches],
                        "prompt_tokens": list(tokens),
                        "height": int(params.get("height", 0)),
                        "poi_version": int(params.get("poi_version", 2)),
                        "model_hash": str(params.get("model_hash") or self._model_hash).lower(),
                        "clean_jobs": bool(params.get("clean_jobs", params.get("clean", False))),
                        "difficulty": float(self._difficulty),
                        "target": target_from_difficulty(self._difficulty),
                        "extranonce1": self._extranonce1,
                        "extranonce2": self._extranonce2,
                        "generation": self._generation,
                        "connection_id": self._connection_id,
                    }
                    self._job = job
                    self._set_ready_locked()
                    LOG.info("stratum_job id=%s height=%s clean=%s difficulty=%s poi_version=%s tokens=%d",
                             job_id, job["height"], job["clean_jobs"], job["difficulty"],
                             job["poi_version"], len(tokens))
                    self._condition.notify_all()
            elif method == "mining.ping":
                self._send({"id": message.get("id"), "result": "pong", "error": None})
            else:
                LOG.debug("stratum_unhandled_notification method=%s", method)
            return

        request_id = message.get("id")
        if request_id is None:
            return
        error = message.get("error")
        result = message.get("result")
        ok = error is None and result is not None and result is not False
        if request_id == 1:
            if not ok or not isinstance(result, dict):
                raise ConnectionError(f"mining.subscribe rejected: {error or result}")
            extranonce1 = str(result.get("extranonce1", "")).lower()
            if not extranonce1 or len(extranonce1) % 2 or any(ch not in "0123456789abcdef" for ch in extranonce1):
                raise ValueError("mining.subscribe returned an invalid extranonce1")
            with self._condition:
                self._extranonce1 = extranonce1
                model = result.get("model", {})
                if isinstance(model, dict):
                    self._model_hash = str(model.get("model_hash", "")).lower()
                if not self._model_hash:
                    self._model_hash = str(result.get("model_hash", "")).lower()
                self._subscribed = True
                if self._job is not None:
                    self._job["extranonce1"] = extranonce1
                    if not self._job.get("model_hash"):
                        self._job["model_hash"] = self._model_hash
                LOG.info("stratum_subscribed extranonce1_bytes=%d", len(extranonce1) // 2)
                self._set_ready_locked()
        elif request_id == 2:
            if not ok:
                raise ConnectionError(f"mining.authorize rejected: {error or result}")
            with self._condition:
                self._authorized = True
                LOG.info("stratum_authorized worker=%s", self.wallet)
                self._set_ready_locked()
        else:
            with self._condition:
                method_name = self._pending.pop(request_id, "unknown")
                if method_name == "mining.submit":
                    if ok:
                        self.accepted += 1
                        LOG.info("share_accepted accepted=%d submitted=%d", self.accepted, self.submitted)
                    else:
                        self.rejected += 1
                        err_text = json.dumps(error if error is not None else result, ensure_ascii=False)
                        if ("code 21" in err_text or '"code":21' in err_text
                                or "stale" in err_text.lower()):
                            self.stale += 1
                        LOG.warning("share_rejected error=%s accepted=%d rejected=%d", err_text, self.accepted, self.rejected)

    def submit_share(self, job, nonce_hex: str, x0_hex: str, hashpoi_hex: str):
        """Submit only if this exact job/session is still current; serialize against reconnect."""
        request_id = None
        params = {
            "worker": self.wallet,
            "job_id": job["job_id"],
            "extranonce2": job["extranonce2"],
            "ntime": job["ntime"],
            "nonce": nonce_hex,
            "x0_hex": x0_hex,
            "hashpoi_hex": hashpoi_hex,
        }
        with self._send_lock:
            with self._condition:
                current = self._job
                if not self._ready.is_set() or self._socket is None or current is None:
                    self.discarded += 1
                    LOG.warning("share_discarded reason=stratum_disconnected")
                    return False
                if job.get("connection_id") != self._connection_id:
                    self.discarded += 1
                    LOG.warning("share_discarded reason=previous_connection job=%s", job.get("job_id"))
                    return False
                if job.get("generation") != current.get("generation"):
                    self.discarded += 1
                    LOG.info("share_discarded reason=superseded_job job=%s", job.get("job_id"))
                    return False
                request_id = self._next_id
                self._next_id += 1
                self._pending[request_id] = "mining.submit"
                self.submitted += 1
                wire = (json.dumps({"id": request_id, "method": "mining.submit", "params": params},
                                   separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                if not self._send_wire_locked(wire):
                    self._pending.pop(request_id, None)
                    self.submitted -= 1
                    self.discarded += 1
                    return False
        LOG.info("share_submitted job=%s nonce=%s extranonce2=%s", job["job_id"], nonce_hex, job["extranonce2"])
        return True
