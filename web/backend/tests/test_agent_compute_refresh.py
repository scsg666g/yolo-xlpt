"""On-demand Agent sampling is additive to the existing report heartbeat."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def agent():
    source = Path(__file__).resolve().parents[2] / "agent/monitor_agent.py"
    spec = importlib.util.spec_from_file_location("refresh_agent", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report_center():
    requests = []
    responses = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(responses.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_refresh_command_generates_second_ordered_sample(agent, report_center, monkeypatch):
    url, requests, responses = report_center
    request_id = "a" * 32
    config = {"center_url": url, "node_id": "b" * 32, "token": "test-token",
              "report_interval_seconds": 5, "nvidia_smi_bin": "nvidia-smi"}
    responses.extend([
        {"report_interval_seconds": 9, "refresh_request_id": request_id},
        {"report_interval_seconds": 9, "refresh_request_id": "c" * 32},
    ])
    collected = []

    def sample(*_args):
        sequence = len(collected) + 1
        payload = {"agent_version": agent.VERSION, "boot_id": "d" * 32,
                   "report_seq": sequence, "compute": {"sampled_at": f"sample-{sequence}"}}
        collected.append(payload)
        return payload

    monkeypatch.setattr(agent, "collect", sample)
    first = agent.collect(config["nvidia_smi_bin"], config)
    interval, command = agent.send(config, first)
    refreshed = agent.fulfill_refresh_request(config, command, threading.Event())

    assert interval == 9
    assert command == request_id
    assert len(requests) == len(collected) == 2  # Do not recursively process next command.
    assert requests[0]["report_seq"] == 1 and requests[1]["report_seq"] == 2
    assert requests[0]["compute"]["sampled_at"] != requests[1]["compute"]["sampled_at"]
    assert "refresh_request_id" not in requests[0]
    assert requests[1]["refresh_request_id"] == request_id
    assert refreshed is collected[1]


@pytest.mark.parametrize("response", [{"report_interval_seconds": 5},
                                     {"report_interval_seconds": 5, "refresh_request_id": "bad"}])
def test_legacy_or_malformed_refresh_command_is_ignored(agent, report_center, response):
    url, requests, responses = report_center
    responses.append(response)
    config = {"center_url": url, "node_id": "b" * 32, "token": "test-token",
              "report_interval_seconds": 5}
    assert agent.send(config, {"compute": {}}) == (5, None)
    assert len(requests) == 1
