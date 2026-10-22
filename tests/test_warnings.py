"""
Warnings of spectrojotometer are returned to the client, per request.
"""

import logging
import pathlib
import threading

import numpy as np
from fastapi.testclient import TestClient

from visualbondweb import api

from .test_evaluate import ALL_CONFS, ring_model, spin_text

client = TestClient(api.app)
FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def new_session_with_ring():
    sid = client.post("/session/new").json()["session_id"]
    api.sessions[sid]["model"] = ring_model()
    return sid


def evaluate(sid, text):
    return client.post(
        "/evaluate",
        json={"session_id": sid, "configs_text": text, "energy_tolerance": 1e-3},
    ).json()


def test_evaluate_without_warnings():
    sid = new_session_with_ring()
    data = evaluate(sid, spin_text(api.sessions[sid]["model"], ALL_CONFS))
    assert data["warnings"] == []


def test_evaluate_reports_incompatible_model():
    sid = new_session_with_ring()
    data = evaluate(sid, spin_text(api.sessions[sid]["model"], ALL_CONFS, shift=1.0))
    assert data["incompatible"]
    assert any("incompatible" in w for w in data["warnings"])


def test_upload_reports_library_warnings():
    path = FIXTURES / "synthetic_mult2.struct"
    with path.open("rb") as f:
        res = client.post(
            "/model/upload",
            files={"file": (path.name, f)},
            data={"primitive_cell": "true"},
        )
    assert res.status_code == 200, res.text
    assert any("primitive_cell=True is not supported" in w for w in res.json()["warnings"])


def test_upload_without_warnings():
    path = FIXTURES / "synthetic_mult2.struct"
    with path.open("rb") as f:
        res = client.post("/model/upload", files={"file": (path.name, f)})
    assert res.status_code == 200, res.text
    assert res.json()["warnings"] == []


def test_warnings_of_other_threads_are_not_collected():
    """Concurrent requests share the library logger: each one only gets its own."""
    log = logging.getLogger("spectrojotometer.test")
    inside, done = threading.Event(), threading.Event()

    def other_request():
        inside.wait(5)
        log.warning("from another request")
        done.set()

    worker = threading.Thread(target=other_request)
    worker.start()
    with api.library_warnings() as warnings:
        log.warning("from this request")
        inside.set()
        done.wait(5)
    worker.join()
    assert warnings == ["from this request"]


def test_handler_is_removed_after_the_block():
    library_logger = logging.getLogger("spectrojotometer")
    before = list(library_logger.handlers)
    with api.library_warnings():
        pass
    assert library_logger.handlers == before
