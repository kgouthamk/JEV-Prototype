"""Headless UI checks: the server-side API key must never reach the browser."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import jev_client

SECRET = "sk-server-secret-do-not-leak"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("BEATS_API_KEY", SECRET)
    return AppTest.from_file("app.py", default_timeout=30).run()


def test_server_key_not_in_any_widget(app):
    assert not app.exception
    widgets = [*app.sidebar.text_input, *app.text_input, *app.text_area]
    assert all(SECRET not in str(w.value) for w in widgets)


def test_endpoint_locked_while_using_server_key(app):
    endpoint = next(w for w in app.sidebar.text_input if w.label == "Endpoint")
    assert endpoint.disabled and endpoint.value == jev_client.DEFAULT_ENDPOINT


def test_own_key_unlocks_endpoint(app):
    own = next(w for w in app.sidebar.text_input if "own" in w.label.lower())
    own.input("visitor-key").run()
    endpoint = next(w for w in app.sidebar.text_input if w.label == "Endpoint")
    assert not endpoint.disabled
