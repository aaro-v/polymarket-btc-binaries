"""HTTP GET with retries and back-off on rate limits."""

from __future__ import annotations

import threading
import time

import requests

_LOCAL = threading.local()


def _session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
        _LOCAL.session.headers["User-Agent"] = "Mozilla/5.0"
    return _LOCAL.session


def get(url: str, params: dict | None = None, retries: int = 6, timeout: float = 60.0) -> requests.Response | None:
    """Return the response, or None for a 404 or after repeated failures."""

    for attempt in range(retries):
        try:
            response = _session().get(url, params=params, timeout=timeout)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if response.status_code == 200:
            return response
        if response.status_code == 404:
            return None
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2**attempt)
            continue
        return None
    return None


def get_json(url: str, params: dict | None = None, **kwargs):
    response = get(url, params, **kwargs)
    return None if response is None else response.json()
