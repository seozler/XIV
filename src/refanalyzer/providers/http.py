"""Small HTTP helpers shared by the managed-API adapters."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ProviderError(RuntimeError):
    pass


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET", "PUT", "POST"}), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def check(resp: requests.Response, what: str) -> Any:
    if resp.status_code >= 400:
        raise ProviderError(f"{what} failed: HTTP {resp.status_code}: {resp.text[:500]}")
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return resp.content


def poll(fetch: Callable[[], dict], done: Callable[[dict], bool], failed: Callable[[dict], str | None],
         interval: float, timeout: float, what: str) -> dict:
    t0 = time.time()
    while True:
        data = fetch()
        err = failed(data)
        if err:
            raise ProviderError(f"{what} failed: {err}")
        if done(data):
            return data
        if time.time() - t0 > timeout:
            raise ProviderError(f"{what} timed out after {timeout:.0f}s")
        time.sleep(interval)


def download(sess: requests.Session, url: str, dst: str | Path) -> Path:
    dst = Path(dst)
    with sess.get(url, stream=True, timeout=600) as r:
        if r.status_code >= 400:
            raise ProviderError(f"download failed: HTTP {r.status_code}")
        with dst.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dst
