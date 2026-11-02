"""Helpers that turn files and web pages into plain text."""
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

MAX_CHARS = 2_000_000


def read_text_files(directory: str) -> list[tuple[str, str]]:
    """Return [(filename, text)] for every .txt / .md file in a folder."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    files = sorted([*folder.glob("*.txt"), *folder.glob("*.md")])
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def _assert_public_url(url: str) -> None:
    """Basic SSRF protection: only http(s) URLs that resolve to public IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host: {parsed.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("URL points to a non-public address")


def fetch_url_text(url: str, timeout: int = 20) -> str:
    """Download a web page and return its visible text."""
    for _ in range(4):  # follow up to 3 redirects, checking every hop
        _assert_public_url(url)
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (NepalTravelBot)"},
            allow_redirects=False,
        )
        if resp.is_redirect:
            url = urljoin(url, resp.headers.get("location", ""))
            continue
        resp.raise_for_status()
        break
    else:
        raise ValueError("Too many redirects")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:MAX_CHARS]
