"""Rich internet and utility tool suite for Colibri AI Agents (`src/ai/tools/internet.py`).

Reuses ``ui.web`` (`search`, `page`, `_public_url`) when available, with a
standalone SSRF-safe fallback. Every tool is decorated for LangChain/LangGraph
compatibility (with a zero-dependency fallback when ``langchain_core`` is not
installed) and exported alongside ``OPENAI_TOOLS_SCHEMA`` and ``TOOL_REGISTRY``.
"""
from __future__ import annotations

import ast
import html
import ipaddress
import json
import math
import operator
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repo root is importable so `ui.web` can be reused seamlessly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from ui import web as _ui_web  # type: ignore
except Exception:  # noqa: BLE001
    _ui_web = None  # type: ignore

try:
    from langchain_core.tools import tool as _lc_tool  # type: ignore
except Exception:  # noqa: BLE001
    _lc_tool = None


def _tool_decorator(func: Callable[..., str]) -> Any:
    """Wrap a function with LangChain's ``@tool`` if installed, while preserving direct callability."""
    if _lc_tool is not None:
        lc_obj = _lc_tool(func)

        class _CallableToolWrapper:
            """Adapter that behaves as both a LangChain BaseTool and a plain Python function."""

            def __init__(self, wrapped_lc: Any, raw_fn: Callable[..., str]) -> None:
                self._lc = wrapped_lc
                self._fn = raw_fn
                self.name = getattr(wrapped_lc, "name", raw_fn.__name__)
                self.description = getattr(
                    wrapped_lc, "description", (raw_fn.__doc__ or "").strip()
                )
                self.args_schema = getattr(wrapped_lc, "args_schema", None)
                self.__name__ = raw_fn.__name__
                self.__doc__ = raw_fn.__doc__

            def __call__(self, *args: Any, **kwargs: Any) -> str:
                return self._fn(*args, **kwargs)

            def invoke(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
                if isinstance(input_data, dict):
                    return self._fn(**input_data)
                return self._lc.invoke(input_data, *args, **kwargs)

            def __getattr__(self, item: str) -> Any:
                return getattr(self._lc, item)

        return _CallableToolWrapper(lc_obj, func)

    # Lightweight stdlib fallback when langchain_core is not installed
    func.name = func.__name__  # type: ignore[attr-defined]
    func.description = (func.__doc__ or "").strip()  # type: ignore[attr-defined]
    func.invoke = lambda inp, _f=func: _f(**inp) if isinstance(inp, dict) else _f(inp)  # type: ignore[attr-defined]
    return func


UA = "Mozilla/5.0 (compatible; ColibriAgent/1.0; +https://github.com/JustVugg/colibri)"


_PUBLIC_HOST_CACHE: Dict[Tuple[str, int], bool] = {}


def _public_url(url: str) -> str:
    """Verify that ``url`` uses http/https and resolves strictly to global public IP addresses."""
    if _ui_web is not None and hasattr(_ui_web, "_public_url"):
        return _ui_web._public_url(url)

    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("http and https links only")
    port = u.port or (443 if u.scheme == "https" else 80)
    cache_key = (u.hostname.lower(), port)
    cached = _PUBLIC_HOST_CACHE.get(cache_key)
    if cached is True:
        return url
    if cached is False:
        raise ValueError(f"not a public address: {u.hostname}")

    for info in socket.getaddrinfo(u.hostname, port):
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            _PUBLIC_HOST_CACHE[cache_key] = False
            raise ValueError(f"not a public address: {u.hostname}")
    _PUBLIC_HOST_CACHE[cache_key] = True
    return url


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that validates every redirect target against SSRF rules."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_public(
    url: str,
    *,
    accept: str = "application/json, text/html, application/xml;q=0.9, */*;q=0.8",
    timeout: float = 12.0,
    max_bytes: int = 3 << 20,
) -> Tuple[bytes, str, str]:
    """Open a public URL safely and return ``(raw_bytes, content_type, final_url)``."""
    _public_url(url)
    opener = urllib.request.build_opener(_CheckedRedirect)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": accept,
        },
    )
    with opener.open(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(max_bytes)
        final_url = resp.geturl()
    return raw, ctype, final_url


class _LinkAndTextExtractor(HTMLParser):
    """Extract outbound links and readable text from HTML when `ui.web` is supplemented."""

    SKIP = {
        "script",
        "style",
        "noscript",
        "svg",
        "template",
        "iframe",
        "nav",
        "footer",
        "form",
        "button",
        "select",
        "aside",
    }

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.skip_stack: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self._cur_href: Optional[str] = None
        self._cur_link_text: List[str] = []
        self.text_parts: List[str] = []
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "title":
            self._in_title = True
        if self.skip_stack:
            self.skip_stack.append(tag)
            return
        if tag in self.SKIP:
            self.skip_stack.append(tag)
            return
        attr_dict = dict(attrs)
        if tag == "a" and attr_dict.get("href"):
            href = urllib.parse.urljoin(self.base_url, attr_dict["href"] or "")
            if href.startswith(("http://", "https://")):
                self._cur_href = href
                self._cur_link_text = []
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "article", "section"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self.skip_stack:
            if tag in self.skip_stack:
                while self.skip_stack and self.skip_stack.pop() != tag:
                    pass
            return
        if tag == "a" and self._cur_href is not None:
            label = re.sub(r"\s+", " ", "".join(self._cur_link_text)).strip()
            if label and len(label) > 2:
                self.links.append((label[:100], self._cur_href))
            self._cur_href = None
            self._cur_link_text = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self.skip_stack:
            return
        self.text_parts.append(data)
        if self._cur_href is not None:
            self._cur_link_text.append(data)


def _extract_links_from_html(url: str, max_links: int = 12) -> List[Dict[str, str]]:
    """Fetch a page and extract up to ``max_links`` unique outbound HTTP(S) links."""
    try:
        raw, ctype, final_url = _open_public(
            url, accept="text/html,text/plain;q=0.9", timeout=10.0, max_bytes=1 << 20
        )
        if "html" not in ctype.lower():
            return []
        charset = re.search(r"charset=([\w-]+)", ctype, re.I)
        html_text = raw.decode(charset.group(1) if charset else "utf-8", errors="replace")
        parser = _LinkAndTextExtractor(final_url)
        parser.feed(html_text)
        seen = set()
        out: List[Dict[str, str]] = []
        for title, href in parser.links:
            if href in seen or href == final_url:
                continue
            seen.add(href)
            out.append({"title": title, "url": href})
            if len(out) >= max_links:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


@_tool_decorator
def web_search(query: str, max_results: int = 5) -> str:
    """Search the live internet (via duckduckgo_search, SearXNG, or DuckDuckGo HTML) and return titles, URLs, and snippets."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "Empty search query"})
    n = max(1, min(int(max_results), 10))

    # 1. Try duckduckgo_search.DDGS().text(...) when installed
    try:
        from duckduckgo_search import DDGS  # type: ignore

        ddgs_rows = list(DDGS().text(query, max_results=n) or [])
        if ddgs_rows:
            results = []
            for r in ddgs_rows[:n]:
                href = r.get("href") or r.get("url") or ""
                body_txt = r.get("body") or r.get("snippet") or ""
                title_txt = r.get("title") or ""
                results.append(
                    {
                        "title": title_txt,
                        "url": href,
                        "href": href,
                        "snippet": body_txt,
                        "body": body_txt,
                        "engine": "ddgs",
                    }
                )
            return json.dumps(
                {"q": query, "results": results, "answers": []},
                ensure_ascii=False,
            )
    except Exception:  # noqa: BLE001 - fall back to ui.web / SearXNG / DDG HTML
        pass

    try:
        if _ui_web is not None:
            res = _ui_web.search(query, n)
            return json.dumps(res, ensure_ascii=False)
        # Fallback if ui.web is unavailable
        body = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=body,
            headers={
                "User-Agent": UA,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            page_html = r.read(2 << 20).decode("utf-8", errors="replace")
        matches = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            page_html,
            re.I | re.S,
        )
        results = []
        for href, raw_title in matches[:n]:
            clean_title = re.sub(r"<[^>]+>", "", html.unescape(raw_title)).strip()
            results.append({"title": clean_title, "url": href, "snippet": ""})
        return json.dumps({"q": query, "results": results, "answers": []}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"web_search failed: {type(exc).__name__}: {exc}"})


@_tool_decorator
def fetch_webpage(url: str, max_chars: int = 4500, include_links: bool = True) -> str:
    """Fetch a public webpage (SSRF-safe) and return its clean readable text and optional outbound links."""
    url = (url or "").strip()
    if not url:
        return json.dumps({"error": "Empty URL"})
    max_chars = max(300, min(int(max_chars), 15000))
    try:
        if _ui_web is not None:
            page_data = _ui_web.page(url, max_chars=max_chars)
        else:
            raw, ctype, final_url = _open_public(
                url, accept="text/html,text/plain;q=0.9", timeout=10.0
            )
            charset = re.search(r"charset=([\w-]+)", ctype, re.I)
            decoded = raw.decode(charset.group(1) if charset else "utf-8", errors="replace")
            parser = _LinkAndTextExtractor(final_url)
            parser.feed(decoded)
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.text_parts))
            text = re.sub(r"\s*\n\s*", "\n", text).strip()
            page_data = {
                "url": final_url,
                "title": parser.title.strip()[:200],
                "text": text[:max_chars],
                "truncated": len(text) > max_chars,
                "chars": len(text),
            }

        if include_links:
            page_data["links"] = _extract_links_from_html(page_data.get("url", url), max_links=10)
        return json.dumps(page_data, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"})


@_tool_decorator
def fetch_webpage_content(url: str, max_chars: int = 4500) -> str:
    """Fetch a public URL (SSRF-safe) and extract its clean article text using trafilatura (with fallback to fetch_webpage)."""
    url = (url or "").strip()
    if not url:
        return json.dumps({"error": "Empty URL"})
    max_chars = max(300, min(int(max_chars), 15000))
    try:
        _public_url(url)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"})

    try:
        import trafilatura  # type: ignore

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
            )
            if extracted and extracted.strip():
                text = extracted.strip()
                return json.dumps(
                    {
                        "url": url,
                        "text": text[:max_chars],
                        "truncated": len(text) > max_chars,
                        "chars": len(text),
                        "extractor": "trafilatura",
                    },
                    ensure_ascii=False,
                )
    except Exception:  # noqa: BLE001 - fall back to fetch_webpage
        pass

    return fetch_webpage(url=url, max_chars=max_chars, include_links=False)


@_tool_decorator
def deep_web_research(query: str, top_pages: int = 3, chars_per_page: int = 2500) -> str:
    """Search the web and automatically fetch + extract the readable text of the top N results in one tool call.

    Ideal for local LLMs (`qwen36-35b`) because it gathers search snippets AND full page contents
    in a single round-trip, avoiding multiple prefill turns.
    """
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "Empty research query"})
    top_pages = max(1, min(int(top_pages), 5))
    chars_per_page = max(500, min(int(chars_per_page), 5000))

    raw_search = web_search(query=query, max_results=max(top_pages + 2, 6))
    try:
        search_data = json.loads(raw_search)
    except Exception:  # noqa: BLE001
        return raw_search

    if "error" in search_data:
        return raw_search

    results = search_data.get("results") or []
    candidate_items = [item for item in results if item.get("url")][:top_pages]

    def _fetch_one(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        u = item.get("url")
        if not u:
            return None
        try:
            page_json = fetch_webpage(url=u, max_chars=chars_per_page, include_links=False)
            page_obj = json.loads(page_json)
            if "error" not in page_obj and page_obj.get("text"):
                return {
                    "title": page_obj.get("title") or item.get("title"),
                    "url": page_obj.get("url") or u,
                    "snippet": item.get("snippet"),
                    "content": page_obj.get("text"),
                }
        except Exception:  # noqa: BLE001
            return None
        return None

    import concurrent.futures

    pages_extracted: List[Dict[str, Any]] = []
    if candidate_items:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidate_items)) as pool:
            for page_res in pool.map(_fetch_one, candidate_items):
                if page_res is not None:
                    pages_extracted.append(page_res)

    return json.dumps(
        {
            "query": query,
            "answers": search_data.get("answers", []),
            "search_results": results[:6],
            "extracted_pages": pages_extracted,
        },
        ensure_ascii=False,
    )


@_tool_decorator
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """Search Wikipedia and return authoritative article summaries and canonical URLs."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "Empty Wikipedia query"})
    sentences = max(1, min(int(sentences), 12))
    try:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
            }
        )
        search_url = f"https://en.wikipedia.org/w/api.php?{params}"
        raw, _, _ = _open_public(search_url, accept="application/json", timeout=10.0)
        data = json.loads(raw.decode("utf-8", errors="replace"))
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return json.dumps({"query": query, "results": [], "message": "No Wikipedia articles found."})

        articles: List[Dict[str, Any]] = []
        for hit in hits[:2]:
            title = hit.get("title", "")
            if not title:
                continue
            ext_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts|info",
                    "exintro": 1,
                    "explaintext": 1,
                    "exsentences": sentences,
                    "inprop": "url",
                    "titles": title,
                    "format": "json",
                }
            )
            ext_url = f"https://en.wikipedia.org/w/api.php?{ext_params}"
            ext_raw, _, _ = _open_public(ext_url, accept="application/json", timeout=10.0)
            ext_data = json.loads(ext_raw.decode("utf-8", errors="replace"))
            pages = ext_data.get("query", {}).get("pages", {})
            for page_info in pages.values():
                articles.append(
                    {
                        "title": page_info.get("title", title),
                        "url": page_info.get(
                            "fullurl",
                            f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                        ),
                        "summary": (page_info.get("extract") or "").strip()[:3000],
                    }
                )
        return json.dumps({"query": query, "articles": articles}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"wikipedia_search failed: {type(exc).__name__}: {exc}"})


@_tool_decorator
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Query the ArXiv API (`export.arxiv.org`) for scientific papers, authors, publication dates, abstracts, and links."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "Empty ArXiv query"})
    n = max(1, min(int(max_results), 10))
    try:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": n,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        url = f"http://export.arxiv.org/api/query?{params}"
        raw, _, _ = _open_public(url, accept="application/atom+xml,application/xml,text/xml", timeout=12.0)
        root = ET.fromstring(raw.decode("utf-8", errors="replace"))
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = re.sub(r"\s+", " ", (entry.findtext("atom:title", "", ns) or "")).strip()
            summary = re.sub(r"\s+", " ", (entry.findtext("atom:summary", "", ns) or "")).strip()
            published = (entry.findtext("atom:published", "", ns) or "")[:10]
            link = entry.findtext("atom:id", "", ns) or ""
            authors = [
                (a.findtext("atom:name", "", ns) or "").strip()
                for a in entry.findall("atom:author", ns)
            ]
            papers.append(
                {
                    "title": title,
                    "authors": [a for a in authors if a][:6],
                    "published": published,
                    "url": link,
                    "abstract": summary[:900],
                }
            )
        return json.dumps({"query": query, "papers": papers}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"arxiv_search failed: {type(exc).__name__}: {exc}"})


@_tool_decorator
def hackernews_search(query: str, max_results: int = 5) -> str:
    """Search Hacker News via the Algolia API for tech news, discussions, points, and URLs."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "Empty Hacker News query"})
    n = max(1, min(int(max_results), 10))
    try:
        params = urllib.parse.urlencode({"query": query, "hitsPerPage": n, "tags": "story"})
        url = f"https://hn.algolia.com/api/v1/search?{params}"
        raw, _, _ = _open_public(url, accept="application/json", timeout=10.0)
        data = json.loads(raw.decode("utf-8", errors="replace"))
        stories: List[Dict[str, Any]] = []
        for hit in data.get("hits") or []:
            oid = hit.get("objectID", "")
            stories.append(
                {
                    "title": hit.get("title"),
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                    "hn_discussion": f"https://news.ycombinator.com/item?id={oid}",
                    "points": hit.get("points"),
                    "num_comments": hit.get("num_comments"),
                    "author": hit.get("author"),
                    "date": (hit.get("created_at") or "")[:10],
                }
            )
        return json.dumps({"query": query, "stories": stories}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"hackernews_search failed: {type(exc).__name__}: {exc}"})


@_tool_decorator
def github_repo_search(query: str, max_results: int = 5) -> str:
    """Search GitHub repositories via the GitHub REST API and return repo names, stars, languages, and descriptions."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "Empty GitHub query"})
    n = max(1, min(int(max_results), 10))
    try:
        params = urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": n})
        url = f"https://api.github.com/search/repositories?{params}"
        raw, _, _ = _open_public(
            url, accept="application/vnd.github+json, application/json", timeout=10.0
        )
        data = json.loads(raw.decode("utf-8", errors="replace"))
        repos: List[Dict[str, Any]] = []
        for item in (data.get("items") or [])[:n]:
            repos.append(
                {
                    "full_name": item.get("full_name"),
                    "url": item.get("html_url"),
                    "description": (item.get("description") or "")[:300],
                    "stars": item.get("stargazers_count"),
                    "language": item.get("language"),
                    "updated_at": (item.get("updated_at") or "")[:10],
                }
            )
        return json.dumps({"query": query, "repositories": repos}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"github_repo_search failed: {type(exc).__name__}: {exc}"})


@_tool_decorator
def weather_lookup(location: str) -> str:
    """Look up current weather and daily forecast for a city/location using Open-Meteo (no API key required)."""
    location = (location or "").strip()
    if not location:
        return json.dumps({"error": "Empty location"})
    try:
        geo_params = urllib.parse.urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{geo_params}"
        geo_raw, _, _ = _open_public(geo_url, accept="application/json", timeout=10.0)
        geo_data = json.loads(geo_raw.decode("utf-8", errors="replace"))
        results = geo_data.get("results") or []
        if not results:
            return json.dumps({"location": location, "error": "Location not found"})

        place = results[0]
        lat, lon = place.get("latitude"), place.get("longitude")
        wx_params = urllib.parse.urlencode(
            {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 3,
            }
        )
        wx_url = f"https://api.open-meteo.com/v1/forecast?{wx_params}"
        wx_raw, _, _ = _open_public(wx_url, accept="application/json", timeout=10.0)
        wx_data = json.loads(wx_raw.decode("utf-8", errors="replace"))
        return json.dumps(
            {
                "location": f"{place.get('name')}, {place.get('country', '')}".strip(", "),
                "coordinates": {"latitude": lat, "longitude": lon},
                "current": wx_data.get("current"),
                "current_units": wx_data.get("current_units"),
                "daily": wx_data.get("daily"),
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"weather_lookup failed: {type(exc).__name__}: {exc}"})


@_tool_decorator
def fetch_json_or_rss(url: str, max_chars: int = 4000) -> str:
    """Fetch a public JSON endpoint or RSS/Atom XML feed (SSRF-safe) and return structured items or JSON text."""
    url = (url or "").strip()
    if not url:
        return json.dumps({"error": "Empty URL"})
    max_chars = max(300, min(int(max_chars), 12000))
    try:
        raw, ctype, final_url = _open_public(
            url,
            accept="application/json, application/rss+xml, application/atom+xml, application/xml, text/xml, text/plain",
            timeout=12.0,
        )
        text = raw.decode("utf-8", errors="replace").strip()

        # Attempt JSON parse first
        if "json" in ctype.lower() or text.startswith(("{", "[")):
            try:
                parsed = json.loads(text)
                compact = json.dumps(parsed, ensure_ascii=False)
                return json.dumps(
                    {
                        "url": final_url,
                        "format": "json",
                        "content": compact[:max_chars],
                        "truncated": len(compact) > max_chars,
                    },
                    ensure_ascii=False,
                )
            except Exception:  # noqa: BLE001
                pass

        # Attempt RSS / Atom XML parse
        if text.startswith("<"):
            try:
                root = ET.fromstring(text)
                items: List[Dict[str, str]] = []
                # RSS 2.0 <item>
                for item in root.findall(".//item")[:10]:
                    items.append(
                        {
                            "title": (item.findtext("title") or "").strip(),
                            "link": (item.findtext("link") or "").strip(),
                            "pubDate": (item.findtext("pubDate") or "").strip(),
                            "description": re.sub(
                                r"<[^>]+>", "", (item.findtext("description") or "")
                            ).strip()[:300],
                        }
                    )
                # Atom <entry>
                if not items:
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall(".//atom:entry", ns)[:10]:
                        link_el = entry.find("atom:link", ns)
                        href = link_el.attrib.get("href", "") if link_el is not None else ""
                        items.append(
                            {
                                "title": (entry.findtext("atom:title", "", ns) or "").strip(),
                                "link": href,
                                "updated": (entry.findtext("atom:updated", "", ns) or "").strip(),
                                "summary": re.sub(
                                    r"<[^>]+>",
                                    "",
                                    (entry.findtext("atom:summary", "", ns) or ""),
                                ).strip()[:300],
                            }
                        )
                if items:
                    return json.dumps(
                        {"url": final_url, "format": "feed", "items": items},
                        ensure_ascii=False,
                    )
            except Exception:  # noqa: BLE001
                pass

        return json.dumps(
            {
                "url": final_url,
                "format": "raw",
                "content": text[:max_chars],
                "truncated": len(text) > max_chars,
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"})


_SAFE_MATH_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
    "min": min,
    "max": max,
}

_SAFE_BIN_OPS: Dict[ Any, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARY_OPS: Dict[Any, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_math_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_MATH_NAMES:
            return _SAFE_MATH_NAMES[node.id]
        raise ValueError(f"Unknown identifier: {node.id}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _SAFE_BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        uop_type = type(node.op)
        if uop_type not in _SAFE_UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {uop_type.__name__}")
        return _SAFE_UNARY_OPS[uop_type](_eval_math_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_MATH_NAMES:
            raise ValueError("Only safe math functions are permitted")
        fn = _SAFE_MATH_NAMES[node.func.id]
        args = [_eval_math_node(arg) for arg in node.args]
        return fn(*args)
    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


@_tool_decorator
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely (supports +, -, *, /, **, %, sin, cos, sqrt, log, log10, pi, e)."""
    expr = (expression or "").strip()
    if not expr:
        return json.dumps({"error": "Empty expression"})
    # Normalize common math notations like ^ to ** and "25%" to "(25/100)"
    normalized = expr.replace("^", "**")
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100.0)", normalized)
    try:
        tree = ast.parse(normalized, mode="eval")
        result = _eval_math_node(tree)
        return json.dumps({"expression": expr, "result": result}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"expression": expr, "error": f"{type(exc).__name__}: {exc}"})


INTERNET_TOOLS: List[Any] = [
    web_search,
    fetch_webpage,
    fetch_webpage_content,
    deep_web_research,
    wikipedia_search,
    arxiv_search,
    hackernews_search,
    github_repo_search,
    weather_lookup,
    fetch_json_or_rss,
    calculate,
]

TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "fetch_webpage_content": fetch_webpage_content,
    "deep_web_research": deep_web_research,
    "wikipedia_search": wikipedia_search,
    "arxiv_search": arxiv_search,
    "hackernews_search": hackernews_search,
    "github_repo_search": github_repo_search,
    "weather_lookup": weather_lookup,
    "fetch_json_or_rss": fetch_json_or_rss,
    "calculate": calculate,
}

OPENAI_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live internet (SearXNG or DuckDuckGo) and return titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch a public webpage (SSRF-safe) and extract its clean readable article text and optional outbound links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public http/https URL to read."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of readable text to return.",
                        "default": 4500,
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Whether to include extracted outbound links.",
                        "default": True,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage_content",
            "description": "Fetch a public URL (SSRF-safe) and extract clean article text and tables using trafilatura (with fallback to fetch_webpage).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public http/https URL to extract."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of readable text to return.",
                        "default": 4500,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_web_research",
            "description": "Search the web AND automatically fetch and extract the top N pages in one call. Prefer this for in-depth web research to save round-trips.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query."},
                    "top_pages": {
                        "type": "integer",
                        "description": "Number of top search result pages to fetch and extract (1-5).",
                        "default": 3,
                    },
                    "chars_per_page": {
                        "type": "integer",
                        "description": "Max characters of text per fetched page.",
                        "default": 2500,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia and return authoritative article summaries and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or entity to look up on Wikipedia."},
                    "sentences": {
                        "type": "integer",
                        "description": "Number of intro sentences to extract (1-12).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": "Search ArXiv for scientific papers, authors, publication dates, abstracts, and links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Scientific topic, paper title, or author query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hackernews_search",
            "description": "Search Hacker News via Algolia API for tech news, discussions, points, and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Tech topic or story query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of stories to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_repo_search",
            "description": "Search GitHub repositories for open-source projects, star counts, languages, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Repository search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of repositories to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_lookup",
            "description": "Look up live current weather and 3-day forecast for any city or location via Open-Meteo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name (e.g. 'Tokyo', 'Rome', 'New York')."},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_json_or_rss",
            "description": "Fetch a public JSON API endpoint or RSS/Atom XML feed (SSRF-safe) and return structured items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public http/https URL of JSON API or RSS/Atom feed."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return.",
                        "default": 4000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Safely evaluate a mathematical expression (sin, cos, sqrt, log, pi, e, powers, percentages).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. 'sqrt(144) + 15% * 200'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


def execute_tool_call(
    name: str,
    arguments: Any,
    max_chars: int = 4000,
) -> str:
    """Execute a registered internet tool by name with JSON string or dict arguments."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}", "available_tools": list(TOOL_REGISTRY)})

    if isinstance(arguments, str):
        arguments = arguments.strip()
        if not arguments:
            kwargs: Dict[str, Any] = {}
        else:
            try:
                kwargs = json.loads(arguments)
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": f"Invalid tool JSON arguments for {name}: {exc}"})
    elif isinstance(arguments, dict):
        kwargs = dict(arguments)
    else:
        kwargs = {}

    try:
        out = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        out = json.dumps({"error": f"Tool {name} raised {type(exc).__name__}: {exc}"})

    if max_chars > 0 and len(out) > max_chars:
        return out[:max_chars] + "...[truncated]"
    return out
