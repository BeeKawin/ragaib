from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup, Tag
from loguru import logger

from config.settings import (
    CURRICULUM,
    RAW_DIR,
    SCRAPE_DELAY_SECONDS,
    SCRAPE_MAX_RETRIES,
    SCRAPE_TIMEOUT,
    SCRAPE_USER_AGENT,
)


HEADERS = {
    "User-Agent": SCRAPE_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class ScrapedPage:
    def __init__(
        self,
        subject: str,
        grade: str,
        topic: str,
        url: str,
        title: str,
        sections: list[dict],
        key_terms: list[str],
        equations: list[str],
        examples: list[str],
        raw_text: str,
    ):
        self.subject = subject
        self.grade = grade
        self.topic = topic
        self.url = url
        self.title = title
        self.sections = sections
        self.key_terms = key_terms
        self.equations = equations
        self.examples = examples
        self.raw_text = raw_text
        self.source = "OpenStax"
        self.scraped_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict:
        return self.__dict__


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _entry_url(entry: dict) -> str:
    url = entry.get("source_url") or entry.get("openstax_url")
    if not url:
        raise ValueError(f"No source URL configured for {entry.get('topic')}")
    return str(url)


def _is_current_cache(path: Path, source_url: str) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return False
    return raw.get("source") == "OpenStax" and raw.get("url") == source_url


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _content_root(soup: BeautifulSoup) -> Tag:
    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", attrs={"data-type": "page"})
        or soup.body
    )
    if root is None:
        raise ValueError("OpenStax page has no parseable content root")
    return root


def _extract_sections(soup: BeautifulSoup) -> list[dict]:
    root = _content_root(soup)
    sections: list[dict] = []
    current_heading = "Introduction"
    current_blocks: list[str] = []

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue

        if el.name in {"h1", "h2", "h3", "h4"}:
            content = _clean_text("\n".join(current_blocks))
            if content:
                sections.append({"heading": current_heading, "content": content})
            current_heading = _clean_text(el.get_text(" ", strip=True))
            current_blocks = []
            continue

        if el.name == "p":
            text = _clean_text(el.get_text(" ", strip=True))
            if text:
                current_blocks.append(text)
        elif el.name in {"ul", "ol"}:
            for li in el.find_all("li", recursive=False):
                text = _clean_text(li.get_text(" ", strip=True))
                if text:
                    current_blocks.append(f"- {text}")

    content = _clean_text("\n".join(current_blocks))
    if content:
        sections.append({"heading": current_heading, "content": content})

    return sections


def _extract_key_terms(soup: BeautifulSoup) -> list[str]:
    terms: list[str] = []
    for heading in soup.find_all(["h2", "h3", "h4"]):
        label = heading.get_text(" ", strip=True).lower()
        if "key terms" not in label and "glossary" not in label:
            continue
        section = heading.find_parent(["section", "div", "article"]) or heading.parent
        if not section:
            continue
        for el in section.find_all(["li", "dt", "p"]):
            text = _clean_text(el.get_text(" ", strip=True))
            if text and len(text) <= 300:
                terms.append(text)
    return list(dict.fromkeys(terms))


def _extract_examples(soup: BeautifulSoup) -> list[str]:
    examples: list[str] = []
    candidates = soup.find_all(
        attrs={
            "data-type": re.compile(r"example|exercise|problem", re.I),
        }
    )
    candidates += [
        h.find_parent(["section", "div", "article"])
        for h in soup.find_all(["h2", "h3", "h4"], string=re.compile(r"example", re.I))
    ]
    for candidate in candidates:
        if not candidate:
            continue
        text = _clean_text(candidate.get_text("\n", strip=True))
        if len(text) >= 40:
            examples.append(text)
    return list(dict.fromkeys(examples))


def _extract_equations(soup: BeautifulSoup) -> list[str]:
    equations: list[str] = []
    for annotation in soup.find_all("annotation", encoding="application/x-tex"):
        text = _clean_text(annotation.get_text(" ", strip=True))
        if text:
            equations.append(text)
    for math in soup.find_all("math"):
        text = _clean_text(math.get_text(" ", strip=True))
        if text:
            equations.append(text)
    return list(dict.fromkeys(equations))


def parse_html(html: str, subject: str, grade: str, topic: str, url: str) -> ScrapedPage:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["nav", "footer", "script", "style", "noscript", "aside", "header", "svg"]):
        tag.decompose()

    title_tag = soup.find("h1") or soup.find("title")
    title = _clean_text(title_tag.get_text(" ", strip=True)) if title_tag else topic
    root = _content_root(soup)
    raw_text = _clean_text(root.get_text("\n", strip=True))

    return ScrapedPage(
        subject=subject,
        grade=grade,
        topic=topic,
        url=url,
        title=title,
        sections=_extract_sections(soup),
        key_terms=_extract_key_terms(soup),
        equations=_extract_equations(soup),
        examples=_extract_examples(soup),
        raw_text=raw_text,
    )


def save_page(page: ScrapedPage) -> Path:
    out_dir = RAW_DIR / page.subject / page.grade
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slugify(page.topic)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(page.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"Saved -> {out_path}")
    return out_path


async def _fetch(client: httpx.AsyncClient, url: str) -> Optional[str]:
    for attempt in range(1, SCRAPE_MAX_RETRIES + 1):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.warning(f"  Attempt {attempt} failed [{url}]: {exc}")
            if attempt < SCRAPE_MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
    return None


async def _scrape_entry(
    client: httpx.AsyncClient,
    entry: dict,
    semaphore: asyncio.Semaphore,
    force: bool = False,
) -> Optional[ScrapedPage]:
    subject, grade, topic = entry["subject"], entry["grade"], entry["topic"]
    url = _entry_url(entry)
    out_path = RAW_DIR / subject / grade / f"{_slugify(topic)}.json"

    if _is_current_cache(out_path, url) and not force:
        logger.debug(f"Skip (cached): {topic}")
        return None

    async with semaphore:
        logger.info(f"Scraping OpenStax [{subject}|{grade}] {topic}")
        html = await _fetch(client, url)
        if not html or len(html) < 500:
            logger.error(f"Empty/blocked response: {topic}")
            return None

        try:
            page = parse_html(html, subject, grade, topic, url)
            save_page(page)
            await asyncio.sleep(SCRAPE_DELAY_SECONDS)
            return page
        except Exception as exc:
            logger.error(f"Parse error [{topic}]: {exc}")
            return None


async def run_scraper(
    subjects: Optional[list] = None,
    grades: Optional[list] = None,
    concurrency: int = 3,
    force: bool = False,
) -> list[ScrapedPage]:
    entries = CURRICULUM
    if subjects:
        entries = [e for e in entries if e["subject"] in subjects]
    if grades:
        entries = [e for e in entries if e["grade"] in grades]

    to_scrape = []
    skipped = 0
    for entry in entries:
        path = RAW_DIR / entry["subject"] / entry["grade"] / f"{_slugify(entry['topic'])}.json"
        if _is_current_cache(path, _entry_url(entry)) and not force:
            skipped += 1
        else:
            to_scrape.append(entry)

    logger.info(
        f"Entries: {len(entries)} total | {skipped} cached | {len(to_scrape)} to scrape"
    )
    if not to_scrape:
        logger.success("All pages already cached.")
        return []

    results: list[ScrapedPage] = []
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(SCRAPE_TIMEOUT)

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
        tasks = [_scrape_entry(client, entry, semaphore, force) for entry in to_scrape]
        for coro in asyncio.as_completed(tasks):
            page = await coro
            if page:
                results.append(page)

    logger.success(
        f"Scraping complete. {len(results)}/{len(to_scrape)} pages fetched successfully."
    )
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="OpenStax scraper - Thailand HS")
    p.add_argument("--subjects", nargs="*", help="math chemistry physics biology")
    p.add_argument("--grades", nargs="*", help="M4 M5 M6")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    asyncio.run(
        run_scraper(
            subjects=args.subjects,
            grades=args.grades,
            concurrency=args.concurrency,
            force=args.force,
        )
    )
    