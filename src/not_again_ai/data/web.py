import asyncio
import io
import mimetypes
from pathlib import Path
import re
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import httpx
from markitdown import MarkItDown, StreamInfo
from pydantic import BaseModel


class Link(BaseModel):
    url: str
    text: str


class URLResult(BaseModel):
    url: str
    markdown: str
    links: list[Link] = []


async def _markitdown_bytes_to_str(file_bytes: bytes, filename_extension: str) -> str:
    """
    Convert a file using MarkItDown defaults.
    """
    with io.BytesIO(file_bytes) as temp:
        result = await asyncio.to_thread(
            MarkItDown(enable_plugins=False).convert,
            source=temp,
            stream_info=StreamInfo(extension=filename_extension),
        )
        text = result.text_content
    return text


def _detect_pdf_extension(url: str) -> bool:
    """
    Detect if the URL is a PDF based on its extension.
    """
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name
    return mimetypes.guess_type(filename)[0] == "application/pdf"


def _detect_google_sheets(url: str) -> bool:
    """
    Detect if the URL is a Google Sheets document.
    """
    is_google_sheets = url.startswith("https://docs.google.com/spreadsheets/")
    return is_google_sheets


async def _handle_pdf_content(url: str) -> URLResult:
    md = MarkItDown(enable_plugins=False)
    result = md.convert(url)
    url_result = URLResult(
        url=url,
        markdown=result.markdown or "",
        links=[],
    )
    return url_result


async def _handle_google_sheets_content(url: str) -> URLResult:
    """
    Handle Google Sheets by using the export URL to get the raw content.
    """
    edit_pattern = r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)/edit"
    export_pattern = r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)/export\?format=csv"

    # Check if it's already an export URL
    export_match = re.search(export_pattern, url)
    if export_match:
        export_url = url
    else:
        # Check if it's an edit URL and extract document ID
        edit_match = re.search(edit_pattern, url)
        if edit_match:
            doc_id = edit_match.group(1)
            export_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid=0"
        else:
            return await _handle_web_content(url)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(export_url)
        response.raise_for_status()
        csv_bytes = response.content

    # Convert CSV to markdown using MarkItDown
    markdown_content = await _markitdown_bytes_to_str(csv_bytes, ".csv")

    url_result = URLResult(
        url=url,
        markdown=markdown_content,
        links=[],
    )
    return url_result


async def _handle_web_content(url: str, verbose: bool = False) -> URLResult:
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=verbose,
        user_agent_mode="random",
        java_script_enabled=True,
    )
    run_config = CrawlerRunConfig(
        scan_full_page=True,
        user_agent_mode="random",
        cache_mode=CacheMode.DISABLED,
        markdown_generator=DefaultMarkdownGenerator(),
        verbose=verbose,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config,
        )

        if result.response_headers.get("content-type") == "application/pdf":
            return await _handle_pdf_content(url)

    links: list[Link] = []
    seen_urls: set[str] = set()
    combined_link_data = result.links.get("internal", []) + result.links.get("external", [])
    for link_data in combined_link_data:
        href = link_data.get("href", "")
        if href and href not in seen_urls:
            seen_urls.add(href)
            link = Link(
                url=href,
                text=link_data.get("title", "") or link_data.get("text", ""),
            )
            links.append(link)

    url_result = URLResult(
        url=url,
        markdown=result.markdown or "",
        links=links,
    )
    return url_result


async def process_url(url: str, verbose: bool = False) -> URLResult:
    """
    Process a URL to extract content and convert it to Markdown and links
    """
    if _detect_pdf_extension(url):
        url_result = await _handle_pdf_content(url)
    elif _detect_google_sheets(url):
        url_result = await _handle_google_sheets_content(url)
    else:
        url_result = await _handle_web_content(url, verbose=verbose)
    return url_result
