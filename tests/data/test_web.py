import pytest

from not_again_ai.data.web import process_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://github.com/unclecode/crawl4ai",
        "https://arxiv.org/pdf/1710.02298",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.nascar.com/news/nascar-craftsman-truck-series/",
        "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?gid=0#gid=0",
        "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv&gid=0",
    ],
)
async def test_process_url(url: str) -> None:
    content = await process_url(url)
    assert content, f"Content should not be empty for URL: {url}"
