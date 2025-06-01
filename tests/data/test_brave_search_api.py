from typing import Any

import pytest

from not_again_ai.data.brave_search_api import search, search_news


@pytest.mark.parametrize(
    ("query", "search_params"),
    [
        ("brave search", {}),
        ("python programming", {"count": 2, "country": "US"}),
        ("machine learning", {"count": 4, "search_lang": "en", "freshness": "pw"}),
        ("AI news", {"count": 1, "offset": 5, "country": "GB", "ui_lang": "en-GB"}),
    ],
)
async def test_brave_search_api(query: str, search_params: dict[str, Any]) -> None:
    """Test the Brave Search API with a sample query and optional parameters."""
    content = await search(query=query, **search_params)
    assert content.results, f"No results returned for query: {query}"


@pytest.mark.skip("API Cost")
@pytest.mark.parametrize(
    ("query", "search_params"),
    [
        ("latest tech news", {}),
        ("AI breakthrough", {"count": 3, "country": "US"}),
    ],
)
async def test_brave_search_news_api(query: str, search_params: dict[str, Any]) -> None:
    """Test the Brave News Search API with a sample query and optional parameters."""
    content = await search_news(query=query, **search_params)
    assert content.results, f"No news results returned for query: {query}"
