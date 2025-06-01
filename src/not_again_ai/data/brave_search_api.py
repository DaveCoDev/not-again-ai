import os

import httpx
from loguru import logger
from pydantic import BaseModel


class SearchWebResult(BaseModel):
    title: str
    url: str
    description: str
    netloc: str | None = None


class SearchWebResults(BaseModel):
    results: list[SearchWebResult]


async def search(
    query: str,
    count: int = 20,
    offset: int = 0,
    country: str = "US",
    search_lang: str = "en",
    ui_lang: str = "en-US",
    freshness: str | None = None,
    timezone: str = "America/New_York",
    state: str = "MA",
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.",
) -> SearchWebResults:
    """
    Search using Brave Search API.

    Args:
        query: The search query string
        count: Number of search results to return (1-20, default 10)
        offset: Number of search results to skip (default 0)
        country: Country code for search results (default "US")
        search_lang: Language for search (default "en")
        ui_lang: User interface language (default "en-US")
        freshness: Freshness of results ("pd", "pw", "pm", "py" or YYYY-MM-DDtoYYYY-MM-DD or None)
        timezone: Timezone for search results (default "America/New_York")
        state: State for search results (default "MA")
        user_agent: User agent string for the request (default is a common browser UA)

    Returns:
        SearchWebResults: A model containing the search results

    Raises:
        httpx.HTTPError: If the request fails
        ValueError: If BRAVE_SEARCH_API_KEY is not set
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY environment variable is not set")

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
        "X-Loc-Country": country,
        "X-Loc-Timezone": timezone,
        "X-Loc-State": state,
        "User-Agent": user_agent,
    }

    params: dict[str, str | int | bool] = {
        "q": query,
        "count": count,
        "offset": offset,
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
        "text_decorations": False,
        "spellcheck": False,
        "units": "imperial",
        "extra_snippets": False,
        "safesearch": "off",
    }

    # Add optional parameters if provided
    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            results_list: list[SearchWebResult] = []
            for item in data.get("web", {}).get("results", []):
                result = SearchWebResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("snippet", ""),
                    netloc=item.get("meta_url", {}).get("netloc", None),
                )
                results_list.append(result)
            return SearchWebResults(results=results_list)

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during Brave search: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Brave search: {e}")
        raise


class SearchNewsResult(BaseModel):
    title: str
    url: str
    description: str
    age: str
    netloc: str | None = None


class SearchNewsResults(BaseModel):
    results: list[SearchNewsResult]


async def search_news(
    query: str,
    count: int = 20,
    offset: int = 0,
    country: str = "US",
    search_lang: str = "en",
    ui_lang: str = "en-US",
    freshness: str | None = None,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.",
) -> SearchNewsResults:
    """
    Search news using Brave News Search API.

    Args:
        query: The search query string
        count: Number of news results to return (1-20, default 20)
        offset: Number of search results to skip (default 0)
        country: Country code for search results (default "US")
        search_lang: Language for search (default "en")
        ui_lang: User interface language (default "en-US")
        freshness: Freshness of results ("pd", "pw", "pm", "py" or YYYY-MM-DDtoYYYY-MM-DD or None)
        user_agent: User agent string for the request (default is a common browser UA)

    Returns:
        SearchNewsResults: A model containing the news search results

    Raises:
        httpx.HTTPError: If the request fails
        ValueError: If BRAVE_SEARCH_API_KEY is not set
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY environment variable is not set")

    url = "https://api.search.brave.com/res/v1/news/search"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
        "User-Agent": user_agent,
    }

    params: dict[str, str | int | bool] = {
        "q": query,
        "count": count,
        "offset": offset,
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
        "spellcheck": False,
        "safesearch": "off",
    }

    # Add optional parameters if provided
    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            results_list: list[SearchNewsResult] = []
            for item in data.get("results", []):
                result = SearchNewsResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                    age=item.get("age"),
                    netloc=item.get("meta_url", {}).get("netloc", None),
                )
                results_list.append(result)
            return SearchNewsResults(results=results_list)

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during Brave news search: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Brave news search: {e}")
        raise
