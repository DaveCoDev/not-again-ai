from loguru import logger
from playwright.sync_api import Browser, Playwright, sync_playwright


def create_browser(headless: bool = True) -> tuple[Playwright, Browser]:
    """Creates and returns a new Playwright instance and browser.

    Args:
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to True.

    Returns:
        tuple[Playwright, Browser]: A tuple containing the Playwright instance and browser.
    """
    pwright = sync_playwright().start()
    browser = pwright.chromium.launch(
        headless=headless,
        chromium_sandbox=False,
        timeout=15000,
    )
    return pwright, browser


def get_raw_web_content(url: str, browser: Browser | None = None, headless: bool = True) -> str:
    """Fetches raw web content from a given URL using Playwright.

    Args:
        url (str): The URL to fetch content from.
        browser (Browser | None, optional): An existing browser instance to use. Defaults to None.
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to True.

    Returns:
        str: The raw web content.
    """
    p = None
    try:
        if browser is None:
            p, browser = create_browser(headless)

        page = browser.new_page(
            accept_downloads=False,
            java_script_enabled=True,
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.3",
        )
        page.goto(url)
        content = page.content()
        page.close()
        return content
    except Exception as e:
        logger.error(f"Failed to get web content: {e}")
        return ""
    finally:
        if browser:
            browser.close()
        if p:
            p.stop()
