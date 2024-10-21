from collections.abc import Generator

from playwright.sync_api import Browser, Playwright
import pytest

from not_again_ai.data.web import create_browser, get_raw_web_content


@pytest.fixture
def browser_fixture() -> Generator[tuple[Playwright, Browser], None, None]:
    p, browser = create_browser(headless=True)
    yield p, browser
    browser.close()
    p.stop()


def test_get_raw_web_content_new_browser() -> None:
    content = get_raw_web_content("https://playwright.dev/")
    assert content, "Content should not be empty"


def test_get_raw_web_content_existing_browser(browser_fixture: tuple[Playwright, Browser]) -> None:
    _, browser = browser_fixture
    content = get_raw_web_content(
        "https://playwright.dev/",
        browser=browser,
    )
    assert content, "Content should not be empty"
