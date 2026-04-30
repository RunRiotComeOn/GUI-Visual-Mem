"""Browser-env variant with less automation-looking Playwright defaults."""

from __future__ import annotations

import json
from pathlib import Path

from browser_env.envs import ScriptBrowserEnv


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
if (!window.chrome) {
  Object.defineProperty(window, 'chrome', { get: () => ({ runtime: {} }) });
}
"""


class StealthScriptBrowserEnv(ScriptBrowserEnv):
    def __init__(
        self,
        *args,
        user_agent: str = DEFAULT_USER_AGENT,
        locale: str = "en-US",
        timezone_id: str = "America/New_York",
        accept_language: str = "en-US,en;q=0.9",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.user_agent = user_agent
        self.locale = locale
        self.timezone_id = timezone_id
        self.accept_language = accept_language

    def setup(self, config_file: Path | None = None) -> None:
        self.context_manager = self._sync_playwright()
        self.playwright = self.context_manager.__enter__()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--no-sandbox",
            ],
        )

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        self.context = self.browser.new_context(
            viewport=self.viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
            user_agent=self.user_agent,
            locale=self.locale,
            timezone_id=self.timezone_id,
            ignore_https_errors=True,
            extra_http_headers={"Accept-Language": self.accept_language},
        )
        self.context.add_init_script(STEALTH_INIT_SCRIPT)
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = self.context.new_page()
                client = page.context.new_cdp_session(page)
                if self.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                page.client = client  # type: ignore[attr-defined]
                page.goto(url)
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            self.page = self.context.new_page()
            client = self.page.context.new_cdp_session(self.page)
            if self.text_observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            self.page.client = client  # type: ignore[attr-defined]

    @staticmethod
    def _sync_playwright():
        from playwright.sync_api import sync_playwright

        return sync_playwright()
