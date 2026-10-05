"""Scraper cache / 404 / 429 behaviour, with the network replaced by a fake."""
import pytest

from etl import scraper


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code, self.text, self.encoding = status_code, text, None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def fake_net(tmp_path, monkeypatch):
    """Redirect the cache to tmp_path and record every 'request'."""
    calls = []
    responses = {}

    def fake_get(url, **kwargs):
        calls.append(url)
        return responses.get(url.rsplit("/", 1)[-1], FakeResponse(404))

    monkeypatch.setattr(scraper, "RAW_DIR", tmp_path)
    monkeypatch.setattr(scraper.requests, "get", fake_get)
    monkeypatch.setattr(scraper.time, "sleep", lambda s: None)
    return calls, responses


def test_page_is_cached_after_first_fetch(fake_net, tmp_path):
    calls, responses = fake_net
    responses["NBA_2025_standings.html"] = FakeResponse(200, "<html>ok</html>")
    assert scraper.fetch_page(2025, "standings") == "<html>ok</html>"
    assert scraper.fetch_page(2025, "standings") == "<html>ok</html>"
    assert len(calls) == 1
    assert (tmp_path / "2025" / "standings.html").exists()


def test_refresh_bypasses_cache(fake_net):
    calls, responses = fake_net
    responses["NBA_2025_standings.html"] = FakeResponse(200, "v1")
    scraper.fetch_page(2025, "standings")
    responses["NBA_2025_standings.html"] = FakeResponse(200, "v2")
    assert scraper.fetch_page(2025, "standings", refresh=True) == "v2"
    assert scraper.fetch_page(2025, "standings") == "v2"
    assert len(calls) == 2


def test_404_is_remembered(fake_net, tmp_path):
    calls, _ = fake_net
    assert scraper.fetch_page(1999, "games-october") is None
    assert scraper.fetch_page(1999, "games-october") is None
    assert len(calls) == 1
    assert (tmp_path / "1999" / "games-october.404").exists()


def test_429_raises_and_caches_nothing(fake_net, tmp_path):
    _, responses = fake_net
    responses["NBA_2025_per_game.html"] = FakeResponse(429)
    with pytest.raises(scraper.RateLimited):
        scraper.fetch_page(2025, "per_game")
    assert not (tmp_path / "2025" / "per_game.html").exists()


def test_missing_required_page_is_a_clear_error(fake_net):
    with pytest.raises(scraper.parsers.TableNotFound):
        scraper.get_standings(1950)
