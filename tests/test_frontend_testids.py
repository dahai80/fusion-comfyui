from fusion_comfyui.server.static_files import get_frontend_dir


def _index_html() -> str:
    f = get_frontend_dir() / "index.html"
    return f.read_text(encoding="utf-8")


def test_bootstrap_script_present():
    html = _index_html()
    assert "__fusionComfyTestids" in html, "testid bootstrap helper missing from index.html"
    assert "comfy:ui-ready" in html, "comfy:ui-ready event not dispatched in index.html"


def test_required_testid_slugs_mapped():
    html = _index_html()
    required = [
        "queue-prompt",
        "queue-front",
        "view-queue",
        "view-history",
        "settings",
        "save",
        "load",
    ]
    for slug in required:
        assert slug in html, f"testid slug {slug!r} not present in index.html bootstrap"


def test_gear_emoji_mapped_to_settings():
    html = _index_html()
    assert "'⚙️': 'settings'" in html, "gear emoji -> settings mapping missing"


def test_mutation_observer_used():
    html = _index_html()
    assert "MutationObserver" in html, "toolbar stamping must use MutationObserver for async Vue mount"


def test_legacy_menu_root_targeted():
    html = _index_html()
    assert "comfy-menu" in html, "bootstrap must target legacy .comfy-menu (gear lives there)"


def test_html_served_is_valid():
    html = _index_html()
    assert html.lstrip().startswith("<!doctype html>")
    assert "</body></html>" in html[-200:]
