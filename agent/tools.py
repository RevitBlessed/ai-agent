import asyncio
import json
import os
from typing import Dict, Tuple


def _typing_delay_ms() -> int:
    raw = os.getenv("TYPE_DELAY_MS", "20")
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 20


def _load_metadata(metadata_path: str) -> Dict[int, Dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(item.get("id")): item for item in data}


def _get_center_point(metadata_path: str, marker_id: int) -> Tuple[int, int]:
    mapping = _load_metadata(metadata_path)
    if marker_id not in mapping:
        raise ValueError(f"Marker id {marker_id} not found in {metadata_path}")
    point = mapping[marker_id].get("center_point") or {}
    return int(point.get("x", 0)), int(point.get("y", 0))


async def tool_click(page, marker_id: int, metadata_path: str = "metadata.json") -> None:
    x, y = _get_center_point(metadata_path, marker_id)
    await page.mouse.click(x, y)


async def tool_type(
    page,
    marker_id: int,
    text: str,
    metadata_path: str = "metadata.json",
) -> None:
    x, y = _get_center_point(metadata_path, marker_id)
    await page.mouse.click(x, y)
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text, delay=_typing_delay_ms())


async def tool_wait(page, seconds: float) -> None:
    timeout_ms = max(200, int(seconds * 1000))
    try:
        state = await page.evaluate("document.readyState")
    except Exception:
        state = "loading"
    if state != "complete":
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass
    # Small buffer to allow UI to settle without long pauses.
    await asyncio.sleep(min(0.3, seconds))


async def tool_scroll(page, delta_y: int) -> None:
    await page.mouse.wheel(0, delta_y)


async def tool_open(page, url: str) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception:
        # Some pages keep network busy; still allow interaction.
        await page.wait_for_timeout(1000)


async def tool_read(page, marker_id: int, metadata_path: str = "metadata.json") -> str:
    x, y = _get_center_point(metadata_path, marker_id)
    text = await page.evaluate(
        """({x, y}) => {
            const el = document.elementFromPoint(x, y);
            if (!el) return "";
            const raw = el.innerText || el.textContent || "";
            return raw.replace(/\\s+/g, " ").trim();
        }""",
        {"x": x, "y": y},
    )
    return (text or "")[:1000]
