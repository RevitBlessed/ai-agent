import io
import json
import os
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


async def _ensure_font_blocking(page) -> None:
    if not _env_bool("ANNOTATE_BLOCK_FONTS", True):
        return
    context = page.context
    if getattr(context, "_fonts_blocked", False):
        return

    async def _route_handler(route, request) -> None:
        if request.resource_type == "font":
            await route.abort()
        else:
            await route.continue_()

    try:
        await context.route("**/*", _route_handler)
        setattr(context, "_fonts_blocked", True)
    except Exception:
        # If routing fails (context closed), just skip blocking.
        pass


async def _take_screenshot_bytes(page) -> bytes | None:
    timeout_ms = _env_int("ANNOTATE_SCREENSHOT_TIMEOUT_MS", 4000)
    retry_timeout_ms = _env_int("ANNOTATE_SCREENSHOT_RETRY_TIMEOUT_MS", 2000)
    retry_wait_ms = _env_int("ANNOTATE_SCREENSHOT_RETRY_WAIT_MS", 2000)
    screenshot_kwargs = {
        "timeout": timeout_ms,
        "animations": "disabled",
        "scale": "css",
    }
    try:
        return await page.screenshot(**screenshot_kwargs)
    except Exception as exc:
        message = str(exc)
        if "Timeout" not in message and "Page.screenshot" not in message:
            raise
        if retry_wait_ms:
            try:
                await page.wait_for_timeout(retry_wait_ms)
            except Exception:
                pass
        try:
            screenshot_kwargs["timeout"] = retry_timeout_ms
            return await page.screenshot(**screenshot_kwargs)
        except Exception as retry_exc:
            message = str(retry_exc)
            if "Timeout" not in message and "Page.screenshot" not in message:
                raise
            return None

async def annotate_page(
    page,
    output_image: str = "avito_marked.png",
    output_data: str = "metadata.json",
    # Расширенный селектор: ищем всё, что может быть кликабельным или текстовым полем
    selector: str = "button, a, input, textarea, select, [contenteditable='true'], [role='button'], [role='link'], [role='checkbox'], [role='menuitem'], [role='switch'], [role='textbox'], [role='combobox'], [role='searchbox'], [onclick], [tabindex], label, summary, [data-marker], [data-qa]",
    settle_ms: int | None = None,
    max_markers: int | None = None,
) -> List[Dict]:
    if settle_ms is None:
        settle_ms = _env_int("ANNOTATE_SETTLE_MS", 300)
    if settle_ms < 0:
        settle_ms = 0
    if max_markers is None:
        max_markers = _env_int("MAX_MARKERS", 0)
        if max_markers <= 0:
            max_markers = None
    load_timeout_ms = _env_int("ANNOTATE_LOAD_TIMEOUT_MS", 10000)
    retry_wait_ms = _env_int("ANNOTATE_RETRY_WAIT_MS", 300)

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=load_timeout_ms)
    except Exception:
        pass
    await _ensure_font_blocking(page)
    if settle_ms:
        await page.wait_for_timeout(settle_ms)

    # Выбираем все потенциально интересные элементы
    elements: List[Dict] = []
    eval_params = {"selector": selector, "maxElements": max_markers or 0}
    eval_script = """({selector, maxElements}) => {
        const nodes = Array.from(document.querySelectorAll(selector));
        const results = [];
        const vw = window.innerWidth || 0;
        const vh = window.innerHeight || 0;
        const limit = maxElements && maxElements > 0 ? maxElements : 0;

        for (const el of nodes) {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            const tag = el.tagName ? el.tagName.toLowerCase() : "";
            const role = el.getAttribute("role") || "";
            const isInputRole = role === "searchbox" || role === "textbox" || role === "combobox";
            const hasLabel = !!el.closest("label");
            const inInputWrapper = !!el.closest("div.input, div[class*='input']");
            const isInput = tag === "input" || tag === "textarea" || el.getAttribute("contenteditable") === "true" || isInputRole || hasLabel || inInputWrapper;
            const isClickable = style.cursor === "pointer" || tag === "button" || tag === "a" || tag === "input" || tag === "textarea";
            const visible = rect.width > 2 && rect.height > 2 &&
                style.display !== "none" &&
                style.visibility !== "hidden" &&
                (isInput || style.opacity !== "0");
            if (!visible) continue;
            if (tag && !["input", "textarea", "button", "a", "select"].includes(tag) && !isClickable && !role && !isInput) {
                continue;
            }
            if (rect.right < 0 || rect.bottom < 0 || rect.left > vw || rect.top > vh) {
                continue;
            }
            const placeholder = el.getAttribute("placeholder") || "";
            const ariaLabel = el.getAttribute("aria-label") || "";
            const title = el.getAttribute("title") || "";
            const text = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
            let description = placeholder || ariaLabel || text || title || "Элемент";
            if (description.length > 100) {
                description = description.slice(0, 100);
            }
            results.push({
                x: rect.left,
                y: rect.top,
                w: rect.width,
                h: rect.height,
                tag,
                isInput,
                description
            });
            if (limit && results.length >= limit) {
                break;
            }
        }
        return results;
    }"""
    for _ in range(3):
        try:
            elements = await page.evaluate(eval_script, eval_params)
            break
        except Exception:
            # Navigation can destroy execution context; retry after short wait.
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=load_timeout_ms)
            except Exception:
                pass
            if retry_wait_ms:
                await page.wait_for_timeout(retry_wait_ms)
    
    metadata: List[Dict] = []
    screenshot_bytes = await _take_screenshot_bytes(page)
    if not screenshot_bytes:
        viewport = page.viewport_size or {"width": 1280, "height": 720}
        blank = Image.new("RGB", (viewport["width"], viewport["height"]), (255, 255, 255))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        screenshot_bytes = buffer.getvalue()

    with Image.open(io.BytesIO(screenshot_bytes)) as img:
        base = img.convert("RGBA")
        # Создаем отдельный слой для прозрачных меток
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        counter = 0
        label_boxes = []
        label_w, label_h = 28, 20
        img_w, img_h = base.size
        seen_boxes = set()

        for info in elements:
            # Скрипт проверяет: виден ли элемент, имеет ли он размер и какой у него курсор
            # Если курсор pointer — значит на элемент можно нажать, даже если это простой div
            x, y, w, h = info["x"], info["y"], info["w"], info["h"]

            # Защита от дублей и выхода за границы
            box_key = (round(x), round(y))
            if box_key in seen_boxes or x < 0 or y < 0 or x >= img_w or y >= img_h:
                continue
            seen_boxes.add(box_key)

            counter += 1

            # Центрируем метку для инпутов и чекбоксов, для остального ставим в угол
            is_small_widget = info.get("isInput") or info["w"] < 50
            if is_small_widget:
                label_x = x + (w / 2) - (label_w / 2)
                label_y = y + (h / 2) - (label_h / 2)
            else:
                label_x, label_y = x, y

            # Умное размещение, чтобы метки не перекрывали друг друга
            for _ in range(10):
                candidate = (label_x, label_y, label_x + label_w, label_y + label_h)
                overlaps = any(
                    not (
                        candidate[2] <= b[0]
                        or candidate[0] >= b[2]
                        or candidate[3] <= b[1]
                        or candidate[1] >= b[3]
                    )
                    for b in label_boxes
                )
                if not overlaps:
                    break
                label_y += label_h + 2
            
            label_boxes.append((label_x, label_y, label_x + label_w, label_y + label_h))

            # Рисуем красную полупрозрачную метку для лучшей заметности
            draw.rectangle(
                [label_x, label_y, label_x + label_w, label_y + label_h],
                fill=(255, 0, 0, 90),
                outline=(255, 255, 255, 140),
                width=1
            )
            draw.text((label_x + 5, label_y + 2), str(counter), fill=(255, 255, 255, 255), font=font)

            # Собираем данные
            description = (info.get("description") or "Элемент").strip()[:100]
            metadata.append({
                "id": counter,
                "tag": info.get("tag", ""),
                "description": description,
                "center_point": {"x": round(x + w / 2), "y": round(y + h / 2)},
            })

        # Совмещаем оригинальное фото и слой с прозрачными метками
        combined = Image.alpha_composite(base, overlay)
        combined.convert("RGB").save(output_image)

    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata