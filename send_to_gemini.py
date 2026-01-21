import argparse
import asyncio
import json
import os
import time
from typing import List, Dict

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from playwright.async_api import async_playwright

from agent.annotator import annotate_page
from agent.response_parser import extract_tool_payloads, extract_json_payload
from agent.tools import tool_click, tool_open, tool_read, tool_scroll, tool_type, tool_wait


def load_metadata(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table(metadata: List[Dict], max_rows: int | None = None) -> str:
    rows = metadata if max_rows is None else metadata[:max_rows]
    lines = ["| id | description |", "|---:|---|"]
    for item in rows:
        item_id = item.get("id", "")
        description = (item.get("description") or "").replace("\n", " ").strip()
        lines.append(f"| {item_id} | {description} |")
    return "\n".join(lines)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(path: str, history: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_goal(path: str, goal: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"goal": goal}, f, ensure_ascii=False, indent=2)


def format_history(history: List[Dict]) -> str:
    if not history:
        return "История подцелей пуста."
    lines = []
    for idx, item in enumerate(history, start=1):
        lines.append(f"{idx}) {item}")
    return "\n".join(lines)


def format_action_history(actions: List[str]) -> str:
    if not actions:
        return "Нет действий."
    return "\n".join(actions[-6:])


_LAST_MODEL_CALL_TS = 0.0


def _get_min_model_interval() -> float:
    raw = os.getenv("MIN_MODEL_INTERVAL", "0.4")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.4


def _get_post_action_wait_ms() -> int:
    raw = os.getenv("POST_ACTION_WAIT_MS", "200")
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 200


def _get_repeat_click_limit() -> int:
    raw = os.getenv("MAX_REPEAT_CLICKS", "10")
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 10


def _format_excerpt(text: str, limit: int = 300) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) > limit:
        return cleaned[:limit] + "..."
    return cleaned


def _add_read_context(observation: str, read_text: str) -> str:
    excerpt = _format_excerpt(read_text)
    if not excerpt:
        return observation
    return f"{observation} Прочитанный текст: {excerpt}"


def _select_read_candidate(metadata: List[Dict]) -> int | None:
    best_id = None
    best_score = -1
    for item in metadata:
        desc = (item.get("description") or "").strip()
        if len(desc) < 3:
            continue
        if desc.lower() in ("элемент", "element"):
            continue
        lowered = desc.lower()
        score = len(desc)
        if "@" in desc:
            score += 40
        if any(token in lowered for token in ("re:", "fw:", "fwd:", "от:", "тема", "subject", "from", "sender")):
            score += 25
        if len(desc.split()) > 3:
            score += 10
        if score > best_score:
            best_score = score
            best_id = item.get("id")
    if best_id is None:
        return None
    try:
        return int(best_id)
    except (TypeError, ValueError):
        return None


async def _describe_page(page, metadata_path: str) -> str:
    url = ""
    title = ""
    try:
        url = page.url
    except Exception:
        pass
    try:
        title = await page.title()
    except Exception:
        pass
    marker_count = 0
    try:
        metadata = load_metadata(metadata_path)
        marker_count = len(metadata)
    except Exception:
        pass
    parts = []
    if url:
        parts.append(f"url={url}")
    if title:
        parts.append(f"title={title}")
    parts.append(f"markers={marker_count}")
    return " ".join(parts) if parts else "url/title недоступны"


async def _describe_observation(
    page,
    metadata_path: str,
    message: str,
    last_read_text: str,
) -> str:
    page_state = await _describe_page(page, metadata_path)
    return _add_read_context(f"{message} {page_state}", last_read_text)


async def _annotate_with_retry(page, output_image: str, output_data: str) -> List[Dict]:
    try:
        return await annotate_page(page, output_image=output_image, output_data=output_data)
    except Exception as exc:
        if "Execution context was destroyed" not in str(exc):
            raise
    try:
        await page.wait_for_load_state("load", timeout=20000)
    except Exception:
        pass
    return await annotate_page(page, output_image=output_image, output_data=output_data)


def build_planner_prompt(
    planner_prompt: str,
    global_goal: str,
    completed_subgoals: List[str],
    observation: str,
) -> str:
    return (
        f"{planner_prompt}\n\n"
        f"Глобальная цель: {global_goal}\n"
        f"Краткая история подцелей:\n{format_history(completed_subgoals)}\n"
        f"Observation: {observation}\n"
    )


def build_executor_prompt(
    executor_prompt: str,
    table: str,
    subgoal: str,
    success_condition: str,
    failure_condition: str,
    action_history: List[str],
    credentials: str,
) -> str:
    return (
        f"{executor_prompt}\n\n"
        f"Текущая подцель: {subgoal}\n"
        f"Условие успеха: {success_condition}\n"
        f"Условие тупика: {failure_condition}\n"
        f"Данные для входа (env):\n{credentials}\n"
        f"История действий:\n{format_action_history(action_history)}\n\n"
        f"{table}\n"
    )


async def _call_model_with_retry(model, payload, retries: int = 3) -> str:
    global _LAST_MODEL_CALL_TS
    for attempt in range(retries + 1):
        try:
            now = time.monotonic()
            elapsed = now - _LAST_MODEL_CALL_TS
            min_interval = _get_min_model_interval()
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            _LAST_MODEL_CALL_TS = time.monotonic()
            response = model.generate_content(payload)
            return response.text or ""
        except Exception as exc:
            message = str(exc)
            if "ResourceExhausted" in message or "429" in message:
                if attempt < retries:
                    print(f"[model] 429, retry {attempt + 1}/{retries}...")
                    await asyncio.sleep(2 ** attempt)
                    continue
            raise


async def _execute_tool(page, payload: Dict, metadata_path: str) -> tuple[bool, str | None]:
    tool = payload.get("tool")
    args = payload.get("args") or {}
    if tool == "none":
        return True, None
    if tool == "open":
        url = args.get("url")
        if url:
            await tool_open(page, url)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
    elif tool == "click":
        marker_id = args.get("marker_id")
        if marker_id is not None:
            await tool_click(page, int(marker_id), metadata_path=metadata_path)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                await page.wait_for_timeout(2000)
    elif tool == "type":
        marker_id = args.get("marker_id")
        text = args.get("text")
        if marker_id is not None and text is not None:
            await tool_type(page, int(marker_id), str(text), metadata_path=metadata_path)
            try:
                wait_ms = _get_post_action_wait_ms()
                if wait_ms:
                    await page.wait_for_timeout(wait_ms)
            except Exception:
                pass
    elif tool == "read":
        marker_id = args.get("marker_id")
        if marker_id is not None:
            text = await tool_read(page, int(marker_id), metadata_path=metadata_path)
            return False, text
    elif tool == "wait":
        seconds = args.get("seconds")
        if seconds is not None:
            await tool_wait(page, float(seconds))
    elif tool == "scroll":
        delta_y = args.get("delta_y")
        if delta_y is not None:
            await tool_scroll(page, int(delta_y))
            try:
                wait_ms = _get_post_action_wait_ms()
                if wait_ms:
                    await page.wait_for_timeout(wait_ms)
            except Exception:
                pass
    return False, None


async def _query_executor(
    model,
    page,
    executor_prompt: str,
    subgoal: str,
    success_condition: str,
    failure_condition: str,
    action_history: List[str],
    credentials: str,
    image_path: str,
    metadata_path: str,
    max_rows: int | None,
) -> str:
    metadata = await _annotate_with_retry(page, output_image=image_path, output_data=metadata_path)
    table = build_table(metadata, max_rows)
    prompt = build_executor_prompt(
        executor_prompt,
        table,
        subgoal,
        success_condition,
        failure_condition,
        action_history,
        credentials,
    )
    image = Image.open(image_path)
    try:
        text = await _call_model_with_retry(model, [prompt, image])
        print("\n[executor] response:\n" + text)
        return text
    finally:
        image.close()


async def _query_planner(
    model,
    planner_prompt: str,
    global_goal: str,
    completed_subgoals: List[str],
    observation: str,
) -> Dict:
    prompt = build_planner_prompt(planner_prompt, global_goal, completed_subgoals, observation)
    text = await _call_model_with_retry(model, prompt)
    print("\n[planner] response:\n" + text)
    payload = extract_json_payload(text)
    return payload or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send screenshot + metadata to Gemini.")
    parser.add_argument("--image", default="avito_marked.png", help="Path to marked screenshot.")
    parser.add_argument("--metadata", default="metadata.json", help="Path to metadata JSON.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows in table.")
    parser.add_argument("--planner-prompt", default="planner_prompt.txt", help="Path to planner prompt.")
    parser.add_argument("--executor-prompt", default="executor_prompt.txt", help="Path to executor prompt.")
    parser.add_argument("--history", default="dialog_history.json", help="Path to history JSON.")
    parser.add_argument("--goal-file", default="global_goal.json", help="Path to global goal JSON.")
    return parser.parse_args()


def _require_env(name: str, message: str, fallback: str | None = None) -> str:
    value = os.getenv(name)
    if not value and fallback:
        value = os.getenv(fallback)
    if not value:
        raise SystemExit(message)
    return value


def _load_models() -> tuple[genai.GenerativeModel, genai.GenerativeModel]:
    planner_key = _require_env(
        "GEMINI_API_KEY",
        "Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.",
        fallback="GOOGLE_API_KEY",
    )
    executor_key = _require_env("GEMINI_API_KEY2", "Missing GEMINI_API_KEY2 in environment.")
    planner_model_name = _require_env("GEMINI_MODEL", "Missing GEMINI_MODEL in environment.")
    executor_model_name = _require_env("GEMINI_MODEL2", "Missing GEMINI_MODEL2 in environment.")

    genai.configure(api_key=planner_key)
    planner_model = genai.GenerativeModel(planner_model_name)
    genai.configure(api_key=executor_key)
    executor_model = genai.GenerativeModel(executor_model_name)
    return planner_model, executor_model


def _load_prompts(planner_prompt_path: str, executor_prompt_path: str) -> tuple[str, str]:
    planner_prompt = load_text(planner_prompt_path)
    executor_prompt = load_text(executor_prompt_path)
    return planner_prompt, executor_prompt


def _build_credentials() -> str:
    hh_login = os.getenv("HH_LOGIN", "")
    hh_password = os.getenv("HH_PASSWORD", "")
    outlook_login = os.getenv("OUTLOOK_LOGIN", "")
    outlook_password = os.getenv("OUTLOOK_PASSWORD", "")
    return (
        f"HH_LOGIN={hh_login}\nHH_PASSWORD={hh_password}\n"
        f"OUTLOOK_LOGIN={outlook_login}\nOUTLOOK_PASSWORD={outlook_password}"
    )


def _init_history_goal(history_path: str, goal_path: str) -> tuple[List[Dict], str]:
    history: List[Dict] = []
    save_history(history_path, history)
    global_goal = ""
    save_goal(goal_path, global_goal)
    return history, global_goal


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    planner_model, executor_model = _load_models()
    planner_prompt, executor_prompt = _load_prompts(
        args.planner_prompt,
        args.executor_prompt,
    )
    credentials = _build_credentials()
    history, global_goal = _init_history_goal(args.history, args.goal_file)

    async with async_playwright() as p:
        print("Запуск браузера...", flush=True)
        try:
            browser = await asyncio.wait_for(
                p.chromium.launch(headless=False, channel="msedge"), timeout=20
            )
            context = await browser.new_context()
            page = await asyncio.wait_for(context.new_page(), timeout=10)
        except asyncio.TimeoutError:
            raise SystemExit("Браузер не запустился за 20с.")
        print("Браузер готов. Введите вопрос.", flush=True)
        try:
            while True:
                user_question = input("Вопрос> ").strip()
                if not user_question:
                    continue
                global_goal = user_question
                save_goal(args.goal_file, global_goal)
                print(f"\n[user] {user_question}")

                completed_subgoals: List[str] = []
                observation = "Нет наблюдений."
                action_history: List[str] = []
                repeat_click_marker_id: int | None = None
                repeat_click_count = 0
                repeat_click_limit = _get_repeat_click_limit()

                current_subgoal = None
                success_condition = ""
                failure_condition = ""
                last_clicked_marker_id: int | None = None
                read_in_subgoal = False
                auto_read_attempted = False
                last_read_text = ""
                planner_cycles = 0
                max_planner_cycles = 10

                while True:
                    if current_subgoal is None:
                        if planner_cycles >= max_planner_cycles:
                            print("Достигнут лимит обращений к planner.")
                            break
                        planner_cycles += 1
                        plan = await _query_planner(
                            planner_model, planner_prompt, global_goal, completed_subgoals, observation
                        )
                        current_subgoal = plan.get("subgoal") or ""
                        success_condition = plan.get("success_condition") or ""
                        failure_condition = plan.get("failure_condition") or ""
                        last_clicked_marker_id = None
                        read_in_subgoal = False
                        auto_read_attempted = False
                        last_read_text = ""
                        if current_subgoal.strip().upper() == "DONE":
                            print("Planner сообщил завершение цели. Останавливаю цикл.")
                            return
                        if not current_subgoal:
                            print("Planner не вернул подцель. Останавливаю цикл.")
                            break

                    last_action = None
                    step_count = 0
                    max_steps = 8
                    while True:
                        if step_count >= max_steps:
                            observation = "Достигнут лимит шагов для подцели."
                            current_subgoal = None
                            break
                        answer = await _query_executor(
                            executor_model,
                            page,
                            executor_prompt,
                            current_subgoal,
                            success_condition,
                            failure_condition,
                            action_history,
                            credentials,
                            args.image,
                            args.metadata,
                            args.max_rows,
                        )
                        payloads = extract_tool_payloads(answer)
                        step_count += 1
                        if not payloads:
                            observation = "Executor не вернул tool."
                            current_subgoal = None
                            break
                        if len(payloads) > 1:
                            print("Получено несколько tool; выполняю только первый.")
                        payload = payloads[0]
                        try:
                            if payload.get("tool") == "click":
                                raw_marker_id = (payload.get("args") or {}).get("marker_id")
                                try:
                                    current_marker_id = int(raw_marker_id)
                                except (TypeError, ValueError):
                                    current_marker_id = None
                                if current_marker_id is not None:
                                    if current_marker_id == repeat_click_marker_id:
                                        repeat_click_count += 1
                                    else:
                                        repeat_click_marker_id = current_marker_id
                                        repeat_click_count = 1
                                    if repeat_click_count >= repeat_click_limit:
                                        action_history.append(
                                            f"guard_wait {{'marker_id': {current_marker_id}}}"
                                        )
                                        await page.wait_for_timeout(2000)
                                        repeat_click_count = 0
                                        repeat_click_marker_id = None
                                        observation = (
                                            f"Слишком много повторных кликов по маркеру {current_marker_id}. "
                                            "Делаю паузу и повторяю анализ."
                                        )
                                        continue
                            else:
                                repeat_click_marker_id = None
                                repeat_click_count = 0
                            action_signature = (
                                payload.get("tool"),
                                json.dumps(payload.get("args") or {}, sort_keys=True),
                            )
                            if action_signature == last_action:
                                await page.wait_for_timeout(3000)
                                observation = (
                                    f"Действие {action_signature[0]} повторено. Жду обновления страницы."
                                )
                                continue
                            if payload.get("tool") == "none":
                                if not read_in_subgoal and not auto_read_attempted:
                                    auto_read_attempted = True
                                    candidate_id = last_clicked_marker_id
                                    if candidate_id is None:
                                        try:
                                            candidate_id = _select_read_candidate(
                                                load_metadata(args.metadata)
                                            )
                                        except Exception:
                                            candidate_id = None
                                    if candidate_id is not None:
                                        try:
                                            auto_text = await tool_read(
                                                page, int(candidate_id), metadata_path=args.metadata
                                            )
                                        except Exception as exc:
                                            observation = f"Авто-чтение не удалось: {exc}"
                                            action_history.append(
                                                f"auto_read {{'marker_id': {candidate_id}}} -> ошибка"
                                            )
                                            continue
                                        summary = _format_excerpt(auto_text, 200) or "пустой результат"
                                        action_history.append(
                                            f"auto_read {{'marker_id': {candidate_id}}} -> {summary}"
                                        )
                                        last_read_text = auto_text or ""
                                        if (auto_text or "").strip():
                                            read_in_subgoal = True
                                        observation = f"Авто-чтение элемента: {summary}"
                                        continue
                                if last_action and last_action[0] == "open":
                                    print("Executor вернул none сразу после open. Проверяю, загрузилась ли страница.")
                                    try:
                                        await page.wait_for_load_state("load", timeout=20000)
                                    except Exception:
                                        pass
                                    try:
                                        loaded_metadata = await _annotate_with_retry(
                                            page, output_image=args.image, output_data=args.metadata
                                        )
                                        if loaded_metadata:
                                            observation = await _describe_observation(
                                                page,
                                                args.metadata,
                                                "Страница загружена, элементы доступны.",
                                                last_read_text,
                                            )
                                            completed_subgoals.append(current_subgoal)
                                            current_subgoal = None
                                            break
                                    except Exception:
                                        pass
                                    page_state = await _describe_page(page, args.metadata)
                                    observation = f"Страница загружалась после open, элементы не готовы. {page_state}"
                                    continue
                                if last_action is None:
                                    observation = await _describe_observation(
                                        page,
                                        args.metadata,
                                        "Executor не смог выполнить действие.",
                                        last_read_text,
                                    )
                                else:
                                    observation = await _describe_observation(
                                        page,
                                        args.metadata,
                                        "Подцель завершена по tool=none.",
                                        last_read_text,
                                    )
                                    completed_subgoals.append(current_subgoal)
                                current_subgoal = None
                                break
                            last_action = action_signature
                            should_stop, tool_result = await _execute_tool(
                                page, payload, metadata_path=args.metadata
                            )
                            if payload.get("tool") == "click":
                                marker_id = (payload.get("args") or {}).get("marker_id")
                                if marker_id is not None:
                                    try:
                                        last_clicked_marker_id = int(marker_id)
                                    except (TypeError, ValueError):
                                        last_clicked_marker_id = None
                        except Exception as exc:
                            observation = f"Ошибка выполнения tool: {exc}"
                            current_subgoal = None
                            break
                        if tool_result:
                            action_history.append(
                                f"{payload.get('tool')} {payload.get('args')} -> {tool_result[:200]}"
                            )
                        else:
                            action_history.append(f"{payload.get('tool')} {payload.get('args')}")
                        if payload.get("tool") == "read":
                            last_read_text = tool_result or ""
                            if (tool_result or "").strip():
                                read_in_subgoal = True
                        if should_stop:
                            observation = await _describe_observation(
                                page,
                                args.metadata,
                                "Executor вернул tool=none.",
                                last_read_text,
                            )
                            completed_subgoals.append(current_subgoal)
                            current_subgoal = None
                            break

                        if current_subgoal is None:
                            history.append({"user": global_goal, "assistant": observation})
                            save_history(args.history, history)
                            continue

                    # If executor repeatedly returns none on a stable page, ask planner for next subgoal.
        except KeyboardInterrupt:
            print("\nДиалог завершен.")
        finally:
            await context.close()
            if browser:
                await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
