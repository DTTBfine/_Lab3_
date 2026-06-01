import json
import os
import time
from datetime import datetime

from prompts import REWRITE_REQUEST_PROMPT
from src.telemetry.logger import logger
from test import (
    collect_research_for_destinations,
    create_llm_provider,
    destination_agent,
    intent_agent,
    load_environment,
    normalize_trip_params,
    planning_agent,
)


HISTORY_PATH = "history.json"
MAX_HISTORY_MESSAGES = 8
MAX_INPUT_LENGTH = 2000


def load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as history_file:
            data = json.load(history_file)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []
    return data


def save_history(history: list) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, ensure_ascii=False, indent=2)


def append_history(history: list, role: str, content: str, metadata: dict = None) -> None:
    item = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    if metadata:
        item["metadata"] = metadata
    history.append(item)
    save_history(history)


def recent_dialogue(history: list) -> list:
    compact = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        compact.append(
            {
                "role": message.get("role"),
                "content": message.get("content"),
            }
        )
    return compact


def validate_user_input(user_input: str) -> str:
    """Sanitize and enforce guardrails on raw user input. Raises ValueError on violations."""
    stripped = user_input.strip()
    if not stripped:
        raise ValueError("Yêu cầu không được để trống.")
    if len(stripped) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Yêu cầu quá dài ({len(stripped)} ký tự). Tối đa {MAX_INPUT_LENGTH} ký tự."
        )
    return stripped


def rewrite_user_request(llm, history: list, user_input: str) -> str:
    if not history:
        return user_input

    prompt = json.dumps(
        {
            "recent_dialogue": recent_dialogue(history),
            "new_user_message": user_input,
        },
        ensure_ascii=False,
        indent=2,
    )
    response = llm.generate(prompt, system_prompt=REWRITE_REQUEST_PROMPT)
    rewritten = str(response.get("content", "")).strip()
    return rewritten or user_input


def run_planner_turn(llm, standalone_request: str) -> tuple:
    timings: dict = {}

    t0 = time.perf_counter()
    params = normalize_trip_params(intent_agent(llm, standalone_request))
    timings["intent_s"] = round(time.perf_counter() - t0, 2)

    if params.get("origin_missing"):
        answer = "Hãy cung cấp cho tôi thêm thông tin về địa điểm xuất phát của bạn"
        metadata = {
            "standalone_request": standalone_request,
            "params": params,
            "needs_origin": True,
            "timings": timings,
        }
        return answer, metadata

    t0 = time.perf_counter()
    destination_options = destination_agent(llm, standalone_request, params)
    timings["destination_s"] = round(time.perf_counter() - t0, 2)

    if not destination_options:
        raise ValueError("Không tìm được địa điểm ứng viên phù hợp.")

    max_options = int(os.getenv("MAX_DESTINATION_OPTIONS", "3"))
    destination_options = destination_options[:max_options]

    t0 = time.perf_counter()
    destination_research = collect_research_for_destinations(
        params,
        destination_options,
    )
    timings["research_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    answer = planning_agent(
        llm,
        standalone_request,
        params,
        destination_options,
        destination_research,
    )
    timings["planning_s"] = round(time.perf_counter() - t0, 2)

    from src.telemetry.metrics import tracker

    metadata = {
        "standalone_request": standalone_request,
        "params": params,
        "destination_options": destination_options,
        "timings": timings,
        "perf_stats": tracker.get_summary_stats(),
    }
    return answer, metadata


def print_intro(history: list) -> None:
    print("=== Travel Planner Chatbot ===")
    print("Nhập yêu cầu du lịch. Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print(f"Lịch sử hội thoại: {HISTORY_PATH} ({len(history)} messages)")
    print("Ví dụ: Gợi ý chuyến đi biển cuối tuần tới")


def main() -> int:
    load_environment()
    history = load_history()
    print_intro(history)

    try:
        llm = create_llm_provider()
    except Exception as error:
        print(f"Lỗi khởi tạo LLM: {error}")
        return 1

    while True:
        user_input = input("\nBạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Tạm biệt.")
            return 0

        try:
            user_input = validate_user_input(user_input)
        except ValueError as validation_error:
            print(f"[Guardrail] {validation_error}")
            continue

        append_history(history, "user", user_input)

        try:
            standalone_request = rewrite_user_request(
                llm,
                history[:-1],
                user_input,
            )
            print("Đang xử lý yêu cầu:", standalone_request)
            answer, metadata = run_planner_turn(llm, standalone_request)
        except Exception as error:
            logger.log_event("PIPELINE_ERROR", {"input": user_input, "error": str(error)})
            answer = f"Mình chưa xử lý được lượt này: {error}"
            metadata = None

        print("\nBot:")
        print(answer)
        append_history(history, "assistant", answer, metadata=metadata)


if __name__ == "__main__":
    raise SystemExit(main())
