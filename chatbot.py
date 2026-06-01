import json
import os
from datetime import datetime

from test import (
    collect_research_for_destinations,
    create_llm_provider,
    destination_agent,
    intent_agent,
    load_environment,
    normalize_trip_params,
    planning_agent,
)
from src.tools.travel_api_tools import validate_travel_input


HISTORY_PATH = "history.json"
MAX_HISTORY_MESSAGES = 8

# Simple state for collecting trip information across conversation
_current_trip_request = {}


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


def rewrite_user_request(llm, history: list, user_input: str) -> str:
    if not history:
        return user_input

    system_prompt = """
Bạn là bộ chuẩn hóa hội thoại cho travel planner.
Nhiệm vụ: dựa trên lịch sử hội thoại và tin nhắn mới nhất, viết lại thành một yêu cầu du lịch đầy đủ, độc lập.

Quy tắc:
- Giữ nguyên các thông tin quan trọng đã có: điểm đến/chủ đề, điểm xuất phát, số ngày, ngân sách, ngày đi, sở thích.
- Nếu user chỉ bổ sung một thông tin như "từ Hà Nội", hãy ghép nó vào yêu cầu du lịch trước đó.
- Nếu assistant vừa hỏi điểm xuất phát và user trả lời địa điểm xuất phát, hãy ghép địa điểm đó vào yêu cầu du lịch gần nhất.
- Không bịa thông tin chưa có.
- Trả về DUY NHẤT nội dung yêu cầu đã viết lại, không markdown, không giải thích.
""".strip()
    prompt = json.dumps(
        {
            "recent_dialogue": recent_dialogue(history),
            "new_user_message": user_input,
        },
        ensure_ascii=False,
        indent=2,
    )
    response = llm.generate(prompt, system_prompt=system_prompt)
    rewritten = str(response.get("content", "")).strip()
    return rewritten or user_input


def run_planner_turn(llm, standalone_request: str, current_trip: dict = None) -> tuple:
    """Run a single turn of the travel planner with input validation."""

    # Merge with existing trip info
    trip = dict(current_trip) if current_trip else {}

    # Validate the request
    validation_result = validate_travel_input(standalone_request)

    # Update trip with validated info
    normalized = validation_result.get("normalized_input", {})
    if normalized.get("origin"):
        trip["origin"] = normalized["origin"]
    if normalized.get("destination"):
        trip["destination"] = normalized["destination"]
    if normalized.get("budget"):
        trip["budget"] = normalized["budget"]
    if normalized.get("people"):
        trip["people"] = normalized["people"]
    if normalized.get("days"):
        trip["days"] = normalized["days"]

    # If validation fails, ask follow-up
    if not validation_result["is_valid"]:
        follow_up = validation_result.get("follow_up_question", "Bạn cần cung cấp thêm thông tin.")
        assumptions = validation_result.get("assumptions", [])
        answer = follow_up
        if assumptions:
            answer += "\n\n" + "Giả định: " + "; ".join(assumptions)
        return answer, trip, validation_result

    # Extract params for planning
    params = normalize_trip_params(intent_agent(llm, standalone_request))

    # Merge with validated info
    if trip.get("origin") and not params.get("origin"):
        params["origin"] = trip["origin"]
    if trip.get("budget") and not params.get("budget_vnd"):
        params["budget_vnd"] = trip["budget"]
    if trip.get("people") and not params.get("adults"):
        params["adults"] = trip["people"]

    if params.get("origin_missing"):
        answer = "Hãy cung cấp cho tôi thêm thông tin về địa điểm xuất phát của bạn"
        return answer, trip, validation_result

    destination_options = destination_agent(llm, standalone_request, params)
    if not destination_options:
        raise ValueError("Không tìm được địa điểm ứng viên phù hợp.")

    max_options = int(os.getenv("MAX_DESTINATION_OPTIONS", "3"))
    destination_options = destination_options[:max_options]
    destination_research = collect_research_for_destinations(
        params,
        destination_options,
    )
    answer = planning_agent(
        llm,
        standalone_request,
        params,
        destination_options,
        destination_research,
    )

    # Clear trip after successful planning
    cleared_trip = {}

    return answer, cleared_trip, validation_result


def print_intro(history: list) -> None:
    print("=== Travel Planner Chatbot ===")
    print("Nhập yêu cầu du lịch. Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print(f"Lịch sử hội thoại: {HISTORY_PATH} ({len(history)} messages)")
    print("Ví dụ: Gợi ý chuyến đi biển cuối tuần tới")


def main() -> int:
    load_environment()
    history = load_history()
    print_intro(history)

    # Simple state for multi-turn conversation
    current_trip = {}

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

        append_history(history, "user", user_input)

        try:
            standalone_request = rewrite_user_request(
                llm,
                history[:-1],
                user_input,
            )
            print("Đang xử lý yêu cầu:", standalone_request)
            answer, current_trip, validation_result = run_planner_turn(
                llm, standalone_request, current_trip
            )
        except Exception as error:
            answer = f"Mình chưa xử lý được lượt này: {error}"
            current_trip = {}
            validation_result = None

        print("\nBot:")
        print(answer)
        append_history(history, "assistant", answer, metadata={
            "validation": validation_result,
            "current_trip": current_trip,
        })


if __name__ == "__main__":
    raise SystemExit(main())
