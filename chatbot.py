import json
import os
import uuid
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


HISTORY_PATH = "history.json"
MAX_HISTORY_MESSAGES = 8


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_conversation(title: str = "Hội thoại mới") -> dict:
    now = _timestamp()
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def _normalize_history_store(data) -> dict:
    if isinstance(data, list):
        conversation = _new_conversation("Hội thoại đã lưu")
        conversation["messages"] = data
        return {
            "active_conversation_id": conversation["id"],
            "conversations": [conversation],
        }

    if not isinstance(data, dict):
        conversation = _new_conversation()
        return {
            "active_conversation_id": conversation["id"],
            "conversations": [conversation],
        }

    conversations = data.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        conversation = _new_conversation()
        conversations = [conversation]

    normalized = []
    for index, conversation in enumerate(conversations, start=1):
        if not isinstance(conversation, dict):
            continue
        conversation_id = conversation.get("id") or str(uuid.uuid4())
        messages = conversation.get("messages")
        normalized.append(
            {
                "id": conversation_id,
                "title": conversation.get("title") or f"Hội thoại {index}",
                "created_at": conversation.get("created_at") or _timestamp(),
                "updated_at": conversation.get("updated_at") or _timestamp(),
                "messages": messages if isinstance(messages, list) else [],
            }
        )

    if not normalized:
        normalized = [_new_conversation()]

    active_id = data.get("active_conversation_id")
    if active_id not in {item["id"] for item in normalized}:
        active_id = normalized[-1]["id"]

    return {
        "active_conversation_id": active_id,
        "conversations": normalized,
    }


def load_history_store(path: str = HISTORY_PATH) -> dict:
    if not os.path.exists(path):
        return _normalize_history_store(None)

    try:
        with open(path, "r", encoding="utf-8") as history_file:
            data = json.load(history_file)
    except (json.JSONDecodeError, OSError):
        data = None

    store = _normalize_history_store(data)
    save_history_store(store, path=path)
    return store


def save_history_store(store: dict, path: str = HISTORY_PATH) -> None:
    with open(path, "w", encoding="utf-8") as history_file:
        json.dump(store, history_file, ensure_ascii=False, indent=2)


def get_conversations(store: dict) -> list:
    return store.get("conversations", [])


def get_active_conversation(store: dict) -> dict:
    conversations = get_conversations(store)
    active_id = store.get("active_conversation_id")
    for conversation in conversations:
        if conversation.get("id") == active_id:
            return conversation
    if conversations:
        store["active_conversation_id"] = conversations[-1]["id"]
        return conversations[-1]

    conversation = _new_conversation()
    store["active_conversation_id"] = conversation["id"]
    store["conversations"] = [conversation]
    return conversation


def get_active_messages(store: dict) -> list:
    return get_active_conversation(store).setdefault("messages", [])


def create_conversation(store: dict, title: str = "Hội thoại mới") -> dict:
    conversation = _new_conversation(title)
    store.setdefault("conversations", []).append(conversation)
    store["active_conversation_id"] = conversation["id"]
    return conversation


def set_active_conversation(store: dict, conversation_id: str) -> None:
    if conversation_id in {item.get("id") for item in get_conversations(store)}:
        store["active_conversation_id"] = conversation_id


def reset_history_store(path: str = HISTORY_PATH) -> dict:
    store = _normalize_history_store(None)
    save_history_store(store, path=path)
    return store


def append_message_to_active(
    store: dict,
    role: str,
    content: str,
    metadata: dict = None,
    path: str = HISTORY_PATH,
) -> None:
    item = {
        "timestamp": _timestamp(),
        "role": role,
        "content": content,
    }
    if metadata:
        item["metadata"] = metadata

    conversation = get_active_conversation(store)
    conversation.setdefault("messages", []).append(item)
    conversation["updated_at"] = item["timestamp"]
    if role == "user" and len(conversation["messages"]) == 1:
        conversation["title"] = content[:48] + ("..." if len(content) > 48 else "")
    save_history_store(store, path=path)


def load_history() -> list:
    return get_active_messages(load_history_store())


def save_history(history: list) -> None:
    conversation = _new_conversation("Hội thoại đã lưu")
    conversation["messages"] = history
    store = {
        "active_conversation_id": conversation["id"],
        "conversations": [conversation],
    }
    save_history_store(store)


def append_history(history: list, role: str, content: str, metadata: dict = None) -> None:
    item = {
        "timestamp": _timestamp(),
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


def run_planner_turn(llm, standalone_request: str) -> tuple:
    params = normalize_trip_params(intent_agent(llm, standalone_request))
    if params.get("origin_missing"):
        answer = "Hãy cung cấp cho tôi thêm thông tin về địa điểm xuất phát của bạn"
        metadata = {
            "standalone_request": standalone_request,
            "params": params,
            "needs_origin": True,
        }
        return answer, metadata

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
    metadata = {
        "standalone_request": standalone_request,
        "params": params,
        "destination_options": destination_options,
    }
    return answer, metadata


def print_intro(history: list) -> None:
    print("=== Travel Planner Chatbot ===")
    print("Nhập yêu cầu du lịch. Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print(f"Lịch sử hội thoại: {HISTORY_PATH} ({len(history)} messages)")
    print("Ví dụ: Gợi ý chuyến đi biển cuối tuần tới")


def main() -> int:
    load_environment()
    store = load_history_store()
    history = get_active_messages(store)
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

        append_message_to_active(store, "user", user_input)
        history = get_active_messages(store)

        try:
            standalone_request = rewrite_user_request(
                llm,
                history[:-1],
                user_input,
            )
            print("Đang xử lý yêu cầu:", standalone_request)
            answer, metadata = run_planner_turn(llm, standalone_request)
        except Exception as error:
            answer = f"Mình chưa xử lý được lượt này: {error}"
            metadata = None

        print("\nBot:")
        print(answer)
        append_message_to_active(store, "assistant", answer, metadata=metadata)
        history = get_active_messages(store)


if __name__ == "__main__":
    raise SystemExit(main())
