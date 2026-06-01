import streamlit as st

from chatbot import (
    append_message_to_active,
    create_conversation,
    get_active_messages,
    get_conversations,
    load_history_store,
    reset_history_store,
    rewrite_user_request,
    run_planner_turn,
    save_history_store,
    set_active_conversation,
)
from test import create_llm_provider, load_environment


st.set_page_config(
    page_title="Travel Planner Agent",
    page_icon="✈️",
    layout="centered",
)


@st.cache_resource
def get_llm():
    load_environment()
    return create_llm_provider()


def init_state() -> None:
    if "history_store" not in st.session_state:
        st.session_state.history_store = load_history_store()


def reset_history() -> None:
    st.session_state.history_store = reset_history_store()


def create_new_chat() -> None:
    create_conversation(st.session_state.history_store)
    save_history_store(st.session_state.history_store)


def render_history() -> None:
    messages = get_active_messages(st.session_state.history_store)
    if not messages:
        st.info("Chưa có tin nhắn trong hội thoại này.")
        return

    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            role = "assistant"
        with st.chat_message(role):
            st.markdown(content)


def handle_user_message(user_input: str) -> None:
    append_message_to_active(st.session_state.history_store, "user", user_input)

    try:
        llm = get_llm()
        messages = get_active_messages(st.session_state.history_store)
        standalone_request = rewrite_user_request(
            llm,
            messages[:-1],
            user_input,
        )

        with st.status("Đang xử lý yêu cầu...", expanded=False):
            st.write(f"Yêu cầu đã chuẩn hóa: {standalone_request}")
            answer, metadata = run_planner_turn(llm, standalone_request)
    except Exception as error:
        answer = f"Mình chưa xử lý được lượt này: {error}"
        metadata = None

    append_message_to_active(
        st.session_state.history_store,
        "assistant",
        answer,
        metadata=metadata,
    )


def conversation_label(index: int, conversation: dict) -> str:
    title = conversation.get("title") or "Hội thoại"
    count = len(conversation.get("messages", []))
    updated_at = conversation.get("updated_at", "")
    suffix = updated_at.replace("T", " ")[:16] if updated_at else ""
    return f"{index + 1}. {title} ({count}) {suffix}"


def render_sidebar() -> None:
    st.header("Lịch sử hội thoại")
    conversations = get_conversations(st.session_state.history_store)
    active_id = st.session_state.history_store.get("active_conversation_id")

    if not conversations:
        create_new_chat()
        conversations = get_conversations(st.session_state.history_store)
        active_id = st.session_state.history_store.get("active_conversation_id")

    labels = [
        conversation_label(index, conversation)
        for index, conversation in enumerate(conversations)
    ]
    ids = [conversation.get("id") for conversation in conversations]
    active_index = ids.index(active_id) if active_id in ids else 0

    selected_label = st.radio(
        "Chọn hội thoại",
        labels,
        index=active_index,
    )
    selected_index = labels.index(selected_label)
    selected_id = ids[selected_index]
    if selected_id != active_id:
        set_active_conversation(st.session_state.history_store, selected_id)
        save_history_store(st.session_state.history_store)
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hội thoại mới", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()
    with col2:
        if st.button("Xóa tất cả", type="secondary", use_container_width=True):
            reset_history()
            st.rerun()

    active_messages = get_active_messages(st.session_state.history_store)
    st.caption(f"{len(conversations)} hội thoại, {len(active_messages)} tin nhắn đang mở")

    with st.expander("Xem cấu trúc lưu trữ", expanded=False):
        st.code("history.json -> conversations[] -> messages[]")


def main() -> None:
    init_state()

    st.title("Travel Planner Agent")
    st.caption("Chatbot gợi ý du lịch nhiều lượt, lưu từng cuộc hội thoại vào history.json")

    with st.sidebar:
        render_sidebar()
        st.divider()

    render_history()

    user_input = st.chat_input("Hãy nhập yêu cầu của bạn...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        handle_user_message(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
