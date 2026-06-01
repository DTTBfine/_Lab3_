import streamlit as st

from chatbot import (
    append_history,
    load_history,
    rewrite_user_request,
    save_history,
)
from test import create_llm_provider, load_environment
from src.tools.travel_api_tools import validate_travel_input


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
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "current_trip" not in st.session_state:
        st.session_state.current_trip = {}


def reset_history() -> None:
    st.session_state.history = []
    st.session_state.current_trip = {}
    save_history([])


def render_history() -> None:
    for message in st.session_state.history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            role = "assistant"
        with st.chat_message(role):
            st.markdown(content)


def handle_user_message(user_input: str) -> None:
    append_history(st.session_state.history, "user", user_input)

    try:
        llm = get_llm()
        standalone_request = rewrite_user_request(
            llm,
            st.session_state.history[:-1],
            user_input,
        )

        # Validate input
        validation_result = validate_travel_input(standalone_request)

        # Update current trip state
        current_trip = dict(st.session_state.current_trip)
        normalized = validation_result.get("normalized_input", {})
        if normalized.get("origin"):
            current_trip["origin"] = normalized["origin"]
        if normalized.get("destination"):
            current_trip["destination"] = normalized["destination"]
        if normalized.get("budget"):
            current_trip["budget"] = normalized["budget"]
        if normalized.get("people"):
            current_trip["people"] = normalized["people"]
        if normalized.get("days"):
            current_trip["days"] = normalized["days"]
        st.session_state.current_trip = current_trip

        with st.status("Đang xử lý yêu cầu...", expanded=False):
            st.write(f"Yêu cầu đã chuẩn hóa: {standalone_request}")

            # Check if validation passed
            if not validation_result["is_valid"]:
                follow_up = validation_result.get("follow_up_question", "")
                assumptions = validation_result.get("assumptions", [])

                st.warning("⚠️ Thiếu thông tin cần thiết:")
                if assumptions:
                    st.info("Giả định: " + "; ".join(assumptions))
                if follow_up:
                    st.info(follow_up)
                answer = follow_up or "Bạn cần cung cấp thêm thông tin."
            else:
                # Show assumptions if any
                if validation_result.get("assumptions"):
                    st.info("Giả định: " + "; ".join(validation_result["assumptions"]))

                # Run planning
                from chatbot import run_planner_turn
                answer, cleared_trip, _ = run_planner_turn(
                    llm, standalone_request, current_trip
                )
                st.session_state.current_trip = cleared_trip
    except Exception as error:
        answer = f"Mình chưa xử lý được lượt này: {error}"
        st.session_state.current_trip = {}

    append_history(st.session_state.history, "assistant", answer, metadata={
        "current_trip": st.session_state.current_trip,
    })


def main() -> None:
    init_state()

    st.title("Travel Planner Agent")
    st.caption("Chatbot gợi ý du lịch nhiều lượt, lưu lịch sử vào history.json")

    with st.sidebar:
        st.header("Lịch sử")
        st.write(f"{len(st.session_state.history)} tin nhắn")
        if st.button("Xóa lịch sử", type="secondary"):
            reset_history()
            st.rerun()
        st.divider()
        st.caption("Ví dụ")
        st.code("Gợi ý chuyến đi biển cuối tuần tới")
        st.code("từ Hà Nội")

    render_history()

    user_input = st.chat_input("Nhập yêu cầu du lịch của bạn...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        handle_user_message(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
