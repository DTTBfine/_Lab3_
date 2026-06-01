import time

import streamlit as st

from chatbot import (
    append_history,
    load_history,
    rewrite_user_request,
    run_planner_turn,
    save_history,
    validate_user_input,
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
    if "history" not in st.session_state:
        st.session_state.history = load_history()


def reset_history() -> None:
    st.session_state.history = []
    save_history([])


def render_history() -> None:
    for message in st.session_state.history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            role = "assistant"
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                total_s = message.get("metadata", {}).get("timings", {}).get("total_s")
                if total_s:
                    st.caption(f"Thời gian phản hồi: {total_s}s")


def handle_user_message(user_input: str) -> None:
    append_history(st.session_state.history, "user", user_input)

    total_start = time.perf_counter()
    rewrite_time = 0.0
    metadata = None

    try:
        llm = get_llm()

        t0 = time.perf_counter()
        standalone_request = rewrite_user_request(
            llm,
            st.session_state.history[:-1],
            user_input,
        )
        rewrite_time = time.perf_counter() - t0

        with st.status("Đang xử lý yêu cầu...", expanded=True):
            st.write(f"Chuẩn hóa hội thoại ({rewrite_time:.1f}s): {standalone_request}")
            answer, metadata = run_planner_turn(llm, standalone_request)
            _render_timing_breakdown(metadata, rewrite_time)
    except Exception as error:
        answer = f"Mình chưa xử lý được lượt này: {error}"

    total_time = time.perf_counter() - total_start
    if metadata is None:
        metadata = {}
    metadata.setdefault("timings", {})
    metadata["timings"]["rewrite_s"] = round(rewrite_time, 2)
    metadata["timings"]["total_s"] = round(total_time, 2)
    st.session_state.last_timing = metadata["timings"]

    append_history(st.session_state.history, "assistant", answer, metadata=metadata)


def _render_timing_breakdown(metadata: dict | None, rewrite_time: float) -> None:
    if not metadata:
        return
    t = metadata.get("timings", {})
    labels = [
        ("Ý định", t.get("intent_s")),
        ("Địa điểm", t.get("destination_s")),
        ("Thu thập dữ liệu", t.get("research_s")),
        ("Lập kế hoạch", t.get("planning_s")),
    ]
    available = [(label, val) for label, val in labels if val is not None]
    if not available:
        return
    cols = st.columns(len(available))
    for col, (label, val) in zip(cols, available):
        col.metric(label, f"{val}s")


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

        if "last_timing" in st.session_state:
            t = st.session_state.last_timing
            st.divider()
            st.subheader("Lần phản hồi cuối")
            total_s = t.get("total_s")
            if total_s:
                st.metric("Tổng thời gian", f"{total_s}s")
            breakdown = [
                ("Chuẩn hóa", t.get("rewrite_s")),
                ("Ý định", t.get("intent_s")),
                ("Địa điểm", t.get("destination_s")),
                ("Dữ liệu", t.get("research_s")),
                ("Lập kế hoạch", t.get("planning_s")),
            ]
            for label, val in breakdown:
                if val is not None:
                    st.caption(f"{label}: {val}s")

        st.divider()
        st.caption("Ví dụ")
        st.code("Gợi ý chuyến đi biển cuối tuần tới")
        st.code("từ Hà Nội")

    render_history()

    user_input = st.chat_input("Nhập yêu cầu du lịch của bạn...")
    if user_input:
        try:
            user_input = validate_user_input(user_input)
        except ValueError as validation_error:
            st.warning(str(validation_error))
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_input)

        handle_user_message(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
