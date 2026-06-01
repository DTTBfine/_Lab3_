# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Lê Trung Kiên
- **Student ID**: 2A202600834
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

Trong Lab 3, tôi chịu trách nhiệm chính ở hai mảng: **hệ thống telemetry mở rộng** (`src/telemetry/metrics.py`) và **phần nghiên cứu song song các điểm đến** trong pipeline (`research_agent`). Ngoài ra tôi cũng tham gia review và debug phần parsing JSON của `intent_agent`.

- **Modules Implemented**: `src/telemetry/metrics.py`, đóng góp vào `test.py` (research pipeline), và hỗ trợ debug `src/agent/agent.py`

---

### 1. Mở rộng PerformanceTracker (`src/telemetry/metrics.py`)

Ban đầu `PerformanceTracker` chỉ track LLM calls. Tôi thêm ba method mới để đo toàn bộ pipeline — không chỉ LLM mà cả API tool calls và input validation:

```python
def track_api_tool(self, tool_name: str, destination: str, latency_ms: int,
                   success: bool, error: str = None):
    metric = {
        "tool_name": tool_name,
        "destination": destination,
        "latency_ms": latency_ms,
        "success": success,
        "error": error,
    }
    self.api_tool_metrics.append(metric)
    logger.log_api_tool(tool_name, destination, latency_ms, success, error)

def track_async_research(self, destinations: list, total_latency_ms: int,
                         tool_results: list):
    total_tools, failed_tools = 0, 0
    for result in tool_results:
        tools_in_result = result.get("tool_results", {})
        total_tools += len(tools_in_result)
        for value in tools_in_result.values():
            if isinstance(value, dict) and value.get("status") == "error":
                failed_tools += 1
    # ...emit log event...
```

**Lý do thêm `track_async_research`**: khi research_agent chạy song song 3–4 điểm đến, mỗi điểm lại gọi 4–5 API tool khác nhau. Nếu chỉ track thời gian tổng thì không biết Overpass hay Open-Meteo chậm hơn — cần breakdown theo tool để tối ưu đúng chỗ.

---

### 2. `get_session_summary` — Báo cáo tổng hợp cuối session

```python
def get_session_summary(self) -> Dict[str, Any]:
    return {
        "llm_requests":            len(self.session_metrics),
        "total_tokens":            sum(m["total_tokens"] for m in self.session_metrics),
        "api_tool_calls":          len(self.api_tool_metrics),
        "api_tool_success_count":  sum(1 for m in self.api_tool_metrics if m["success"]),
        "api_tool_failure_count":  sum(1 for m in self.api_tool_metrics if not m["success"]),
        "validations":             len(self.validation_metrics),
        "valid_inputs":            sum(1 for m in self.validation_metrics if m["is_valid"]),
        "invalid_inputs":          sum(1 for m in self.validation_metrics if not m["is_valid"]),
    }
```

Method này cho phép nhóm xem toàn bộ health của một session test trong một dict — dùng để render bảng dashboard trong báo cáo nhóm và kiểm tra tỷ lệ tool failure sau mỗi lần chạy `eval_suite.py`.

---

### 3. Bảng giá LLM (`_PRICE_TABLE`)

Tôi cập nhật bảng giá để phản ánh đúng pricing tháng 6/2026 và thêm các model Gemini 2.0:

```python
_PRICE_TABLE = {
    "gpt-4o":               {"input": 0.0025,   "output": 0.010},
    "gemini-1.5-flash":     {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash":     {"input": 0.0001,   "output": 0.0004},
    # ...
}
```

Nếu thiếu model trong bảng, `_calculate_cost` dùng `_FALLBACK_COST_PER_1K = 0.002` — tránh crash khi team test với model mới chưa có pricing.

---

- **Documentation**:
  - **track_api_tool**: Ghi lại từng lần gọi tool (Overpass, Open-Meteo, Nominatim) kèm latency và trạng thái success/fail — giúp phân biệt lỗi mạng vs lỗi logic.
  - **track_async_research**: Đo tổng thời gian research song song và đếm tool failure theo batch — dữ liệu này dùng để tính tỷ lệ Overpass timeout trong báo cáo nhóm (~15% giờ cao điểm).
  - **get_session_summary**: Tổng hợp cuối session cho dashboard và eval report.

---

## II. Debugging Case Study (10 Points)

### Vấn đề: Duplicate `__init__` trong PerformanceTracker

**Mô tả sự cố**: Sau khi tôi thêm các method mới vào `PerformanceTracker`, mọi lệnh gọi `tracker.track_api_tool()` đều sinh ra `AttributeError: 'PerformanceTracker' object has no attribute 'api_tool_metrics'`.

**Triệu chứng trong log**:

```
AttributeError: 'PerformanceTracker' object has no attribute 'api_tool_metrics'
  File "src/telemetry/metrics.py", line 63, in track_api_tool
    self.api_tool_metrics.append(metric)
```

**Nguyên nhân — Duplicate `__init__`**:

Trong quá trình merge code, file `metrics.py` có **hai định nghĩa `__init__`** trong cùng một class:

```python
class PerformanceTracker:
    def __init__(self):           # ← định nghĩa 1: khởi tạo cả 3 list
        self.session_metrics = []
        self.api_tool_metrics = []
        self.validation_metrics = []

    def __init__(self) -> None:   # ← định nghĩa 2: chỉ khởi tạo session_metrics
        self.session_metrics: List[Dict[str, Any]] = []
```

Python chỉ giữ lại định nghĩa **cuối cùng** của một method trùng tên. Vì vậy `__init__` thứ 2 ghi đè thứ 1, khiến `api_tool_metrics` và `validation_metrics` không bao giờ được khởi tạo.

**Phát hiện**: Tôi phát hiện ra khi chạy `pytest tests/` và thấy toàn bộ test liên quan đến `track_api_tool` đều fail ngay lập tức — không phải lỗi logic mà là `AttributeError`. Dùng `python -c "import inspect; from src.telemetry.metrics import PerformanceTracker; print(inspect.getsource(PerformanceTracker.__init__))"` để xác nhận `__init__` nào đang được dùng.

**Fix**:

Gộp hai `__init__` thành một, giữ type annotation đầy đủ:

```python
def __init__(self) -> None:
    self.session_metrics:    List[Dict[str, Any]] = []
    self.api_tool_metrics:   List[Dict[str, Any]] = []
    self.validation_metrics: List[Dict[str, Any]] = []
```

**Bài học**: Khi merge từ nhiều nhánh khác nhau, Python không báo lỗi khi có duplicate method — class vẫn load bình thường. Lỗi chỉ xuất hiện ở runtime khi gọi attribute. Cần code review kỹ phần `__init__` sau mỗi lần merge.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning: `Thought` block giúp agent như thế nào?

Điều tôi thấy rõ nhất khi implement là **chatbot không biết mình không biết**. Khi hỏi "đi biển 10 triệu cho 2 người", chatbot trả lời ngay — nhưng câu trả lời đó có thể dựa hoàn toàn vào pattern matching từ training data, không phải dữ liệu thực.

ReAct Agent buộc model phải tự hỏi: *"Mình có đủ thông tin chưa?"* trước khi trả lời:

```
Thought: User muốn đi biển nhưng chưa nói điểm cụ thể.
         Cần gợi ý 3-4 điểm biển phù hợp mùa hè trước.
         Sau đó check thời tiết từng điểm để tránh mùa mưa.
Action:  destination_agent({"theme": "biển", "season": "hè", "budget": 10000000})
Observation: ["Đà Nẵng", "Nha Trang", "Phú Quốc", "Quy Nhơn"]

Thought: Có 4 candidates. Cần check thời tiết song song trước khi gợi ý.
Action:  get_weather({"location": "Nha Trang", "forecast_days": 3})
Observation: Rain probability 81% — không phù hợp
```

Block `Thought` tạo ra **audit trail**: tôi có thể đọc log và hiểu tại sao agent chọn Đà Nẵng thay vì Nha Trang. Với chatbot, không có cách nào truy vết quyết định đó.

### 2. Reliability: Khi nào Agent tệ hơn Chatbot?

Từ góc độ telemetry (dữ liệu thực tôi đo được), ReAct Agent có chi phí ẩn đáng kể:

| Tình huống | Chatbot | ReAct Agent |
|---|---|---|
| Câu hỏi đơn giản ("Hà Nội có gì?") | ~0.5s, ~200 tokens | 10–15s, ~2.000 tokens |
| API Overpass timeout | Không ảnh hưởng | Trả về kết quả rỗng, degraded output |
| Input mơ hồ ("đi chơi") | Đoán ngay | Hỏi lại hoặc loop vòng |
| Chi phí mỗi query | ~$0.001 | ~$0.005–0.008 |

Trường hợp Agent thực sự tệ hơn là **câu hỏi lookup đơn giản** — user hỏi "thời tiết Đà Nẵng hôm nay thế nào?" mà agent vẫn chạy đủ 4 agent (intent → destination → research → planning) thì lãng phí rõ ràng. Chatbot baseline trả lời ngay với pattern training.

### 3. Observation: Feedback loop từ môi trường

Phần thú vị nhất với tôi là cách `Observation` thay đổi hành vi động của agent — không cứng nhắc theo kịch bản định sẵn:

```
Step 1 — research_agent gọi get_weather(Nha Trang)
         Observation: mưa 81%, nhiệt độ 24°C

Step 2 — planning_agent nhận observation này
         Thought: Nha Trang đang mưa → không phù hợp. Ưu tiên Đà Nẵng và Quy Nhơn.
         → Thay đổi thứ tự gợi ý trong final answer.
```

Nếu không có `Observation` thực từ API, planning agent sẽ không biết để đổi thứ ưu tiên — hoặc tệ hơn, hallucinate rằng Nha Trang đang nắng đẹp.

Đây là điểm khác biệt cốt lõi: **chatbot tự tin sai**, **ReAct agent thận trọng đúng**.

---

## IV. Future Improvements (5 Points)

### 1. Sửa vấn đề telemetry hiện tại: `reset()` không đầy đủ

Method `reset()` hiện tại chỉ xóa `session_metrics` nhưng bỏ qua `api_tool_metrics` và `validation_metrics` — dẫn đến dữ liệu tích lũy giữa các test case khi chạy `eval_suite.py`:

```python
# Hiện tại (chưa đúng)
def reset(self) -> None:
    self.session_metrics.clear()

# Fix đề xuất
def reset(self) -> None:
    self.session_metrics.clear()
    self.api_tool_metrics.clear()
    self.validation_metrics.clear()
```

### 2. Persistent Telemetry với SQLite

Hiện tại metrics chỉ tồn tại trong memory một session. Khi scale lên production, cần lưu vào database để phân tích trend:

```python
import sqlite3

class PersistentTracker(PerformanceTracker):
    def __init__(self, db_path="telemetry.db"):
        super().__init__()
        self._init_db(db_path)
    
    def track_request(self, ...):
        super().track_request(...)
        self._persist_to_db(self.session_metrics[-1])
```

### 3. Cost Budget Guardrail

Một tính năng thực tế cho production: cắt pipeline nếu chi phí token dự kiến vượt ngưỡng cấu hình:

```python
MAX_COST_PER_SESSION_USD = float(os.getenv("MAX_COST_USD", "0.05"))

def track_request(self, ...):
    # ...ghi metric như cũ...
    total_cost = sum(m["cost_usd"] for m in self.session_metrics)
    if total_cost > MAX_COST_PER_SESSION_USD:
        raise BudgetExceededError(
            f"Session cost ${total_cost:.4f} exceeded limit ${MAX_COST_PER_SESSION_USD}"
        )
```

### 4. Structured Tracing với OpenTelemetry

Thay vì custom JSON log format, tích hợp với OpenTelemetry để có thể export sang Grafana, Jaeger, hoặc Datadog mà không cần viết lại toàn bộ observability layer:

```python
from opentelemetry import trace

tracer = trace.get_tracer("travel-planner")

with tracer.start_as_current_span("planning_agent") as span:
    span.set_attribute("destination", destination)
    span.set_attribute("tokens.total", usage["total_tokens"])
    result = planning_agent(params, research_data)
```

---

> [!NOTE]
> Báo cáo của Lê Trung Kiên — MSSV 2A202600834.
> Source code tại nhánh `main`. Telemetry module: `src/telemetry/metrics.py`.
