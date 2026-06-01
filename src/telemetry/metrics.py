import time
from typing import Dict, Any, List
from src.telemetry.logger import logger

class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs and API tools.
    """
    def __init__(self):
        self.session_metrics = []
        self.api_tool_metrics = []
        self.validation_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_estimate": self._calculate_cost(model, usage)
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def track_api_tool(self, tool_name: str, destination: str, latency_ms: int,
                       success: bool, error: str = None):
        """Track API tool call metrics."""
        metric = {
            "tool_name": tool_name,
            "destination": destination,
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
        }
        self.api_tool_metrics.append(metric)
        logger.log_api_tool(tool_name, destination, latency_ms, success, error)

    def track_validation(self, is_valid: bool, missing_fields: List[str],
                        assumptions: List[str]):
        """Track validation metrics."""
        metric = {
            "is_valid": is_valid,
            "missing_fields": missing_fields,
            "assumptions_count": len(assumptions),
            "validation_missing_fields_count": len(missing_fields),
        }
        self.validation_metrics.append(metric)
        logger.log_event("VALIDATION_METRIC", metric)

    def track_async_research(self, destinations: List[str], total_latency_ms: int,
                             tool_results: List[Dict]):
        """Track async research metrics."""
        total_tools = 0
        failed_tools = 0
        for result in tool_results:
            tools_in_result = result.get("tool_results", {})
            total_tools += len(tools_in_result)
            for key, value in tools_in_result.items():
                if isinstance(value, dict) and value.get("status") == "error":
                    failed_tools += 1

        metric = {
            "destinations_count": len(destinations),
            "total_research_latency_ms": total_latency_ms,
            "total_tools_called": total_tools,
            "api_tool_success_count": total_tools - failed_tools,
            "api_tool_failure_count": failed_tools,
        }
        logger.log_async_research(destinations, total_latency_ms, total_tools, failed_tools)
        return metric

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        TODO: Implement real pricing logic.
        For now, returns a dummy constant.
        """
        return (usage.get("total_tokens", 0) / 1000) * 0.01

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics for the session."""
        return {
            "llm_requests": len(self.session_metrics),
            "total_tokens": sum(m["total_tokens"] for m in self.session_metrics),
            "api_tool_calls": len(self.api_tool_metrics),
            "api_tool_success_count": sum(1 for m in self.api_tool_metrics if m["success"]),
            "api_tool_failure_count": sum(1 for m in self.api_tool_metrics if not m["success"]),
            "validations": len(self.validation_metrics),
            "valid_inputs": sum(1 for m in self.validation_metrics if m["is_valid"]),
            "invalid_inputs": sum(1 for m in self.validation_metrics if not m["is_valid"]),
        }

# Global tracker instance
tracker = PerformanceTracker()
