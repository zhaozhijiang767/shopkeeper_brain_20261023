import json
import queue
from typing import Dict, Any


class SSEEvent:
    """SSE 事件类型常量"""
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号

# 全局 SSE 会话队列存储
_session_stream: Dict[str, queue.Queue] = {}


def create_sse_queue(session_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _session_stream[session_id] = q
    return q


def get_sse_queue(session_id: str):
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列"""
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """向指定会话推送事件"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})