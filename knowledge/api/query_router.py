"""知识库查询 API 路由"""

import asyncio
import os
import uuid

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.main_graph import query_app
from knowledge.schemas.query_schema import QueryRequest
from knowledge.tools.paths_utils import get_front_page_dir
from knowledge.tools.sse_utils import create_sse_queue, sse_generator
from knowledge.tools.task_utils import (
    TASK_STATUS_COMPLETED, TASK_STATUS_PROCESSING,
    get_task_result, update_task_status,
)
from knowledge.tools.mongo_history_utils import get_recent_messages, clear_history


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="Query Service",
        description="知识库查询服务"
    )

    # 允许跨域（开发环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # 允许所有来源
        allow_credentials=True,   # 允许携带凭证
        allow_methods=["*"],      # 允许所有方法
        allow_headers=["*"],      # 允许所有头部
    )

    # 静态文件挂载（前端页面）
    page_dir = get_front_page_dir()
    if os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir), name="front")

    # 注册路由
    _register_routes(app)

    return app


def _register_routes(app: FastAPI):
    """注册路由"""

    @app.get("/chat.html")
    async def chat_page():
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query")
    async def query(request: QueryRequest, background_tasks: BackgroundTasks):
        """处理查询请求"""

        # 1. 解析参数
        user_query = request.query
        session_id = request.session_id or str(uuid.uuid4())
        is_stream = request.is_stream

        # 2. 更新任务状态
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

        # 3. 根据模式选择处理方式
        if is_stream:
            # 流式模式：后台任务 + SSE
            background_tasks.add_task(
                _run_query_graph, session_id, user_query, is_stream
            )
            await asyncio.sleep(0.1)  # 等待队列创建
            return {"message": "Query submitted", "session_id": session_id}
        else:
            # 非流式模式：同步执行
            _run_query_graph(session_id, user_query, is_stream)
            answer = get_task_result(session_id, "answer", "")
            return {
                "message": "处理完成",
                "session_id": session_id,
                "answer": answer
            }

    @app.get("/stream/{session_id}")
    async def stream(session_id: str, request: Request):
        """SSE 实时返回结果"""
        return StreamingResponse(
            sse_generator(session_id, request),
            media_type="text/event-stream"
        )

    @app.get("/history/{session_id}")
    async def history(session_id: str, limit: int = 50):
        """查询当前会话历史记录"""
        records = get_recent_messages(session_id, limit=limit)
        items = [
            {
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "ts": r.get("ts")
            }
            for r in records
        ]
        return {"session_id": session_id, "items": items}

    @app.delete("/history/{session_id}")
    async def clear_chat_history(session_id: str):
        """清除会话历史记录"""
        count = clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def _run_query_graph(session_id: str, user_query: str, is_stream: bool):
    """后台任务：执行查询流程图"""

    # 流式模式：创建 SSE 队列
    if is_stream:
        create_sse_queue(session_id)

    # 构建初始状态
    default_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream
    }

    # 执行查询流程图
    query_app.invoke(default_state)

    # 更新任务状态
    update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)


app = create_app()

if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=8001)
