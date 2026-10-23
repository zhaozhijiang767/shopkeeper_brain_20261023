"""知识库导入 API 路由"""

import os

from typing import List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.processor.import_process.base import setup_logging
from knowledge.schemas.task_schema import TaskStatusResponse
from knowledge.schemas.upload_schema import UploadResponse
from knowledge.services.file_import_service import get_file_import_service
from knowledge.services.task_service import TaskService
from knowledge.tools.paths_utils import get_front_page_dir


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="Import Service",
        description="知识库导入服务"
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

    @app.post("/upload", response_model=UploadResponse)
    async def upload_files(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...)
    ):
        """上传文件"""

        # 1. 获取文件导入 service
        service = get_file_import_service()

        # 2. 处理导入文件（保存到本地和对象存储）
        task_ids, date_dir = service.process_files(files)

        # 3. 添加后台任务处理导入流程
        for task_id, file in zip(task_ids, files):
            file_dir = os.path.join(date_dir, task_id)
            import_file_path = os.path.join(file_dir, file.filename)

            background_tasks.add_task(
                _run_import_task,
                task_id,
                file_dir,
                import_file_path
            )

        # 4. 立即返回任务 ID（不等待处理完成）
        return UploadResponse(
            message="Files uploaded successfully",
            task_ids=task_ids
        )

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_task(task_id: str):
        """获取任务状态"""
        task_info = TaskService.get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def _run_import_task(task_id: str, file_dir: str, import_file_path: str):
    """后台任务：执行导入流程"""
    setup_logging()

    service = get_file_import_service()
    service.run_import_task(task_id, file_dir, import_file_path)


app = create_app()

if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=8000)
