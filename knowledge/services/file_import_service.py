"""文件导入服务"""

import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Tuple

from dotenv import load_dotenv
from fastapi import UploadFile

from knowledge.processor.import_process.main_graph import kb_import_app
from knowledge.processor.import_process.state import create_default_state
from knowledge.services.task_service import TaskService
from knowledge.tools.paths_utils import get_local_base_dir
from knowledge.tools.rustfs_utils import upload_file as upload_to_rustfs_object

load_dotenv()

logger = logging.getLogger(__name__)


class FileImportService:
    """文件导入服务类。

    1. 上传文件落盘到本地归档目录
    2. 同步一份到对象存储（RustFS）
    3. 后台运行导入流程图
    """

    def __init__(self, file_dir: str):
        self.file_dir = file_dir
        os.makedirs(file_dir, exist_ok=True)

    def get_date_dir(self) -> str:
        """按日期分层的归档目录。"""
        return os.path.join(self.file_dir, datetime.now().strftime("%Y%m%d"))

    def save_upload_file(self, file: UploadFile, file_dir: str) -> str:
        """将上传文件写入本地目录，返回其绝对路径。"""
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)
        with open(import_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return import_file_path

    def archive_to_rustfs(self, import_file_path: str, file: UploadFile) -> str:
        """把原始文件归档一份到对象存储。"""
        object_name = (
            f"origin_files/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        )
        return upload_to_rustfs_object(import_file_path, object_name=object_name)

    def process_files(self, files: List[UploadFile]) -> Tuple[List[str], str]:
        """处理多个上传文件，返回 (task_ids, 日期目录)。"""
        date_dir = self.get_date_dir()
        task_ids = []

        for file in files:
            # 1. 生成任务 ID
            task_id = str(uuid.uuid4())
            task_ids.append(task_id)

            # 2. 创建任务状态
            TaskService.create_task_node(task_id, "upload_file")

            # 3. 保存文件
            file_dir = os.path.join(date_dir, task_id)
            import_file_path = self.save_upload_file(file, file_dir)

            # 4. 归档到对象存储
            self.archive_to_rustfs(import_file_path, file)

            # 5. 标记上传完成
            TaskService.complete_task_node(task_id, "upload_file")

        return task_ids, date_dir

    def run_import_task(self, task_id: str, file_dir: str, import_file_path: str):
        """后台运行导入流程图，并逐节点更新任务进度。"""
        try:
            # 1. 更新任务状态
            TaskService.update_task_status(task_id, "processing")

            # 2. 创建初始状态
            initial_state = create_default_state(
                task_id=task_id,
                file_dir=file_dir,
                import_file_path=import_file_path,
            )

            # 3. 运行图并追踪节点完成
            for event in kb_import_app.stream(initial_state):
                for key in event:
                    TaskService.complete_task_node(task_id, key)

            # 4. 更新任务状态
            TaskService.update_task_status(task_id, "completed")

        except Exception as e:
            TaskService.update_task_status(task_id, "failed")
            logger.error(f"[{task_id}] 导入流程失败: {e}", exc_info=True)


_file_import_service = None


def get_file_import_service() -> FileImportService:
    """获取文件导入服务单例。"""
    global _file_import_service
    if _file_import_service is None:
        _file_import_service = FileImportService(get_local_base_dir())
    return _file_import_service
