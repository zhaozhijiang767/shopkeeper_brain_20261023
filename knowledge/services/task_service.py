"""任务状态服务"""

from typing import Any, Dict

from knowledge.tools.task_utils import (
    add_done_task, add_running_task, get_done_task_list,
    get_running_task_list, get_task_status, update_task_status,
)


class TaskService:
    """任务服务类：把图节点名翻译成前端可读的进度信息。"""

    @staticmethod
    def create_task_node(task_id: str, initial_node: str = "upload_file"):
        """创建新任务（标记首个节点运行中）。"""
        add_running_task(task_id, initial_node)

    @staticmethod
    def complete_task_node(task_id: str, node_name: str):
        """标记节点完成。"""
        add_done_task(task_id, node_name)

    @staticmethod
    def update_task_status(task_id: str, status: str, push_queue: bool = False):
        """更新任务总体状态。"""
        update_task_status(task_id, status, push_queue)

    @staticmethod
    def get_task_info(task_id: str) -> Dict[str, Any]:
        """获取任务完整信息。"""
        return {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        }
