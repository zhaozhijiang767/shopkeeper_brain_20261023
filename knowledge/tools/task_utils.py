from typing import Dict, List

from knowledge.tools.sse_utils import push_to_session

# 内存态任务追踪
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, str]] = {}

# 节点名 -> 中文名映射
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "entry": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_img_node": "Markdown图片处理",
    "document_split_node": "文档切分",
    "item_name_recognition": "主体名称识别",
    "beg_embedding_chunks_node": "向量生成",
    "import_milvus_node": "导入向量数据库",
    "knowledge_graph_node": "导入知识图谱",
    "__end__": "处理完成",

    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "item_name_confirm_node": "确认问题产品",
    "answer_output_node": "生成答案",
    "rerank_node": "重排序",
    "rrf_node": "倒排融合",
    "mcp_search_node": "网络搜索",
    "vector_search_node": "切片搜索",
    "hyde_search_node": "切片搜索(假设性文档)",
    "kg_search_node": "查询知识图谱"

}


def _ensure_task(task_id: str):
    _tasks_running_list.setdefault(task_id, [])
    _tasks_done_list.setdefault(task_id, [])
    _tasks_result.setdefault(task_id, {})
    _tasks_status.setdefault(task_id, "")


def add_running_task(task_id: str, node_name: str, push_queue: bool = False):
    """添加正在运行的节点任务"""
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)

    if push_queue:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, push_queue: bool = False):
    """添加已完成的节点任务"""
    _ensure_task(task_id)

    # 从 running 中移除
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 追加到 done
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if push_queue:
        task_push_queue(task_id)


def update_task_status(task_id: str, status_name: str, push_queue: bool = False):
    """更新任务状态"""
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    """推送任务进度到 SSE"""
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def _to_cn(node_name: str) -> str:
    # 1. 从节点映射字典中获取中文名，若未配置则直接返回原英文名
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def get_running_task_list(task_id: str) -> List[str]:
    # 1. 获取指定任务运行中的节点列表，并通过列表推导式统一转换为中文展示名返回
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    # 1. 获取指定任务已完成的节点列表，并通过列表推导式统一转换为中文展示名返回
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    """
    根据任务ID 获取任务状态
    :param task_id:
    :return:
    """
    # 1. 安全获取指定任务的总体运行状态，若不存在则返回空字符串
    return _tasks_status.get(task_id, "")
