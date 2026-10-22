"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """查询流程图状态。

    包含整个查询流程中传递的所有数据。

    Attributes:
        session_id: 会话 ID，用于追踪多轮对话。
        message_id: 消息 ID，标识单次查询。
        original_query: 原始用户查询。
        embedding_chunks: 向量检索结果列表。
        hyde_embedding_chunks: HyDE 检索结果列表。
        rrf_chunks: RRF 融合后的切片列表。
        web_search_docs: 网页搜索结果列表。
        reranked_docs: 重排序后的文档列表。
        prompt: 构造的提示词。
        answer: 最终生成的答案。
        item_names: 识别的商品名称列表。
        item_keys: 商品名匹配键列表，与 item_names 一一对应，供下游标量过滤。
        rewritten_query: 重写后的查询。
        history: 历史对话列表。
        is_stream: 是否启用流式输出。
        kg_chunks: 知识图谱相关切片列表。
        kg_triples: 知识图谱三元组列表。
    """

    session_id: str
    message_id: str
    original_query: str
    embedding_chunks: list
    hyde_embedding_chunks: list
    rrf_chunks: list
    web_search_docs: list
    reranked_docs: list
    prompt: str
    answer: str
    item_names: List[str]
    item_keys: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
    kg_chunks: list
    kg_triples: list


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",               # 会话 ID
    "message_id": "",               # 消息 ID
    "original_query": "",           # 原始查询
    "embedding_chunks": [],         # 向量检索结果
    "hyde_embedding_chunks": [],    # HyDE 检索结果
    "rrf_chunks": [],               # RRF 融合后的切片
    "web_search_docs": [],          # 网页搜索结果
    "reranked_docs": [],            # 重排序后的文档
    "prompt": "",                   # 提示词
    "answer": "",                   # 答案
    "item_names": [],               # 商品名称
    "item_keys": [],                # 商品名匹配键（下游标量过滤用）
    "rewritten_query": "",          # 重写查询
    "history": [],                  # 历史对话
    "is_stream": False,             # 是否流式输出
    "kg_chunks": [],                # 知识图谱切片
    "kg_triples": []                # 知识图谱关系
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。

    Examples:
        >>> state = create_default_state(
        ...     session_id="session_001",
        ...     original_query="万用表如何测量电压？"
        ... )
    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本。

    Returns:
        状态副本，避免修改全局默认值。
    """
    return copy.deepcopy(DEFAULT_STATE)


# 兼容旧版变量名
graph_default_state = DEFAULT_STATE