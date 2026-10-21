from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class SearchEmbeddingNode(BaseNode):
    """向量检索节点。

    对原始查询进行 embedding 向量检索，结果写入 state['embedding_chunks']。
    """

    name = "search_embedding"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行向量检索（stub：无状态更新）。

        TODO: 实现 embedding 向量检索逻辑。
        """
        return {}
