from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class RerankNode(BaseNode):
    """重排序节点。

    对 RRF 融合后的候选切片进行 rerank 排序，结果写入 state['reranked_docs']。
    """

    name = "rerank"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行重排序（stub：无状态更新）。

        TODO: 实现 rerank 模型调用与排序逻辑。
        """
        return {}