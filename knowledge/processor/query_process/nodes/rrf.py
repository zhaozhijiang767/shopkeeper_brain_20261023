from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class RrfNode(BaseNode):
    """RRF 融合节点。

    将多路检索结果（embedding、hyde、kg、web）用 Reciprocal Rank Fusion 融合，
    结果写入 state['rrf_chunks']。
    """

    name = "rrf"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行 RRF 融合（stub：无状态更新）。

        TODO: 实现 RRF 多路结果融合逻辑。
        """
        return {}