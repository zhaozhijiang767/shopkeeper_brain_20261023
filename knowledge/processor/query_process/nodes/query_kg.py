from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class QueryKgNode(BaseNode):
    """知识图谱查询节点。

    对查询进行知识图谱检索，结果写入 state['kg_chunks'] / state['kg_triples']。
    """

    name = "query_kg"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行知识图谱查询（stub：无状态更新）。

        TODO: 实现 KG 三元组检索逻辑。
        """
        return {}