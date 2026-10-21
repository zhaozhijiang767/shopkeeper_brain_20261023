from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class AnswerOutputNode(BaseNode):
    """答案输出节点。

    将最终检索/融合结果生成答案并写入 state['answer']。
    """

    name = "answer_output"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """输出答案（stub：无状态更新）。

        TODO: 实现基于 reranked_docs / prompt 的答案生成逻辑。
        """
        return {}