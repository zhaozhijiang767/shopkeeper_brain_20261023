from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class WebSearchMcpNode(BaseNode):
    """网页搜索节点。

    通过 MCP / 外部 API 进行网页搜索，结果写入 state['web_search_docs']。
    """

    name = "web_search_mcp"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行网页搜索（stub：无状态更新）。

        TODO: 实现 MCP 网页搜索逻辑。
        """
        return {}