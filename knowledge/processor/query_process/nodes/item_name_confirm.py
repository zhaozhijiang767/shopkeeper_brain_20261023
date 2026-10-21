from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode


class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点。

    作为流程图入口节点，负责确认/识别用户查询中涉及的商品名称，
    并在已有答案时路由到直接输出。
    """

    name = "item_name_confirm"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """确认商品名称（stub：无状态更新）。

        TODO: 实现商品名称识别与确认逻辑。
        """
        return {}
