"""MCP 网络搜索节点

通过 MCP 协议调用网络搜索服务获取外部信息。
"""

import asyncio
import os
import json

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    """MCP 网络搜索节点。

    通过 MCP 协议连接到网络搜索服务，
    根据用户查询获取相关的网络搜索结果。
    """

    name = "web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行网络搜索。

        Args:
            state: 查询图状态，需包含 rewritten_query。

        Returns:
            更新后的状态，包含 web_search_docs 搜索结果。
        """
        # Step 1: 获取查询内容
        self.log_step("step_1", "获取查询内容")
        query = state.get("rewritten_query", "")
        docs = []

        if not query:
            self.logger.warning("查询内容为空，跳过网络搜索")
            return {}

        # Step 2-5: 执行 MCP 搜索
        self.log_step("step_2", f"执行 MCP 搜索: {query}")
        try:
            result = asyncio.run(self._mcp_call(query))
            if result:
                pages = json.loads(result.content[0].text).get("pages") or []

                for item in pages:
                    snippet = (item.get("snippet") or "").strip()
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not snippet:
                        continue
                    docs.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    })

                self.log_step("step_3", f"搜索完成，返回 {len(docs)} 条结果")
        except Exception as e:
            self.logger.error(f"MCP 搜索失败: {self._root_error(e)}")

        # Step 6: 返回结果
        if docs:
            return {"web_search_docs": docs}
        return {}

    @staticmethod
    def _root_error(exc: BaseException) -> str:
        """取异常链里最深一层真实原因。

        anyio 的 TaskGroup 会把服务端错误包成 "unhandled errors in a TaskGroup"，
        直接打印外层异常看不到根因。
        """
        seen = []
        cur = exc
        while cur is not None and cur not in seen:
            seen.append(cur)
            subs = getattr(cur, "exceptions", None)
            cur = subs[0] if subs else cur.__cause__ or cur.__context__
        return f"{type(seen[-1]).__name__}: {seen[-1]}"

    async def _mcp_call(self, query: str):
        """调用 MCP 搜索服务。

        Args:
            query: 搜索查询。

        Returns:
            MCP 搜索结果。
        """
        from agents.mcp import MCPServerStreamableHttp

        # Step 2: 创建 MCP 客户端
        # 百炼托管 MCP 只提供 streamable-http（/mcp）：/sse 端点会回 200 但一个字节都不发，
        # 用 MCPServerSse 连它只会得到被 TaskGroup 包住的 "Child exited without calling task_status.started()"
        dashscope_base_url = os.environ.get("MCP_DASHSCOPE_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")

        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": dashscope_base_url,
                "headers": {"Authorization": api_key},
                "timeout": 300,
                "sse_read_timeout": 300
            }
        )

        try:
            # Step 3: 连接 MCP 服务
            await search_mcp.connect()

            # Step 4: 调用搜索工具
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5}
            )
            return result
        finally:
            # Step 6: 清理资源
            await search_mcp.cleanup()


# 兼容原有调用方式
_node_instance = WebSearchMcpNode()


def node_web_search_mcp(state: QueryGraphState) -> QueryGraphState:
    """MCP 网络搜索节点入口函数（兼容原有调用方式）。"""
    return _node_instance(state)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    # 1. 加载环境变量
    load_dotenv()

    # 2. 初始化日志
    try:
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("开始测试: MCP 网络搜索节点 (WebSearchMcpNode)")
    print("=" * 60)

    # 3. 构造模拟的初始状态
    test_query = "数字万用表怎么测电压？"

    initial_state = {
        "original_query": test_query,
        "rewritten_query": test_query
    }

    print("【输入状态】:")
    print(json.dumps(initial_state, ensure_ascii=False, indent=2))
    print("-" * 60)

    try:
        # 4. 执行网络搜索节点
        final_state = node_web_search_mcp(initial_state)

        # 5. 打印搜索结果
        docs = final_state.get("web_search_docs", [])

        if not docs:
            print("\n搜索执行完成，但未返回任何结果。")
        else:
            print(f"\n执行完成！共搜索到 {len(docs)} 条相关内容:\n")

            for i, doc in enumerate(docs, 1):
                title = doc.get('title', '无标题')
                url = doc.get('url', '无链接')
                snippet = doc.get('snippet', '')

                print(f"[{i}] 标题: {title}")
                print(f"    链接: {url}")
                print(f"    摘要: {snippet[:100]}...")
                print("-" * 60)

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()