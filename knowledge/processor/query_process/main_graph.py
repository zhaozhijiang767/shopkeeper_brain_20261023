"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode
from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.query_kg import QueryKgNode
from knowledge.processor.query_process.nodes.rerank import RerankNode
from knowledge.processor.query_process.nodes.rrf import RrfNode
from knowledge.processor.query_process.nodes.search_embedding import SearchEmbeddingNode
from knowledge.processor.query_process.nodes.search_embedding_hyde import SearchEmbeddingHydeNode
from knowledge.processor.query_process.nodes.web_search_mcp import WebSearchMcpNode
from knowledge.processor.query_process.state import (
    QueryGraphState,
    create_default_state
)

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> StateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    流程结构::

        item_name_confirm
              │
              ├── (有答案) ──────────────────────────> answer_output
              │                                              │
              └── (无答案) ──> multi_search ─────┬──────────>│
                                   │             │           │
                         ┌─────────┼─────────────┼───────┐   │
                         │         │             │       │   │
                         v         v             v       v   │
                   embedding  hyde_embedding  query_kg  web  │
                         │         │             │       │   │
                         └─────────┴─────────────┴───────┘   │
                                       │                     │
                                       v                     │
                                     join                    │
                                       │                     │
                                       v                     │
                                      rrf                    │
                                       │                     │
                                       v                     │
                                    rerank                   │
                                       │                     │
                                       v                     │
                               answer_output <───────────────┘
                                       │
                                       v
                                      END
    """
    # 1. 创建状态图
    workflow = StateGraph(QueryGraphState)

    # 2. 实例化节点
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: x,  # 多路搜索分发（虚节点）
        "search_embedding": SearchEmbeddingNode(),
        "search_embedding_hyde": SearchEmbeddingHydeNode(),
        "query_kg": QueryKgNode(),
        "web_search_mcp": WebSearchMcpNode(),
        "join": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode(),
    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",  # 无答案，继续检索
            True: "answer_output"  # 有答案，直接输出
        }
    )

    # 6. 多路搜索分发（并行执行）
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "query_kg")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 7. 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("query_kg", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 8. 顺序边
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    # 9. 编译并返回
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()


def run_query(
        query: str,
        session_id: str = "",
        item_names: list = None,
        is_stream: bool = False
) -> dict:
    """便捷函数：运行查询流程。

    Args:
        query: 用户查询文本。
        session_id: 会话 ID。
        item_names: 已知的商品名称列表。
        is_stream: 是否启用流式输出。

    Returns:
        最终状态字典。
    """
    # 1. 创建初始状态
    initial_state = create_default_state(
        session_id=session_id or "default",
        original_query=query,
        item_names=item_names or [],
        is_stream=is_stream,
    )

    final_state = None

    # 2. 运行图的工作流节点
    for event in query_app.stream(initial_state):
        for key, value in event.items():
            print(f"节点: {key}")
            final_state = value

    # 3. 返回最终状态
    return final_state or initial_state

# ==================== 命令行入口 ====================

if __name__ == "__main__":
    import sys
    from knowledge.processor.query_process.base import setup_logging

    # 1. 配置日志
    setup_logging()

    print("=" * 60)
    print("知识库查询流程测试")
    print("=" * 60)

    # 2. 准备测试查询
    # 支持命令行参数或使用默认值
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "万用表如何测量电压？"

    print(f"查询: {test_query}")
    print("-" * 60)

    # 3. 运行查询流程
    try:
        result = run_query(
            query=test_query,
            session_id="test_001",
            item_names=[],        # 初始不指定商品名
            is_stream=False       # 非流式输出
        )

        print("-" * 60)
        print("流程完成!")
        print("-" * 60)

        # 4. 输出结果摘要
        print(f"识别商品: {result.get('item_names', [])}")
        print(f"检索切片数: {len(result.get('embedding_chunks', []))}")
        print(f"HyDE 切片数: {len(result.get('hyde_embedding_chunks', []))}")
        print(f"KG 切片数: {len(result.get('kg_chunks', []))}")
        print(f"RRF 融合数: {len(result.get('rrf_chunks', []))}")
        print(f"Rerank 结果数: {len(result.get('reranked_docs', []))}")
        print("-" * 60)

        # 5. 输出答案（截取前 500 字符）
        answer = result.get('answer', 'N/A')
        print("答案:")
        print(answer[:500] + "..." if len(answer) > 500 else answer)

    except Exception as e:
        print(f"流程执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 6. 打印图结构（ASCII 可视化）
    print("-" * 60)
    print("图结构:")
    query_app.get_graph().print_ascii()