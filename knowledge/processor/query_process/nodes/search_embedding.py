"""向量搜索节点

对用户查询进行向量化，在 Milvus 中执行混合搜索（稠密 + 稀疏），返回相关切片。
"""

import os
from typing import List, Optional

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.exceptions import EmbeddingError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.tools.item_name_normalize import build_item_key
from knowledge.tools.milvus_utils import escape_expr_value


class SearchEmbeddingNode(BaseNode):
    """向量搜索节点。

    流程: 查询向量化 → 构建混合搜索请求 → 执行检索 → 返回结果
    """

    name = "search_embedding"

    # 检索参数
    SEARCH_TOP_K = 10        # 每路检索候选数
    RERANK_TOP_K = 5         # 融合后返回数
    RANKER_WEIGHTS = (0.5, 0.5)  # 稠密:稀疏权重
    OUTPUT_FIELDS = ["chunk_id", "content", "item_name"]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行向量搜索。

                Args:
                    state: 需包含 rewritten_query，以及 item_keys（或由 item_names 现算）。

                Returns:
                    {"embedding_chunks": [...]} 搜索结果列表。
                """
        from knowledge.tools.embedding_utils import (
            get_bge_m3_model,
            generate_hybrid_embeddings,
        )
        from knowledge.tools.milvus_utils import (
            get_milvus_client,
            build_hybrid_search_requests,
            execute_hybrid_search,
        )
        config = get_config()

        # 获取查询参数
        query = state.get("rewritten_query", "")
        item_names = state.get("item_names")
        collection_name = config.chunks_collection

        # Step 1: 向量化
        self.log_step("step_1", f"查询向量化: {query}")
        model = get_bge_m3_model()
        if not model:
            raise EmbeddingError(
                "无法加载 BGE-M3 嵌入模型，请检查 BGE_M3_PATH 配置",
                node_name=self.name,
            )

        embeddings = generate_hybrid_embeddings(model, [query])
        if not embeddings:
            raise EmbeddingError("查询向量化失败（返回空向量）", node_name=self.name)

        # Step 2: 构建过滤表达式（用匹配键而非商品名，空格/全角差异不影响等值过滤）
        filter_expr = self._build_filter_expr(self._resolve_item_keys(state, item_names))
        self.logger.debug(f"过滤表达式: {filter_expr}")

        # Step 3: 构建混合搜索请求
        reqs = build_hybrid_search_requests(
            dense_vector=embeddings["dense"][0],
            sparse_vector=embeddings["sparse"][0],
            dense_params={"metric_type": "IP"},
            sparse_params={"metric_type": "IP"},
            expr=filter_expr,
            limit=self.SEARCH_TOP_K,
        )

        # Step 4: 执行混合检索
        self.log_step("step_2", "执行混合搜索")
        res = execute_hybrid_search(
            milvus_client=get_milvus_client(),
            collection_name=collection_name,
            search_requests=reqs,
            ranker_weights=self.RANKER_WEIGHTS,
            norm_score=True,
            limit=self.RERANK_TOP_K,
            output_fields=self.OUTPUT_FIELDS,
        )

        # Step 5: 返回结果
        chunks = res[0] if res else []
        self.log_step("step_3", f"搜索完成，返回 {len(chunks)} 条结果")

        return {"embedding_chunks": chunks}

    @staticmethod
    def _resolve_item_keys(state: QueryGraphState, item_names: Optional[List[str]]) -> List[str]:
        """取商品匹配键：优先用确认节点带回的 key（库里原值），缺失时按同一函数现算。"""
        keys = [k for k in (state.get("item_keys") or []) if k]
        if keys:
            return keys
        return [build_item_key(n) for n in (item_names or [])]

    @staticmethod
    def _build_filter_expr(item_keys: Optional[List[str]]) -> Optional[str]:
        """将商品匹配键列表转换为 Milvus 过滤表达式。

        Args:
            item_keys: build_item_key 生成的匹配键列表。

        Returns:
            如 `item_key in ["a", "b"]`；列表为空则返回 None。
        """
        if not item_keys:
            return None
        quoted = ", ".join(f'"{escape_expr_value(v)}"' for v in item_keys)
        return f"item_key in [{quoted}]"


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

_node_instance = SearchEmbeddingNode()


def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("向量检索节点测试")
    print("=" * 60)

    # 1. 准备测试状态
    test_state = {
        "session_id": "test_001",
        "rewritten_query": "如何使用万用表测量电压？",
        "item_names": ["RS-12 数字万用表"],
        "embedding_chunks": [],
    }

    print(f"\n输入状态:")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print(f"  item_names: {test_state['item_names']}")
    print("-" * 60)

    # 2. 执行节点
    try:
        result = node_search_embedding(test_state)
        chunks = result.get("embedding_chunks", [])

        print(f"\n检索到 {len(chunks)} 条结果:")
        print("-" * 60)

        for i, chunk in enumerate(chunks, 1):
            # 兼容不同的返回格式
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            content = entity.get("content", "")
            item_name = entity.get("item_name", "未知")
            chunk_id = entity.get("chunk_id", "N/A")
            score = chunk.get("distance", 0)

            print(f"[{i}] 商品: {item_name}")
            print(f"    ID: {chunk_id}")
            print(f"    分数: {score:.4f}")
            print(f"    内容: {content[:80]}...")
            print()

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 测试无过滤条件的场景
    print("=" * 60)
    print("测试无商品名过滤")
    print("=" * 60)

    test_state_no_filter = {
        "session_id": "test_002",
        "rewritten_query": "如何测量电压？",
        "item_names": [],  # 无商品名
        "embedding_chunks": [],
    }

    print(f"\n输入状态:")
    print(f"  rewritten_query: {test_state_no_filter['rewritten_query']}")
    print(f"  item_names: {test_state_no_filter['item_names']} (无过滤)")
    print("-" * 60)

    try:
        result2 = node_search_embedding(test_state_no_filter)
        chunks2 = result2.get("embedding_chunks", [])
        print(f"\n检索到 {len(chunks2)} 条结果（无过滤）")

        # 打印前 3 条
        for i, chunk in enumerate(chunks2[:3], 1):
            entity = chunk.get("entity", chunk) if isinstance(chunk, dict) else {}
            print(f"[{i}] {entity.get('item_name', '?')} | score={chunk.get('distance', 0):.4f}")

    except Exception as e:
        print(f"\n执行失败: {e}")
