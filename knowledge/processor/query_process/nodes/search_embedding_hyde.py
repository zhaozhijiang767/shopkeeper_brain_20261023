"""HyDE 向量搜索节点

使用 Hypothetical Document Embedding 技术：
先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量。
"""

import os
from typing import List, Optional

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.tools.item_name_normalize import build_item_key
from knowledge.tools.milvus_utils import escape_expr_value


class SearchEmbeddingHydeNode(BaseNode):
    """HyDE 向量搜索节点。

    流程: LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """

    name = "search_embedding_hyde"

    # 检索参数
    SEARCH_TOP_K = 10
    RERANK_TOP_K = 5
    RANKER_WEIGHTS = (0.5, 0.5)
    OUTPUT_FIELDS = ["chunk_id", "content", "item_name"]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 HyDE 向量搜索。

        Args:
            state: 需包含 rewritten_query（或 original_query）和 item_names。

        Returns:
            {"hyde_embedding_chunks": [...], "hyde_doc": "..."}
        """

        # 获取查询参数
        query = state.get("rewritten_query") or state.get("original_query", "")
        if not query:
            self.logger.error("未找到用户查询")
            return {}

        item_names = state.get("item_names")

        try:
            # Step 1: 生成假设文档
            self.log_step("step_1", "生成假设性文档")
            hyde_doc = self._generate_hyde_doc(query, item_names)

            # Step 2-4: 拼接后向量化检索
            self.log_step("step_2", "执行混合搜索")
            chunks = self._search(query, hyde_doc, item_names, state)

            # Step 5: 返回结果
            self.log_step("step_3", f"搜索完成，返回 {len(chunks)} 条结果")
            return {"hyde_embedding_chunks": chunks, "hyde_doc": hyde_doc}

        except Exception as e:
            self.logger.error(f"HyDE 搜索失败: {e}")
            return {}

    def _generate_hyde_doc(self, query: str, item_names: Optional[List[str]] = None) -> str:
        """使用 LLM 根据用户查询生成假设性文档。"""
        from knowledge.tools.llm_utils import get_llm_client

        llm = get_llm_client()
        prompt = USER_HYDE_PROMPT_TEMPLATE.format(item_hint=item_names, rewritten_query=query)
        return llm.invoke(prompt).content

    def _search(
            self, query: str, hyde_doc: str,
            item_names: Optional[List[str]] = None,
            state: QueryGraphState = None,
    ) -> List:
        """将查询与假设文档拼接后执行混合检索。"""
        from knowledge.tools.embedding_utils import get_bge_m3_model, generate_hybrid_embeddings
        from knowledge.tools.milvus_utils import (
            get_milvus_client,
            build_hybrid_search_requests,
            execute_hybrid_search,
        )

        config = get_config()

        collection_name = config.chunks_collection

        # 拼接文本
        combined_text = f"{query} {hyde_doc}"

        # 加载 BGE-M3 模型
        model = get_bge_m3_model()
        if not model:
            self.logger.error("无法加载 BGE-M3 嵌入模型，请检查 BGE_M3_PATH 配置")
            return []

        # 向量化
        embeddings = generate_hybrid_embeddings(model, [combined_text])
        if not embeddings:
            self.logger.error("混合向量化失败（返回空向量）")
            return []

        # 构建搜索请求（用 item_key 做标量过滤，吸收空格/全角/大小写差异）
        reqs = build_hybrid_search_requests(
            dense_vector=embeddings["dense"][0],
            sparse_vector=embeddings["sparse"][0],
            dense_params={"metric_type": "IP"},
            sparse_params={"metric_type": "IP"},
            expr=self._build_filter_expr(self._resolve_item_keys(state, item_names)),
            limit=self.SEARCH_TOP_K,
        )

        # 执行混合检索
        res = execute_hybrid_search(
            milvus_client=get_milvus_client(),
            collection_name=collection_name,
            search_requests=reqs,
            ranker_weights=self.RANKER_WEIGHTS,
            norm_score=True,
            limit=self.RERANK_TOP_K,
            output_fields=self.OUTPUT_FIELDS,
        )

        return res[0] if res else []

    @staticmethod
    def _resolve_item_keys(state: Optional[QueryGraphState], item_names: Optional[List[str]]) -> List[str]:
        """取商品匹配键：优先用确认节点带回的 key（库里原值），缺失时按同一函数现算。"""
        keys = [k for k in ((state.get("item_keys") or []) if state else []) if k]
        if keys:
            return keys
        return [build_item_key(n) for n in (item_names or [])]

    @staticmethod
    def _build_filter_expr(item_keys: Optional[List[str]]) -> Optional[str]:
        """将商品匹配键列表转换为 Milvus 过滤表达式。

        使用 item_key 字段（build_item_key 生成），吸收空格、全角半角、大小写等
        纯文本差异；展示仍用 item_name。

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

_node_instance = SearchEmbeddingHydeNode()


def node_search_embedding_hyde(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    # 1. 加载环境变量
    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("HyDE 向量搜索节点测试")
    print("=" * 60)

    # 2. 构造测试状态
    test_state = {
        "session_id": "test_001",
        "rewritten_query": "如何使用万用表测量电压？",
        "original_query": "如何使用万用表测量电压？",
        "item_names": ["RS-12数字万用表"],
    }

    print("\n【输入状态】:")
    print(f"  rewritten_query: {test_state['rewritten_query']}")
    print(f"  item_names: {test_state['item_names']}")
    print("-" * 60)

    # 3. 执行节点
    result = node_search_embedding_hyde(test_state)

    # 4. 打印假设文档
    hyde_doc = result.get("hyde_doc", "")
    print("\n【LLM 生成的假设性文档】:")
    print(f"  {hyde_doc[:200]}...")
    print("-" * 60)

    # 5. 打印检索结果
    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【检索结果】: 共 {len(chunks)} 条")
    print("-" * 60)

    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"[{i}] 商品: {entity.get('item_name', '?')}")
        print(f"    ID: {entity.get('chunk_id', 'N/A')}")
        print(f"    分数: {chunk.get('distance', 0):.4f}")
        print(f"    内容: {entity.get('content', '')[:80]}...")
        print()