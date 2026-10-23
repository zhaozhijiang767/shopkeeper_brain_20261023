"""RRF 融合排序节点

使用 Reciprocal Rank Fusion 算法融合多路检索结果。
"""

from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config


class RrfNode(BaseNode):
    """RRF 融合排序节点。

    流程: 收集三路检索结果 → 带权重 RRF 融合 → 按得分降序返回
    """

    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        config = get_config()

        # Step 1: 收集三路检索结果
        sources = {
            "embedding": (
                self._extract_entities(state.get("embedding_chunks")),
                1.0
            ),
            "hyde": (
                self._extract_entities(state.get("hyde_embedding_chunks")),
                1.0
            ),
            "kg": (
                self._extract_entities(state.get("kg_chunks")),
                config.rrf_kg_weight
            ),
        }

        self.logger.info(
            f"RRF 输入: {', '.join(f'{k}={len(v[0])}' for k, v in sources.items())}"
        )

        # Step 2-5: 执行 RRF 融合
        source_weights = list(sources.values())
        rrf_results = self._reciprocal_rank_fusion(
            source_weights,
            k=config.rrf_k,
            max_results=config.rrf_max_results,
        )

        # Step 6: 输出结果
        rrf_chunks = [doc for doc, _ in rrf_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        if rrf_results:
            scores = [s for _, s in rrf_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        return {"rrf_chunks": rrf_chunks}

    @staticmethod
    def _reciprocal_rank_fusion(
            source_weights: List[Tuple[List[Dict], float]],
            k: int = 60,
            max_results: int = None,
    ) -> List[Tuple[Dict, float]]:
        """带权重的 RRF 融合。

        公式: score(d) = Σ weight_i / (k + rank_i(d))

        Args:
            source_weights: [(文档列表, 权重), ...]
            k: RRF 常数，值越大则排名差异对得分的影响越平滑。
            max_results: 返回前 N 个，None 表示全部。

        Returns:
            [(文档, 得分), ...] 按得分降序。
        """
        # Step 3: 构建得分映射表
        score_map: Dict[str, float] = {}
        # Step 4: 构建文档映射表
        chunk_map: Dict[str, Dict] = {}

        for rank_list, weight in source_weights:
            for pos, item in enumerate(rank_list, start=1):
                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    continue
                # RRF 公式
                score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight / (k + pos)
                chunk_map.setdefault(chunk_id, item)

        # Step 5: 排序与截断
        merged = sorted(
            [(chunk_map[cid], score) for cid, score in score_map.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return merged[:max_results] if max_results else merged

    @staticmethod
    def _extract_entities(state_list) -> List[Dict[str, Any]]:
        """将上游节点输出统一规整为 entity 字典列表。

        兼容格式：
        - {"entity": {...}, "distance": ...} → 取 entity
        - {..., "chunk_id": ...} → 直接使用
        """
        out: List[Dict[str, Any]] = []
        for doc in (state_list or []):
            if not doc or not hasattr(doc, "get"):
                continue
            out.append(doc.get("entity") or doc)
        return out

# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

_node_instance = RrfNode()


def node_rrf(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)


if __name__ == "__main__":
    import json
    setup_logging()

    print("=" * 60)
    print("开始测试: RRF 融合节点 (RrfNode)")
    print("=" * 60)

    # 模拟三路检索结果
    # chunk_1 命中 3 路（最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4, chunk_5 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
        "kg_chunks": [
            {"chunk_id": "chunk_5", "content": "知识图谱结果#1"},
            {"chunk_id": "chunk_1", "content": "知识图谱结果#2"},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)

    # 执行 RRF 融合
    result = node_rrf(mock_state)

    # 打印结果
    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")