"""重排序节点

使用 Reranker 模型对 RRF 融合结果和网络搜索结果进行重排序，
并通过断崖检测实现动态 TopK 截断。
"""

from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config
from knowledge.tools.reranker_utils import get_reranker_model


class RerankNode(BaseNode):
    """重排序节点。

    流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
    """

    name = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        config = get_config()

        # Step 1: 获取查询文本
        question = state.get("rewritten_query") or state.get("original_query", "")

        # Step 2-3: 合并文档
        doc_items = self._merge_docs(state)

        # Step 4-5: 重排序
        self.log_step("step_1", f"重排序 {len(doc_items)} 篇文档")
        scored_docs = self._rerank(question, doc_items)

        # Step 6: 动态 TopK 截断
        topk_docs = self._cliff_cutoff(scored_docs, config)

        # Step 7: 返回结果
        self.logger.info(f"重排序完成: {len(doc_items)} → {len(topk_docs)}")
        return {"reranked_docs": topk_docs}


    def _merge_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式。"""
        doc_items = []

        # Step 2: 本地 RRF 结果
        for doc in (state.get("rrf_chunks") or []):
            if not isinstance(doc, dict) or not doc.get("content"):
                continue
            doc_items.append(self._make_doc_item(
                text=doc["content"],
                chunk_id=doc.get("chunk_id") or doc.get("id"),
                title=doc.get("title", ""),
                source="local",
            ))

        # Step 3: 网络搜索结果
        for doc in (state.get("web_search_docs") or []):
            text = (doc.get("snippet") or doc.get("content") or "").strip()
            if not text:
                continue
            doc_items.append(self._make_doc_item(
                text=text,
                title=doc.get("title", "").strip(),
                url=doc.get("url", "").strip(),
                source="web",
            ))

        self.logger.info(f"合并文档: {len(doc_items)} 篇")
        return doc_items

    @staticmethod
    def _make_doc_item(
            text: str, source: str = "",
            chunk_id=None, title: str = "", url: str = "",
    ) -> Dict[str, Any]:
        """构建统一的文档结构。"""
        return {
            "text": text, "source": source,
            "chunk_id": chunk_id, "doc_id": chunk_id,
            "title": title, "url": url,
        }

    def _rerank(
            self, question: str, doc_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """计算相关性得分并排序，失败时降级返回原序。"""
        if not doc_items or not question:
            return []

        try:
            # Step 4: 构建 Query-Document 对
            reranker = get_reranker_model()
            pairs = [[question, item["text"]] for item in doc_items]

            # Step 5: 计算得分并排序
            # normalize=True 把 logit 压到 0~1，断崖阈值按该量纲设定
            scores = reranker.compute_score(pairs, normalize=True)
            scored = [
                {**item, "score": float(s)}
                for item, s in zip(doc_items, scores)
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

        except Exception as e:
            """
            为什么需要降级？
                Reranker 模型可能加载失败
                GPU 显存不足
                输入文本过长
                降级后系统仍可工作，只是精度下降
            """
            self.logger.error(f"重排序失败，降级为原序: {e}")
            return [{**item, "score": None} for item in doc_items]

    def _cliff_cutoff(
            self, scored_docs: List[Dict[str, Any]], config,
    ) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not scored_docs:
            return []

        max_topk = min(config.rerank_max_topk, len(scored_docs))
        min_topk = config.rerank_min_topk

        topk = max_topk
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")
            if s1 is None or s2 is None:
                continue

            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)

            if gap >= config.rerank_gap_abs or rel >= config.rerank_gap_ratio:
                topk = i + 1
                self.logger.info(
                    f"断崖检测: 位置 {i + 1}, gap={gap:.4f}, rel={rel:.4f}"
                )
                break

        return scored_docs[:topk]


# 兼容原有调用方式
_node_instance = RerankNode()


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    # 模拟输入状态
    # 包含 2 篇相关文档 + 2 篇不相关文档
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，"
                          "可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感的对地阻值，"
                          "阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    # 执行重排序
    result = node_rerank(mock_state)

    # 打印结果
    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        source = doc['source']
        text = doc['text'][:50]
        print(f"[{i}] score={score_str} | {source:5} | {text}...")

    print("-" * 60)
    print("测试完成")