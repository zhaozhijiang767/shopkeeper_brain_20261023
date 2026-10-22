"""商品名称确认节点

从用户查询中提取商品名称，通过向量相似度匹配与数据库中已有商品对齐确认。
"""

import os
import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompt import ITEM_NAME_EXTRACT_TEMPLATE


class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点。

    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 写入历史
    """

    name = "item_name_confirm"

    # 对齐阈值
    HIGH_CONFIDENCE_THRESHOLD = 0.63  # 高置信阈值
    MID_CONFIDENCE_THRESHOLD = 0.6    # 中置信阈值
    MAX_OPTIONS = 5                    # 最大候选数

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行商品名称识别"""
        session_id = state["session_id"]
        query = state.get("original_query", "")

        # 1. 获取历史记录
        history = self._get_history(session_id)

        # 1.1 保存用户问题（获取 message_id）
        message_id = self._save_message(
            session_id, "user", query,
            item_names=state.get("item_names", [])
        )

        # 2. LLM 提取商品名称
        extract_res = self._extract_item_names(query, history)
        item_names = extract_res.get("item_names", [])
        rewritten_query = extract_res.get("rewritten_query", query)
        # 3. 向量匹配 + 评分对齐
        align_result = self._match_and_align(item_names) if item_names else {}

        # 4. 更新状态
        state = self._update_state(state, align_result, rewritten_query, history)

        # 5. 写入历史
        self._write_history(state, session_id, rewritten_query, message_id)
        state["history"] = history

        return state

    def _get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取历史会话记录。"""
        from knowledge.tools.mongo_history_utils import get_recent_messages
        try:
            return get_recent_messages(session_id, limit=limit)
        except Exception as e:
            self.logger.warning(f"获取历史记录失败: {e}")
            return []

    def _save_message(
            self, session_id: str, role: str, text: str,
            rewritten_query: str = "", item_names: List[str] = None,
            message_id: str = "",
    ) -> str:
        """保存单条消息到历史记录。"""
        from knowledge.tools.mongo_history_utils import save_chat_message
        try:
            return save_chat_message(
                session_id=session_id,
                role=role,
                text=text,
                rewritten_query=rewritten_query,
                item_names=item_names or [],
                **({"message_id": message_id} if message_id else {}),
            )
        except Exception as e:
            self.logger.warning(f"保存消息失败: {e}")
            return ""

    def _extract_item_names(self, query: str, history: List[Dict]) -> Dict[str, Any]:
        """使用 LLM 从查询和历史中提取商品名称。

        Returns:
            {"item_names": [...], "rewritten_query": "..."}
        """
        from knowledge.tools.llm_utils import get_llm_client

        # 1. 获取 LLM 客户端
        client = get_llm_client(os.getenv("ITEM_MODEL"), response_format=True)

        # 2. 构建历史对话文本
        history_text = "".join(
            f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"
            for msg in history
        )

        # 3. 组装提示词
        prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text, query=query
        )

        try:
            # 4. 调用 LLM
            response = client.invoke([
                SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
                HumanMessage(content=prompt),
            ])

            # 5. 清洗空格
            content = response.content.strip()

            # 6. 清洗代码块围栏
            if content.startswith("```"):
                content = content.split("```")[1].removeprefix("json").strip()

            # 7. 解析 JSON
            result = json.loads(content)
            result.setdefault("item_names", [])
            result.setdefault("rewritten_query", query)
            result["item_names"] = [n.strip() for n in result["item_names"]]
            return result

        except Exception as e:
            self.logger.error(f"LLM 提取商品名称失败: {e}")
            return {"item_names": [], "rewritten_query": query}

    def _match_and_align(self, item_names: List[str]) -> Dict[str, Any]:
        """向量检索 + 评分对齐。

        Returns:
            {"confirmed_item_names": [...], "options": [...]}
        """
        # 1. 向量检索
        query_results = self._vector_search(item_names)

        # 2. 评分对齐
        return self._align_by_score(query_results)

    def _vector_search(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """批量向量检索商品名称。"""
        config = get_config()
        from knowledge.tools.embedding_utils import (
            get_bge_m3_model,
            generate_hybrid_embeddings,
        )
        from knowledge.tools.milvus_utils import (
            get_milvus_client,
            build_hybrid_search_requests,
            execute_hybrid_search,
        )

        # 1. 获取 Milvus 客户端
        client = get_milvus_client()
        if not client:
            self.logger.error("无法连接到 Milvus")
            return []

        # 2. 集合名称
        collection_name = config.item_name_collection
        if not collection_name:
            self.logger.error("未配置商品名集合名称（ITEM_NAME_COLLECTION）")
            return []

        # 3. 获取嵌入模型
        model = get_bge_m3_model()
        if not model:
            self.logger.error("无法加载 BGE-M3 嵌入模型")
            return []

        # 3.1 生成混合向量
        embeddings = generate_hybrid_embeddings(model, item_names)
        if not embeddings:
            self.logger.error("生成混合向量失败")
            return []

        results = []

        # 4. 遍历每个候选商品名
        for i, name in enumerate(item_names):
            try:
                # 4.1 构建混合检索请求
                #      dense/sparse 的度量类型必须与集合索引一致（建库时均为 IP）
                reqs = build_hybrid_search_requests(
                    dense_vector=embeddings["dense"][i],
                    sparse_vector=embeddings["sparse"][i],
                    dense_params={"metric_type": "IP"},
                    sparse_params={"metric_type": "IP"},
                    limit=self.MAX_OPTIONS,
                )

                # 4.2 执行混合检索
                search_res = execute_hybrid_search(
                    milvus_client=client,
                    collection_name=collection_name,
                    search_requests=reqs,
                    ranker_weights=(0.5, 0.5),
                    limit=self.MAX_OPTIONS,
                    norm_score=True,
                    output_fields=["item_name"],
                )

                # 4.3 整理匹配结果
                matches = [
                    {"item_name": hit["entity"]["item_name"], "score": hit["distance"]}
                    for hit in (search_res[0] if search_res else [])
                ]

                results.append({"extracted_name": name, "matches": matches})

            except Exception as e:
                self.logger.error(f"查询商品名称 {name} 失败: {e}")

        return results

    def _align_by_score(self, query_results: List[Dict]) -> Dict[str, Any]:
        """根据评分对齐商品名称。

        规则:
            - score > 0.63 且唯一 → 直接确认
            - score > 0.63 且多条 → 优先取与提取名完全匹配的，否则取最高分
            - 0.6 ≤ score < 0.63 → 作为候选选项
            - score < 0.6 → 忽略
        """
        confirmed: List[str] = []
        options: List[str] = []

        for res in query_results:
            extracted = (res.get("extracted_name") or "").strip()

            # 按分数降序排序
            matches = sorted(
                res.get("matches") or [],
                key=lambda m: m.get("score", 0),
                reverse=True,
            )

            if not matches:
                continue

            # 划分分数区间
            high = [m for m in matches if m["score"] > self.HIGH_CONFIDENCE_THRESHOLD]
            mid = [m for m in matches if m["score"] >= self.MID_CONFIDENCE_THRESHOLD]

            if high:
                # 高置信: 优先精确匹配，否则取最高分
                exact = next(
                    (m for m in high if m["item_name"].strip() == extracted),
                    None
                )
                confirmed.append((exact or high[0])["item_name"])
            elif mid:
                # 中置信: 作为候选选项
                options.extend(m["item_name"] for m in mid[:self.MAX_OPTIONS])

        return {
            "confirmed_item_names": confirmed,
            "options": options[:self.MAX_OPTIONS],
        }

    def _update_state(
            self, state: QueryGraphState, align_result: Dict,
            rewritten_query: str, history: List[Dict],
    ) -> QueryGraphState:
        """根据对齐结果更新 state。"""
        confirmed = align_result.get("confirmed_item_names", [])
        options = align_result.get("options", [])

        if confirmed:
            # 确认成功：回填历史，更新状态
            self._backfill_history_item_names(history, confirmed)
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query

        elif options:
            # 有候选：设置选择提示
            state["answer"] = f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}？"

        else:
            # 无匹配：设置无法识别提示
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"

        return state

    def _backfill_history_item_names(self, history: List[Dict], item_names: List[str]):
        """将确认的商品名称回填到没有商品名的历史记录。"""
        from knowledge.tools.mongo_history_utils import update_message_item_names

        # 1. 获取要更新的消息 ID
        ids_to_update = [
            msg["_id"] for msg in history if not msg.get("item_names")
        ]
        if not ids_to_update:
            return

        # 2. 更新内存中的 history 对象
        for msg in history:
            if not msg.get("item_names"):
                msg["item_names"] = item_names

        # 3. 批量更新 MongoDB
        try:
            update_message_item_names(ids_to_update, item_names)
        except Exception as e:
            self.logger.warning(f"回填历史商品名称失败: {e}")

    def _write_history(
            self, state: QueryGraphState, session_id: str,
            rewritten_query: str, message_id: str,
    ):
        """将本轮对话写入历史（用户问题 + 助手回复）。"""
        query = (state.get("original_query") or "").strip()
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []

        # 更新用户问题消息
        if query:
            self._save_message(
                session_id,
                "user",
                query,
                rewritten_query=rewritten_query,
                item_names=item_names,
                message_id=message_id,
            )

        # 写入助手回复
        if answer:
            self._save_message(
                session_id,
                "assistant",
                answer,
                item_names=item_names,
            )

# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

_node_instance = ItemNameConfirmNode()


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)

if __name__ == "__main__":
    import uuid
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("商品名称确认节点测试")
    print("=" * 60)

    # 1. 准备测试状态
    test_state = {
        "session_id": f"test_{uuid.uuid4().hex[:8]}",
        "original_query": "你们店里那款苏伯尔RS-12数字万用表怎么测电压？",
        "item_names": [],
        "rewritten_query": "",
        "answer": "",
        "history": [],
        "is_stream": False,
    }

    print(f"\n输入状态:")
    print(f"  session_id: {test_state['session_id']}")
    print(f"  original_query: {test_state['original_query']}")
    print(f"  item_names: {test_state['item_names']}")
    print("-" * 60)

    # 2. 执行节点
    try:
        result = node_item_name_confirm(test_state)

        print("\n输出状态:")
        print(f"  item_names: {result.get('item_names')}")
        print(f"  rewritten_query: {result.get('rewritten_query')}")

        if result.get("answer"):
            print(f"\n拦截回复（流程中断）:")
            print(f"  {result.get('answer')}")
        else:
            print(f"\n确认成功，继续检索流程")

        print(f"\n历史记录条数: {len(result.get('history', []))}")

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 测试多轮对话场景
    print("\n" + "=" * 60)
    print("测试多轮对话（代词指代）")
    print("=" * 60)

    # 假设第一轮已经确认了商品名
    test_state_round2 = {
        "session_id": test_state["session_id"],  # 同一个会话
        "original_query": "那它怎么换电池呢？",
        "item_names": [],
        "rewritten_query": "",
        "answer": "",
        "history": [],
        "is_stream": False,
    }

    print(f"\n第二轮输入:")
    print(f"  original_query: {test_state_round2['original_query']}")
    print("-" * 60)

    try:
        result2 = node_item_name_confirm(test_state_round2)

        print("\n第二轮输出:")
        print(f"  item_names: {result2.get('item_names')}")
        print(f"  rewritten_query: {result2.get('rewritten_query')}")

        if result2.get("answer"):
            print(f"\n拦截回复: {result2.get('answer')}")
        else:
            print(f"\n代词已解析，确认成功")

    except Exception as e:
        print(f"\n执行失败: {e}")