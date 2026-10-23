"""知识图谱查询节点

从用户查询中抽取实体，经 Milvus 对齐后在 Neo4j 中检索相关子图，
最终回填切片文本内容。
"""
import json
from contextlib import contextmanager
from typing import Any, List, Set, Dict, Optional, Tuple, TypedDict, Iterator

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.tools.item_name_normalize import build_item_key
from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.milvus_utils import escape_expr_value


# ================================================================== #
#  Neo4j 返回行的类型别名（仅作类型提示，不参与运行时）              #
# ================================================================== #
class NodeRef(TypedDict):
    """Neo4j 种子节点行。item_key 参与图内匹配，item_name 只用于展示。"""
    name: str
    item_key: str
    item_name: str


class Triple(TypedDict):
    """一跳扩展返回的三元组行。"""
    item_key: str
    item_name: str
    head: str
    rel: str
    tail: str


class QueryKgNode(BaseNode):
    """知识图谱查询节点。"""

    name = "query_kg"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 预处理
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = self._clean_item_names(state.get("item_names"))
        # 图内等值匹配一律用 item_key（与向量检索同口径），item_name 只用于展示与去重
        item_keys = self._resolve_item_keys(state, item_names)
        for name in item_names:
            question = question.replace(name, "")

        # 2. LLM 实体抽取
        self.log_step("step_1", "LLM 抽取实体")
        entities = self._extract_entities(question, get_llm_client(response_format=True))

        # 3. Milvus 实体对齐
        self.log_step("step_2", "Milvus 对齐实体")
        align_result = self._align_entities(entities, item_names, state) if entities else {}
        aligned = align_result.get("aligned_entities", entities)
        self.logger.info(f"对齐后实体: {aligned}")

        # 4. Neo4j 种子节点 + 扩展一跳
        self.log_step("step_3", "Neo4j 种子节点 + 扩展一跳")
        self.logger.info(f"商品匹配键: {item_keys}")
        seed_nodes = self._find_seed_nodes(aligned, item_keys)
        self.logger.info(f"种子节点: {len(seed_nodes)}")

        # 5. 一跳扩展
        triples = self._expand_one_hop(seed_nodes)
        self.logger.info(f"三元组: {len(triples)}")

        # 6. 获取关联切片
        kg_chunk_hits = self._get_chunk_refs(seed_nodes, triples)

        # 7. Milvus 文本回填
        self.log_step("step_4", "整理输出")
        kg_chunks = self._fetch_chunk_texts(kg_chunk_hits)

        # 8. 返回结果
        return {
            "kg_chunks": kg_chunks,
            "kg_triples": self._triples_to_docs(triples),
            "kg_seed_nodes": seed_nodes,
            "kg_entities": entities,
            "kg_aligned_entities": aligned,
            "kg_alignments": align_result.get("alignments", []),
        }

    @staticmethod
    def _clean_item_names(item_names: Any) -> List[str]:
        """清理商品名称：兼容 None/str/list，返回去重列表。"""
        if not item_names:
            return []
        if isinstance(item_names, str):
            return [item_names.strip()] if item_names.strip() else []

        seen: Set[str] = set()
        return [
            s for x in item_names
            if (s := str(x).strip()) and s not in seen and not seen.add(s)
        ]

    def _extract_entities(self, question: str, llm_client) -> List[str]:
        """LLM 抽取实体并解析 JSON。"""
        try:
            # 1. 调用 LLM
            resp = llm_client.invoke(
                [SystemMessage(content=ENTITY_EXTRACT_SYSTEM_PROMPT),
                 HumanMessage(content=f"用户问题：{question}")],
                extra_body={"enable_thinking": False},
            )

            # 2. 解析 JSON
            data = json.loads((resp.content or "").strip())

            # 3. 提取并去重
            entities = list({
                e.strip() for e in data.get("entities", []) if e.strip()
            })

            self.logger.info(f"抽取到 {len(entities)} 个实体: {entities}")
            return entities

        except Exception as e:
            self.logger.error(f"实体抽取失败: {e}")
            return []

    def _align_entities(
            self, entities: List[str],
            item_names: List[str],
            top_k: int = 5,
            state: QueryGraphState = None,
    ) -> Dict[str, Any]:
        """向量检索对齐实体名称。"""
        from knowledge.tools.embedding_utils import generate_hybrid_embeddings
        from knowledge.tools.milvus_utils import (
            get_milvus_client, build_hybrid_search_requests, execute_hybrid_search,
        )

        config = get_config()

        collection = config.entity_name_collection
        client = get_milvus_client()
        min_score = config.kg_entity_align_min_score
        # entity 集合 schema 没有 item_key 字段（只有 item_name），
        # 实体对齐是向量模糊匹配，不做硬过滤，由相似度自然收敛到正确商品
        expr = None

        # 1. 生成混合向量
        from knowledge.tools.embedding_utils import get_bge_m3_model
        model = get_bge_m3_model()
        if model is None:
            self.logger.error("无法加载 BGE-M3 模型")
            return {"aligned_entities": entities, "alignments": []}
        try:
            emb = generate_hybrid_embeddings(model, entities)
        except Exception as e:
            self.logger.error(f"Embedding 生成失败: {e}")
            return {"aligned_entities": entities, "alignments": []}
        if not emb:
            self.logger.error("混合向量化失败（返回空向量）")
            return {"aligned_entities": entities, "alignments": []}

        alignments: List[Dict] = []
        aligned: List[str] = []
        seen: Set[str] = set()

        # 2. 遍历每个实体进行对齐
        for idx, entity in enumerate(entities):
            dense = emb["dense"][idx]
            sparse = emb["sparse"][idx]

            try:
                # 2.1 构建搜索请求
                #     entity 集合的 dense_vector 索引是 COSINE，sparse 是 IP
                reqs = build_hybrid_search_requests(
                    dense_vector=dense,
                    sparse_vector=sparse,
                    dense_params={"metric_type": "COSINE"},
                    sparse_params={"metric_type": "IP"},
                    expr=expr,
                    limit=top_k,
                )

                # 2.2 执行混合搜索
                res = execute_hybrid_search(
                    milvus_client=client,
                    collection_name=collection,
                    search_requests=reqs,
                    norm_score=True,
                    ranker_weights=(0.5, 0.5),
                    output_fields=["entity_name", "item_name"],
                )

                # 2.3 选择最佳命中
                best = self._pick_best_hit(res[0] if res else [], min_score)

                if best:
                    name = best["entity"]["entity_name"]
                    if name not in seen:
                        seen.add(name)
                        aligned.append(name)
                    alignments.append({
                        "original": entity,
                        "aligned": name,
                        "score": best["distance"],
                    })
                else:
                    alignments.append({
                        "original": entity,
                        "aligned": None,
                        "reason": "no_hit"
                    })

            except Exception as e:
                alignments.append({
                    "original": entity,
                    "aligned": None,
                    "reason": f"error:{e}"
                })

        return {"aligned_entities": aligned, "alignments": alignments}

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

        Args:
            item_keys: build_item_key 生成的匹配键列表。

        Returns:
            如 `item_key in ["a", "b"]`；列表为空则返回 None。
        """
        if not item_keys:
            return None
        quoted = ", ".join(f'"{escape_expr_value(v)}"' for v in item_keys)
        return f"item_key in [{quoted}]"

    def _pick_best_hit(self, hits: List[Dict[str, Any]], min_score: Optional[float]) -> Optional[Dict[str, Any]]:
        """从混合检索命中中选出分数最高的，且不低于 min_score。"""
        if not hits:
            return None
        best = max(hits, key=lambda h: h.get("distance", 0.0))
        if min_score is not None and best.get("distance", 0.0) < min_score:
            return None
        return best

    # ------------------------------------------------------------------ #
    #  Neo4j 相关                                                        #
    # ------------------------------------------------------------------ #
    @contextmanager
    def _neo4j_session(self) -> Iterator:
        """获取 Neo4j 会话（自动管理 driver 连接）。"""
        from knowledge.tools.neo4j_utils import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            raise RuntimeError("Neo4j 驱动不可用")

        database = get_config().neo4j_database or "neo4j"
        with driver.session(database=database) as session:
            yield session

    @staticmethod
    def _tx_find_seeds(tx, entity_name: str, item_keys: List[str], limit: int) -> List[Dict[str, Any]]:
        """在 Neo4j 中查找与实体匹配的 :Entity 节点（按 item_key 过滤，不按 item_name）。"""
        cypher = """
        MATCH (e:Entity {name: $name})
        WHERE e.item_key IN $keys
        RETURN e.name AS name, e.item_key AS item_key, e.item_name AS item_name
        LIMIT $limit
        """
        result = tx.run(cypher, name=entity_name, keys=item_keys, limit=limit)
        return [dict(r) for r in result]

    @staticmethod
    def _tx_expand_triples(tx, name: str, item_key: str, limit: int) -> List[Dict[str, Any]]:
        """扩展种子节点的一跳关系三元组（同 item_key 范围内）。"""
        cypher = """
        MATCH (a:Entity {name: $name, item_key: $item_key})-[r]-(b:Entity {item_key: $item_key})
        WHERE a.name <> b.name
        RETURN a.name AS head, type(r) AS rel, b.name AS tail,
               a.item_key AS item_key, a.item_name AS item_name
        LIMIT $limit
        """
        result = tx.run(cypher, name=name, item_key=item_key, limit=limit)
        return [dict(r) for r in result]

    # ------------------------------------------------------------------ #
    #  结果格式化                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _triples_to_docs(triples: List[Triple]) -> List[Dict[str, Any]]:
        """把 Neo4j 三元组转成下游（RRF/Rerank）可消费的文档格式。"""
        docs = []
        for t in triples:
            docs.append({
                "head": t.get("head", ""),
                "rel": t.get("rel", ""),
                "tail": t.get("tail", ""),
                "item_name": t.get("item_name", ""),
                "text": f"{t.get('head','')} --[{t.get('rel','')}]-> {t.get('tail','')}",
            })
        return docs

    def _find_seed_nodes(
            self, entities: List[str], item_keys: List[str],
            per_entity: int = 3, max_total: int = 30,
    ) -> List[NodeRef]:
        """在 Neo4j 中查找种子节点（精确 → 模糊）。"""
        if not entities or not item_keys:
            return []

        try:
            with self._neo4j_session() as session:
                seeds: List[NodeRef] = []
                seen: Set[Tuple[str, str]] = set()

                for name in entities:
                    # 事务读取
                    rows = session.execute_read(
                        self._tx_find_seeds, name, item_keys, per_entity
                    )

                    for s in rows:
                        key = (s["item_key"], s["name"])
                        if key not in seen:
                            seen.add(key)
                            seeds.append(s)

                        if len(seeds) >= max_total:
                            break

                    if len(seeds) >= max_total:
                        break

                return seeds

        except Exception as e:
            self.logger.error(f"Neo4j 种子查询异常: {e}")
            return []

    def _expand_one_hop(
            self, seed_nodes: List[NodeRef],
            per_seed: int = 50, max_total: int = 200,
    ) -> List[Triple]:
        """扩展种子节点的一跳关系。"""
        if not seed_nodes:
            return []

        try:
            with self._neo4j_session() as session:
                triples: List[Triple] = []
                seen: Set[Tuple[str, ...]] = set()

                for s in seed_nodes:
                    rows = session.execute_read(
                        self._tx_expand_triples,
                        s["name"], s["item_key"], per_seed
                    )

                    for tr in rows:
                        key = (tr["item_key"], tr["head"], tr["rel"], tr["tail"])
                        if key not in seen:
                            seen.add(key)
                            triples.append(tr)

                        if len(triples) >= max_total:
                            break

                    if len(triples) >= max_total:
                        break

                return triples

        except Exception as e:
            self.logger.error(f"Neo4j 扩展异常: {e}")
            return []

    def _get_chunk_refs(
            self, seed_nodes: List[NodeRef], triples: List[Triple],
            max_chunks: int = 200,
    ) -> List[Dict[str, Any]]:
        """通过 MENTIONED_IN 关系获取关联切片 ID（带权重排序）。"""

        # 1. 构建带权重的节点列表
        nodes = self._build_weighted_nodes(seed_nodes, triples)
        if not nodes:
            return []

        try:
            with self._neo4j_session() as session:
                # 2. 查询关联切片
                rows = session.run("""
                    UNWIND $nodes AS n
                    MATCH (e:Entity {name: n.name, item_key: n.item_key})
                          -[:MENTIONED_IN]->(c:Chunk {item_key: n.item_key})
                    WITH c, sum(n.w) AS score, count(DISTINCT e) AS cnt
                    RETURN c.id AS chunk_id, c.item_name AS item_name,
                           score, cnt
                    ORDER BY score DESC, cnt DESC, chunk_id ASC
                    LIMIT $limit
                """, nodes=nodes, limit=max_chunks).data()

            return [
                {
                    "id": None,
                    "distance": float(r.get("score", 0)),
                    "entity": {
                        "chunk_id": str(r["chunk_id"]),
                        "item_name": str(r["item_name"])
                    }
                }
                for r in rows
            ]

        except Exception as e:
            self.logger.error(f"切片引用查询异常: {e}")
            return []

    def _fetch_chunk_texts(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从 Milvus 回填切片文本。"""
        from knowledge.tools.milvus_utils import fetch_chunks_by_chunk_ids

        if not hits:
            return []

        config = get_config()
        collection = config.chunks_collection

        # 1. 提取去重 chunk_id
        chunk_ids = list({
            int(str(h["entity"]["chunk_id"]))
            for h in hits
            if h.get("entity", {}).get("chunk_id") is not None
        })

        if not chunk_ids:
            return []

        # 2. 从 Milvus 批量查询
        try:
            rows = fetch_chunks_by_chunk_ids(
                collection_name=collection,
                chunk_ids=chunk_ids,
                output_fields=["chunk_id", "content", "title", "item_name"],
            )
        except Exception as e:
            self.logger.error(f"Milvus 回填异常: {e}")
            rows = []

        # 3. 构建映射表
        row_map = {
            str(r["chunk_id"]): r
            for r in (rows or [])
            if r.get("chunk_id") is not None
        }

        # 4. 按原顺序回填
        result = []
        for h in hits:
            ent = h.get("entity", {})
            row = row_map.get(str(ent.get("chunk_id")))
            if row:
                merged = dict(row)
                # 合并 item_name
                if ent.get("item_name") and not merged.get("item_name"):
                    merged["item_name"] = ent["item_name"]
                result.append(merged)

        self.logger.info(f"回填完成: {len(result)} 条切片")
        return result

    @staticmethod
    def _build_weighted_nodes(
            seed_nodes: List[NodeRef], triples: List[Triple],
            w_seed: float = 2.0, w_neighbor: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """构建带权重的节点列表（种子权重 > 邻居权重）。"""
        weights: Dict[Tuple[str, str], float] = {}

        # 种子节点: 权重 2.0
        for s in seed_nodes or []:
            key = (s["item_key"], s["name"])
            weights[key] = max(weights.get(key, 0), w_seed)

        # 邻居节点（三元组的头尾）: 权重 1.0
        for tr in triples or []:
            it = tr["item_key"]
            for n in (tr["head"], tr["tail"]):
                key = (it, n)
                weights[key] = max(weights.get(key, 0), w_neighbor)

        return [
            {"item_key": it, "name": n, "w": w}
            for (it, n), w in weights.items()
        ]


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

_node_instance = QueryKgNode()


def node_query_kg(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    try:
        from knowledge.processor.query_process.base import setup_logging
        setup_logging()
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("知识图谱查询节点测试")
    print("=" * 60)

    # 构造测试状态
    test_state = {
        "original_query": "RS-12数字万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表怎么测电压？",
        "item_names": ["RS-12数字万用表"]
    }

    print("【输入状态】:")
    print(json.dumps(test_state, ensure_ascii=False, indent=2))
    print("-" * 60)

    try:
        result = node_query_kg(test_state)

        print("\n执行完成！全链路输出如下:\n")

        # 第1步: LLM 实体抽取
        print("[第1步] LLM 原始抽取实体 (kg_entities):")
        print(f"   {result.get('kg_entities', [])}")

        # 第2步: Milvus 对齐
        print("\n[第2步] Milvus 对齐后实体 (kg_aligned_entities):")
        print(f"   {result.get('kg_aligned_entities', [])}")

        # 第3步: Neo4j 种子节点
        print("\n[第3步] Neo4j 命中的种子节点 (kg_seed_nodes):")
        for seed in result.get("kg_seed_nodes", []):
            print(f"   - {seed}")

        # 第4步: 三元组
        triples = result.get("kg_triples", [])
        print(f"\n[第4步] 扩展的一跳知识三元组 (共 {len(triples)} 条):")
        for t in triples[:5]:
            print(f"   - {t}")
        if len(triples) > 5:
            print(f"   - ... (省略其余 {len(triples) - 5} 条)")

        # 第5步: 切片
        chunks = result.get("kg_chunks", [])
        print(f"\n[第5步] 最终召回的切片 (共 {len(chunks)} 条):")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   [{i}] ID: {chunk.get('chunk_id')}")
            print(f"       商品: {chunk.get('item_name')}")
            print(f"       内容: {chunk.get('content', '')[:80]}...")

    except Exception as e:
        print(f"\n执行失败: {e}")
        import traceback
        traceback.print_exc()