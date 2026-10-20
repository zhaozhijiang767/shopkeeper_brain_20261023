"""
知识图谱构建节点

从切片中提取实体和关系，写入 Neo4j 和 Milvus。

代码结构：
    1. LLM 提示词
    2. 主流程 (process → _process_single_chunk)
    3. LLM 提取
    4. JSON 解析与清洗
    5. Milvus 写入
    6. Neo4j 写入（Cypher 常量 + 原子操作）
"""

import json
import re
from typing import Dict, List, Any, Set, Optional

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.tools.embedding_utils import get_bge_m3_model

from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.item_name_normalize import build_item_key
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient, DataType

from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector


class KnowledgeGraphNode(BaseNode):
    """知识图谱构建节点。

    对每个文本切片执行：LLM 实体/关系提取 → JSON 清洗 → Milvus 写入 → Neo4j 写入。
    """

    name = "knowledge_graph"

    # 实体名最大长度（超过则触发截断）
    MAX_ENTITY_NAME_LENGTH = 20

    # ================================================================== #
    #                       1. LLM 提取提示词                              #
    # ================================================================== #

    SYSTEM_PROMPT = """你是知识图谱信息抽取器。给你一段设备操作手册的文本切片，你必须抽取实体与关系，并只输出一个 JSON 对象（不要输出解释、不要 Markdown）。

## 允许的实体类型（label）
- Device：设备整体（如"万用表""仪表"）
- Part：部件或零件（如"电池后盖""螺母""表笔"）
- Operation：操作/功能名称（如"电池安装""电阻测量"），通常对应章节标题
- Step：操作步骤，name 用"步骤N-动作短语"格式（如"步骤1-断开表笔"），description 存原文
- Warning：警告/注意事项，name 用"警告-核心要点"格式（如"警告-操作前断开电源"），description 存原文
- Condition：前置条件或约束（如"电阻小于30Ω"）
- Tool：工具（如"螺丝刀"）

## 实体命名规则（非常重要）
- name 必须简短，不超过15个字。这是硬性要求。
- 禁止将整句原文作为 name。
- Step 格式：name="步骤N-动作短语"，description="原文完整步骤"
- Warning 格式：name="警告-核心要点"，description="原文完整警告"
- 同名同类型的实体只保留一个，不要重复。

## 允许的关系类型（type）
- HAS_OPERATION：Device → Operation
- HAS_PART：Device → Part
- HAS_STEP：Operation → Step
- USES_TOOL：Step → Tool
- HAS_WARNING：Operation/Step → Warning
- NEXT_STEP：Step → Step（按步骤顺序串联）
- AFFECTS：Step → Part（该步骤操作了哪个部件）
- REQUIRES：Step/Operation → Condition

## 抽取原则
- 只抽取文本中明确出现或可直接对应的实体与关系，禁止臆造。
- 步骤编号(1/2/3)时：每条作为 Step，并按顺序生成 NEXT_STEP 关系链。
- 关系的 head 和 tail 必须使用实体的 name 值（简短名），不要用 description。
- 如果无法判断某个关系，不要输出该关系。

## 输出 JSON Schema
{
  "entities": [
    {"name": "简短名称", "label": "类型", "description": "可选，原文内容或补充说明"}
  ],
  "relations": [
    {"head": "头实体name", "tail": "尾实体name", "type": "关系类型"}
  ]
}
"""

    # ================================================================== #
    #                        2. 主流程                                     #
    # ================================================================== #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """执行知识图谱构建。"""
        chunks = state.get("chunks", [])
        if not chunks:
            self.logger.info("chunks 为空，跳过知识图谱构建")
            return state

        self.log_step("start", f"开始处理 {len(chunks)} 个切片")

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content", "")
            chunk_id = str(chunk.get("chunk_id", f"temp_{i}"))
            item_name = chunk.get("item_name") or state.get("item_name", "")

            if not content or not item_name:
                continue

            self.logger.debug(f"处理切片 {i + 1}/{len(chunks)}: {chunk_id}")
            self._process_single_chunk(content, chunk_id, item_name)

        self.log_step("end", "知识图谱构建完成")
        return state

    def _process_single_chunk(self, content: str, chunk_id: str, item_name: str):
        """处理单个切片：提取 → 清洗 → 写入。"""
        config = get_config()

        # 提取
        raw_response = self._llm_extract(content)
        if not raw_response:
            return

        # 清洗
        graph_data = self._parse_and_clean(raw_response)
        if not graph_data.get("entities"):
            return

        self.logger.info(
            f"切片 {chunk_id}: "
            f"提取到 {len(graph_data['entities'])} 个实体, "
            f"{len(graph_data['relations'])} 条关系"
        )

        # 写入
        self._save_entities_to_milvus(
            graph_data.get("entities", []), chunk_id, content, item_name, config,
        )
        self._save_graph_to_neo4j(graph_data, chunk_id, item_name, config)

    # ================================================================== #
    #                      3. LLM 提取                                    #
    # ================================================================== #

    def _llm_extract(self, content: str) -> str:
        """调用 LLM 提取实体和关系。"""
        try:
            # 开启 JSON 模式：让模型侧保证输出是合法 JSON，从源头减少解析失败
            llm = get_llm_client(response_format=True)
            response = llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"请处理以下文本切片：\n\n{content}"),
            ])
            return (response.content or "").strip()
        except Exception as e:
            self.logger.warning(f"LLM 提取失败: {e}")
            return ""

    # ================================================================== #
    #                   4. JSON 解析与清洗                                 #
    # ================================================================== #

    # 关系类型白名单
    ALLOWED_RELATION_TYPES = {
        "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
        "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
        "MENTIONED_IN", "RELATED_TO",
    }

    def _parse_and_clean(self, raw_text: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 并执行清洗。"""
        if not raw_text:
            return {"entities": [], "relations": []}

        # 去除 Markdown 代码围栏
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
        # 丢弃 JSON 前后的解释性文字，只保留最外层对象
        cleaned_text = self._locate_json_object(cleaned_text)

        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            # 日志里直接给出出错位置附近的原文，便于定位是引号、逗号还是截断问题
            near = cleaned_text[max(0, e.pos - 60): e.pos + 60]
            self.logger.warning(
                f"JSON 解析失败: {e}, 错误位置附近原文: {near!r}"
            )
            data = self._repair_json(cleaned_text)

        if not isinstance(data, dict):
            self.logger.warning(
                f"JSON 无法解析为对象，已跳过该切片，原文前300字: {raw_text[:300]}"
            )
            return {"entities": [], "relations": []}

        cleaned_entities = self._clean_entities(data.get("entities", []))
        valid_names = {e["name"] for e in cleaned_entities}
        cleaned_relations = self._clean_relations(data.get("relations", []), valid_names)

        return {"entities": cleaned_entities, "relations": cleaned_relations}

    # ---------- JSON 修复 ----------

    @staticmethod
    def _locate_json_object(text: str) -> str:
        """截取文本中最外层的 {...}，去掉模型额外输出的说明文字。"""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return text
        return text[start:end + 1]

    def _repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """修复 LLM 常见的 JSON 语法问题（缺逗号、引号未转义、尾逗号、截断）。

        使用 json_repair 尽力抢救，修复后仍不是对象则返回 None。
        """
        try:
            from json_repair import repair_json
        except ImportError:
            self.logger.debug("未安装 json_repair，跳过 JSON 修复")
            return None

        try:
            repaired = repair_json(text)
            data = json.loads(repaired)
        except Exception as e:
            self.logger.warning(f"JSON 修复仍失败: {e}")
            return None

        if not isinstance(data, dict):
            return None

        self.logger.info("JSON 已通过 json_repair 修复")
        return data

    # ---------- 实体清洗 ----------

    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """清洗实体：过滤无效项、截断过长名称、去重。"""
        seen: Set[tuple] = set()
        cleaned: List[Dict] = []

        for entity in entities:
            name = str(entity.get("name", "")).strip()
            label = str(entity.get("label", "")).strip()
            description = str(entity.get("description", "")).strip()

            if not name or not label:
                continue

            dedup_key = (name, label)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            result = {"name": name, "label": label}
            if description:
                result["description"] = description
            cleaned.append(result)

        return cleaned

    # ---------- 关系清洗 ----------

    def _clean_relations(
        self,
        relations: List[Dict],
        valid_entity_names: Set[str],
    ) -> List[Dict]:
        """清洗关系：修正字段、白名单校验、过滤悬空引用。"""
        cleaned: List[Dict] = []

        for rel in relations:
            head = str(rel.get("head", "")).strip()
            tail = str(rel.get("tail", "")).strip()
            if not head or not tail:
                continue

            rel_type = str(rel.get("type") or rel.get("label") or "RELATED_TO").strip()
            if rel_type not in self.ALLOWED_RELATION_TYPES:
                rel_type = "RELATED_TO"

            if head not in valid_entity_names or tail not in valid_entity_names:
                self.logger.debug(f"悬空关系已跳过: {head} -[{rel_type}]-> {tail}")
                continue

            cleaned.append({"head": head, "tail": tail, "type": rel_type})

        return cleaned

    # ================================================================== #
    #                     5. Milvus 写入                                  #
    # ================================================================== #

    def _save_entities_to_milvus(
        self,
        entities: List[Dict],
        chunk_id: str,
        content: str,
        item_name: str,
        config,
    ):
        """将实体向量化并写入 Milvus（稠密 + 稀疏双向量）。"""
        if not entities or not config.entity_name_collection:
            return

        dedup_map = self._dedup_entities_by_name(entities)
        if not dedup_map:
            return

        try:
            bge_ef = get_bge_m3_model()
            milvus_client = MilvusClient(uri=config.milvus_url)
            collection_name = config.entity_name_collection

            self._ensure_entity_collection(milvus_client, collection_name)

            names = list(dedup_map.keys())

            vectors = bge_ef.encode_documents(names)
            insert_data = self._build_milvus_records(
                names, vectors, chunk_id, content, item_name,
            )
            if insert_data:
                milvus_client.insert(collection_name=collection_name, data=insert_data)
                milvus_client.load_collection(collection_name=collection_name)
                self.logger.debug(f"写入 {len(insert_data)} 个实体到 Milvus")

        except Exception as e:
            self.logger.warning(f"Milvus 写入失败: {e}")

    @staticmethod
    def _dedup_entities_by_name(entities: List[Dict]) -> Dict[str, Dict]:
        """按 name 去重，合并同名不同类型的 label。"""
        dedup: Dict[str, Dict] = {}
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            label = str(entity.get("label", "")).strip()
            description = str(entity.get("description", "")).strip()

            if name not in dedup:
                dedup[name] = {"labels": set(), "description": description}
            if label:
                dedup[name]["labels"].add(label)
        return dedup

    @staticmethod
    def _ensure_entity_collection(client: MilvusClient, collection_name: str):
        """确保实体集合存在，不存在则创建完整 schema 和索引。"""
        if client.has_collection(collection_name):
            return

        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field(field_name="pk",              datatype=DataType.INT64,
                         is_primary=True, auto_id=True)
        schema.add_field(field_name="entity_name",     datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="dense_vector",    datatype=DataType.FLOAT_VECTOR,
                         dim=1024)
        schema.add_field(field_name="sparse_vector",   datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="source_chunk_id", datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="context",         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="item_name",       datatype=DataType.VARCHAR,
                         max_length=65535)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_milvus_records(
        names: List[str],
        vectors: Dict[str, Any],
        chunk_id: str,
        content: str,
        item_name: str,
    ) -> List[Dict]:
        """组装 Milvus 插入记录（稠密 + 稀疏双向量）。"""

        dense_list = vectors.get("dense", [])
        sparse_raw = vectors.get("sparse")

        records = []
        for idx, name in enumerate(names):
            if idx >= len(dense_list):
                break

            dense_vector = dense_list[idx]
            if hasattr(dense_vector, "tolist"):
                dense_vector = dense_vector.tolist()

            record = {
                "entity_name": name,
                "dense_vector": dense_vector,
                "source_chunk_id": chunk_id,
                "context": content[:200],
                "item_name": item_name,
                # 动态字段（schema 未声明但 enable_dynamic_field=True），便于下游按 item_key 过滤
                "item_key": build_item_key(item_name),
            }

            if sparse_raw is not None:
                start, end = sparse_raw.indptr[idx], sparse_raw.indptr[idx + 1]
                indices = sparse_raw.indices[start:end].tolist()
                data = sparse_raw.data[start:end].tolist()
                sparse_vector = {k: v for k, v in zip(indices, data)}
                record["sparse_vector"] = normalize_sparse_vector(sparse_vector)

            records.append(record)

        return records

    # ================================================================== #
    #                      6. Neo4j 写入                                  #
    # ================================================================== #

    # ---------- Cypher 常量 ----------
    #
    # 图内等值匹配只用 item_key（build_item_key 算出的匹配键），item_name 只作展示：
    # 空格、全角半角、大小写这类纯文本差异会让 item_name 等值匹配漏掉节点，
    # 与 Milvus 侧的过滤口径保持一致。

    CYPHER_MERGE_CHUNK = """
        MERGE (c:Chunk {id: $chunk_id, item_key: $item_key})
        ON CREATE SET c.item_name = $item_name
    """

    CYPHER_MERGE_ENTITY = """
        MERGE (n:Entity {name: $name, item_key: $item_key})
        ON CREATE SET
            n.item_name       = $item_name,
            n.source_chunk_id = $chunk_id,
            n.description     = $description,
            n.types           = CASE
                                    WHEN $label = "" THEN []
                                    ELSE [$label]
                                END
        ON MATCH SET
            n.item_name   = coalesce(n.item_name, $item_name),
            n.description = CASE
                                WHEN $description <> "" THEN $description
                                ELSE coalesce(n.description, "")
                            END,
            n.types       = CASE
                                WHEN $label = ""                       THEN coalesce(n.types, [])
                                WHEN $label IN coalesce(n.types, [])   THEN n.types
                                ELSE coalesce(n.types, []) + $label
                            END
    """

    CYPHER_LINK_ENTITY_TO_CHUNK = """
        MATCH (n:Entity {name: $name, item_key: $item_key})
        MATCH (c:Chunk  {id: $chunk_id, item_key: $item_key})
        MERGE (n)-[:MENTIONED_IN]->(c)
    """

    CYPHER_MERGE_RELATION_TEMPLATE = """
        MATCH (h:Entity {{name: $head, item_key: $item_key}})
        MATCH (t:Entity {{name: $tail, item_key: $item_key}})
        MERGE (h)-[:{rel_type}]->(t)
    """

    # ---------- 入口 ----------

    def _save_graph_to_neo4j(self, graph_data, chunk_id, item_name, config):
        """将图数据保存到 Neo4j。"""
        entities = graph_data.get("entities", [])
        relations = graph_data.get("relations", [])

        if not entities:
            return

        item_key = build_item_key(item_name)

        try:
            from knowledge.tools.neo4j_utils import get_neo4j_driver
            driver = get_neo4j_driver()

            with driver.session(database=config.neo4j_database) as session:
                session.execute_write(
                    self._write_graph_in_tx,
                    entities, relations, chunk_id, item_name, item_key,
                )

            self.logger.debug(
                f"写入 {len(entities)} 个实体, {len(relations)} 条关系到 Neo4j"
            )
        except Exception as e:
            self.logger.warning(f"Neo4j 写入失败: {e}")

    # ---------- 事务总控 ----------

    @staticmethod
    def _write_graph_in_tx(tx, entities, relations, chunk_id, item_name, item_key):
        """在单个事务内完成所有写入。"""
        cls = KnowledgeGraphNode

        cls._tx_merge_chunk(tx, chunk_id, item_name, item_key)

        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            label = str(entity.get("label", "")).strip()
            description = str(entity.get("description", "")).strip()

            cls._tx_merge_entity(tx, name, label, description, chunk_id, item_name, item_key)
            cls._tx_link_entity_to_chunk(tx, name, chunk_id, item_key)

        for rel in relations:
            head = str(rel.get("head", "")).strip()
            tail = str(rel.get("tail", "")).strip()
            if not head or not tail:
                continue
            rel_type = str(rel.get("type", "RELATED_TO")).strip() or "RELATED_TO"
            cls._tx_merge_relation(tx, head, tail, rel_type, item_key)

    # ---------- 原子操作 ----------

    @staticmethod
    def _tx_merge_chunk(tx, chunk_id, item_name, item_key):
        tx.run(KnowledgeGraphNode.CYPHER_MERGE_CHUNK,
               chunk_id=chunk_id, item_name=item_name, item_key=item_key)

    @staticmethod
    def _tx_merge_entity(tx, name, label, description, chunk_id, item_name, item_key):
        tx.run(KnowledgeGraphNode.CYPHER_MERGE_ENTITY,
               name=name, label=label, description=description,
               chunk_id=chunk_id, item_name=item_name, item_key=item_key)

    @staticmethod
    def _tx_link_entity_to_chunk(tx, name, chunk_id, item_key):
        tx.run(KnowledgeGraphNode.CYPHER_LINK_ENTITY_TO_CHUNK,
               name=name, chunk_id=chunk_id, item_key=item_key)

    @staticmethod
    def _tx_merge_relation(tx, head, tail, rel_type, item_key):
        if rel_type not in KnowledgeGraphNode.ALLOWED_RELATION_TYPES:
            rel_type = "RELATED_TO"
        cypher = KnowledgeGraphNode.CYPHER_MERGE_RELATION_TEMPLATE.format(
            rel_type=rel_type
        )
        tx.run(cypher, head=head, tail=tail, item_key=item_key)


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

node_knowledge_graph = KnowledgeGraphNode()


def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM 提取 → 解析清洗全流程。"""
    print("=== 开始测试知识图谱构建流程 ===\n")

    mock_state = {
        "chunks": [
            {
                "content": """# 电池安装
警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
1. 把表笔与仪表断开。
2. 用螺丝刀拧开电池后盖上的螺母。
3. 正确安装电池，正负极应一致。
4. 盖上电池后盖并拧紧螺丝钉。
警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。""",
                "chunk_id": "chunk_test_001",
                "item_name": "万用表",
            }
        ]
    }

    node_knowledge_graph.process(mock_state)
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    setup_logging()
    test_kg_extraction()