"""
商品名称识别节点

从文档切片中识别商品/产品名称
"""
import json
import os
import re
from typing import List, Tuple, Optional
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import LLMError, ValidationError

from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.embedding_utils import get_bge_m3_model
from knowledge.tools.milvus_utils import get_milvus_client, escape_expr_value
from knowledge.tools.item_name_normalize import (
    build_item_key,
    clean_item_name,
    is_same_name,
    resolve_item_name,
)
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector
from langchain_core.messages import SystemMessage, HumanMessage


class ItemNameRecognitionNode(BaseNode):
    """
    商品名称识别节点

    处理流程：
    1. 接收输入验证
    2. 从前几个切片构造识别上下文
    3. 调用 LLM 识别商品名称
    4. 回填 item_name 到 state 和 chunks
    5. 生成商品名称的向量
    6. 保存到 Milvus
    """

    name = "item_name_recognition"

    MAX_RECOGNIZE_ATTEMPTS = 2  # 商品名识别最多尝试次数

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """执行商品名称识别"""
        config = get_config()

        # Step 1: 验证输入
        file_title, chunks = self._validate_inputs(state)

        # Step 2: 构造识别上下文
        context = self._build_context(chunks, config.item_name_chunk_k)

        # Step 3: 调用 LLM 识别
        raw_name = self._recognize_item_name(file_title, context, config)

        # Step 3.1: 商品名清洗与已入库商品名归并（型号一致则复用旧名字）
        item_name = resolve_item_name(raw_name)
        if item_name != raw_name.strip():
            self.logger.info(f"商品名归并: {raw_name!r} -> {item_name!r}")

        # Step 4: 回填到 state 和 chunks
        self._backfill_item_name(state, chunks, item_name)

        # Step 5: 生成向量
        dense_vector, sparse_vector = self._generate_vectors(item_name)

        # Step 6: 保存到 Milvus
        self._save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector, config)

        return state

    def _validate_inputs(self, state: ImportGraphState) -> Tuple[str, List[dict]]:
        """验证输入"""
        self.log_step("step_1", "验证输入")

        file_title = state.get("file_title", "") # 文件标题（不含扩展名）
        chunks = state.get("chunks", []) # 文档切片列表

        if not file_title:
            raise ValidationError("file_title 为空", node_name=self.name)

        if not isinstance(chunks, list) or not chunks:
            raise ValidationError("chunks 为空或无效", node_name=self.name)

        self.logger.info(f"文件标题: {file_title}, 切片数: {len(chunks)}")
        return file_title, chunks

    def _build_context(self, chunks: List[dict], k: int, max_chars: int = 2500) -> str:
        """构造识别上下文"""
        self.log_step("step_2", "构造识别上下文")

        parts = []
        total = 0

        for i, chunk in enumerate(chunks[:k]):
            if not isinstance(chunk, dict):
                continue

            title = (chunk.get("title") or "").strip()
            content = (chunk.get("content") or "").strip()

            if not (title or content):
                continue

            # 截断过长内容
            if len(content) > 800:
                content = content[:800] + "..."

            piece = f"【切片{i + 1}】\n标题：{title}\n内容：{content}"
            parts.append(piece)
            total += len(piece)

            if total >= max_chars:
                break

        return "\n\n".join(parts)[:max_chars]

    def _recognize_item_name(self, file_title: str, context: str, config) -> str:
        """调用 LLM 识别商品名称。

        结果无效（空、与文件名相同）时重试；多次仍无效则抛错。
        不再用文件名兜底，避免把文档标题当成商品名入库。
        """
        self.log_step("step_3", "调用 LLM 识别")

        prompt = f"""
请从以下信息中识别出商品名称与型号：
文件名：{file_title}

正文切片（用于辅助识别）：
{context}

要求：
1. 文件名只用于确认型号，不要把文件名里的词直接拼进商品名；
2. 商品名格式优先为「品牌 + 型号 + 品类」，没有品牌时用「型号 + 品类」；
3. 品牌、型号、品类各出现一次，不要重复。正确示例：RS-12 数字万用表；错误示例：万用表RS-12数字万用表；
4. 不要包含“的使用、说明书”等文档字样，不要添加解释；
5. 如果无法识别商品名称，请返回空字符串。
"""

        last_raw = ""
        last_error = None

        for attempt in range(1, self.MAX_RECOGNIZE_ATTEMPTS + 1):
            # 1. 调用 LLM
            try:
                llm = get_llm_client(mode_name=config.item_model, response_format=False)
                resp = llm.invoke([
                    SystemMessage(content="你是商品识别专家，只输出字符串。"),
                    HumanMessage(content=prompt),
                ])
                last_raw = getattr(resp, "content", "").strip()
            except Exception as e:
                last_error = e
                self.logger.error(
                    f"商品名识别第 {attempt}/{self.MAX_RECOGNIZE_ATTEMPTS} 次调用失败: {e}"
                )
                continue

            # 2. 校验结果
            if self._is_valid_item_name(last_raw, file_title):
                self.logger.info(f"识别结果: {last_raw}")
                return last_raw

            self.logger.error(
                f"商品名识别第 {attempt}/{self.MAX_RECOGNIZE_ATTEMPTS} 次结果无效: "
                f"{last_raw!r}（文件标题: {file_title!r}）"
            )

        # 3. 重试后仍失败：明确报错，不再用文件名兜底
        raise LLMError(
            f"商品名识别失败（已尝试 {self.MAX_RECOGNIZE_ATTEMPTS} 次），"
            f"最后一次结果: {last_raw!r}，文件标题: {file_title!r}",
            node_name=self.name,
            cause=last_error,
        )

    @staticmethod
    def _is_valid_item_name(item_name: str, file_title: str) -> bool:
        """判断识别结果能否当商品名：非空、含文字、不能是文件名的复制。"""
        cleaned = clean_item_name(item_name)
        if not cleaned:
            return False
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned):
            return False
        if is_same_name(item_name, file_title):
            return False
        return True

    def _backfill_item_name(self, state: ImportGraphState, chunks: List[dict], item_name: str):
        """回填 item_name 与匹配键 item_key 到 state 和 chunks"""
        self.log_step("step_4", "回填 item_name")

        item_key = build_item_key(item_name)

        state["item_name"] = item_name
        state["item_key"] = item_key

        for chunk in chunks:
            chunk["item_name"] = item_name
            chunk["item_key"] = item_key

        state["chunks"] = chunks

    def _generate_vectors(self, item_name: str) -> Tuple[Optional[List[float]], Optional[dict]]:
        """生成向量"""
        self.log_step("step_5", "生成向量")

        try:
            bge_m3_ef = get_bge_m3_model()
            vectors = bge_m3_ef.encode_documents([item_name])

            if vectors:
                dense_vector = vectors["dense"][0].tolist()

                # 提取稀疏向量
                start_idx = vectors["sparse"].indptr[0]
                end_idx = vectors["sparse"].indptr[1]
                token_ids = vectors["sparse"].indices[start_idx:end_idx].tolist()
                weights = vectors["sparse"].data[start_idx:end_idx].tolist()
                sparse_vector = dict(zip(token_ids, weights))

                self.logger.info("向量生成成功")
                return dense_vector, sparse_vector

        except Exception as e:
            self.logger.warning(f"向量生成失败: {e}")

        return None, None

    def _save_to_milvus(
            self,
            state: ImportGraphState,
            file_title: str,
            item_name: str,
            dense_vector: Optional[List[float]],
            sparse_vector: Optional[dict],
            config
    ):
        """保存到 Milvus"""
        self.log_step("step_6", "保存到 Milvus")

        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整，跳过保存")
            return

        try:
            # 1. 获取 Milvus 客户端
            client = get_milvus_client()

            # 2. 获取集合名字
            collection_name = config.item_name_collection

            # 3. 检查并创建集合
            if not client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(client, collection_name)

            # 4. 查重：同一匹配键只保留一条，避免对齐时返回重复候选
            item_key = build_item_key(item_name)
            exists = client.query(
                collection_name=collection_name,
                filter=f'item_key == "{escape_expr_value(item_key)}"',
                output_fields=["item_name"],
                limit=1,
            )
            if exists:
                self.logger.info(f"商品名已存在，跳过写入: {item_name}")
                state["item_name"] = item_name
                return

            # 5. 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "item_key": item_key,
            }

            # 6. 构建稠密向量
            if dense_vector is not None:
                data["dense_vector"] = dense_vector

            # 7. 构建稀疏向量
            if sparse_vector is not None:
                data["sparse_vector"] = normalize_sparse_vector(sparse_vector)

            # 8. 插入数据
            result = client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已保存到 Milvus，ID: {result['ids'][0]}")

            state["item_name"] = item_name

        except Exception as e:
            self.logger.warning(f"Milvus 保存失败: {e}")

    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合"""
        self.logger.info(f"创建集合: {collection_name}")

        # 1. 定义字段
        schema = client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # 匹配键：与切片集合同名，确认节点把它带回给下游做标量过滤
        schema.add_field(field_name="item_key", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功")


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

# 兼容原有调用方式
node_item_name_recognition = ItemNameRecognitionNode()

if __name__ == '__main__':
    """
    商品名识别节点测试

    测试不同场景下的商品名识别逻辑
    """
    import json
    import os

    from knowledge.processor.import_process.base import setup_logging
    from knowledge.processor.import_process.nodes.item_name_recognition import node_item_name_recognition

    # 1. 开启日志
    setup_logging()

    print("=" * 60)
    print("ItemNameRecognitionNode 节点测试")
    print("=" * 60)

    # -------------------- 测试用例 1: 从 chunks.json 加载 -------------------- #
    print("\n--- 测试用例 1: 从 chunks.json 加载并识别 ---")

    # 获取临时目录
    temp_dir = r"D:\projects\shopkeeper_brain\knowledge\temp_data\input"
    chunk_json_input_path = os.path.join(temp_dir, "chunks.json")

    # 检查文件是否存在
    if os.path.exists(chunk_json_input_path):
        with open(chunk_json_input_path, "r", encoding="utf-8") as f:
            chunk_list = json.load(f)

        # 构建 state 状态
        state = {
            "file_title": "万用表的使用",
            "chunks": chunk_list
        }

        # 调用处理方法
        result = node_item_name_recognition.process(state)

        print(f"\n识别结果:")
        print(f"  item_name: {result.get('item_name', '未识别')}")
        print(f"  chunks 数量: {len(result.get('chunks', []))}")

        # 检查 chunks 是否已回填 item_name
        if result.get("chunks"):
            first_chunk = result["chunks"][0]
            print(f"  首个 chunk 的 item_name: {first_chunk.get('item_name', '未回填')}")

        # 备份结果
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "chunks_item_name.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  已备份到: {output_path}")

    else:
        print(f"    chunks.json 文件不存在: {chunk_json_input_path}")
        print("  请先运行 document_split 节点生成 chunks.json")

    # -------------------- 测试用例 2: 使用模拟数据 -------------------- #
    print("\n\n--- 测试用例 2: 使用模拟数据 ---")

    mock_chunks = [
        {
            "title": "# 福禄克 15B+ 数字万用表",
            "content": "福禄克 15B+ 是一款专业级数字万用表，适用于电子工程师和技术人员。\n\n主要特点：\n- 自动量程\n- 高精度测量\n- 坚固耐用",
            "file_title": "万用表说明书"
        },
        {
            "title": "## 产品规格",
            "content": "直流电压：0.1mV - 600V\n交流电压：0.1mV - 600V\n电阻：0.1Ω - 40MΩ",
            "file_title": "万用表说明书"
        },
        {
            "title": "## 安全须知",
            "content": "使用前请仔细阅读本手册。不要测量超过额定值的电压。",
            "file_title": "万用表说明书"
        }
    ]

    mock_state = {
        "file_title": "万用表说明书",
        "chunks": mock_chunks
    }

    mock_result = node_item_name_recognition.process(mock_state)

    print(f"识别结果:")
    print(f"  item_name: {mock_result.get('item_name', '未识别')}")

    # -------------------- 测试用例 3: 空 chunks -------------------- #
    print("\n\n--- 测试用例 3: 空 chunks (预期抛出异常) ---")

    try:
        empty_state = {
            "file_title": "测试文件",
            "chunks": []
        }
        node_item_name_recognition.process(empty_state)
    except Exception as e:
        print(f"捕获到预期异常: {e}")

    # -------------------- 测试用例 4: 缺少 file_title -------------------- #
    print("\n\n--- 测试用例 4: 缺少 file_title (预期抛出异常) ---")

    try:
        no_title_state = {
            "file_title": "",
            "chunks": mock_chunks
        }
        node_item_name_recognition.process(no_title_state)
    except Exception as e:
        print(f"捕获到预期异常: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)