"""
Milvus 导入节点

将向量化后的切片数据批量导入 Milvus，自动创建集合和索引（如果不存在）。

代码结构：
    1. 主流程 (process)
    2. 集合创建（schema 构建 + 索引构建 + 创建集合）
    3. 数据插入与 chunk_id 回填
    4. 测试入口
"""

import json
import os
from typing import List, Dict

from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import MilvusError
from knowledge.tools.milvus_utils import get_milvus_client


class ImportMilvusNode(BaseNode):
    """Milvus 导入节点。

    将包含稠密/稀疏向量的切片数据批量导入 Milvus，
    集合不存在时自动创建 schema 和索引。
    """

    name = "import_milvus"

    # ================================================================== #
    #                        1. 主流程                                     #
    # ================================================================== #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """执行 Milvus 导入。

        流程：校验数据 → 连接 Milvus → 创建集合（如需） → 插入数据 → 回填 chunk_id。

        Args:
            state: 图状态，需包含带向量的 chunks 列表。

        Returns:
            更新后的状态（chunks 中包含 chunk_id）。

        Raises:
            MilvusError: 数据无效或 Milvus 操作失败时抛出。
        """
        config = get_config()
        chunks = state.get("chunks", [])

        # 1. 校验
        if not chunks:
            self.logger.warning("chunks 为空，跳过导入")
            return state

        vector_dim = self._get_vector_dim(chunks)
        self.log_step("step_1", f"准备导入 {len(chunks)} 条数据，向量维度: {vector_dim}")

        # 2. 连接 & 操作
        try:
            client = get_milvus_client()
            collection_name = config.chunks_collection

            # 3. 创建集合（如不存在）
            if not client.has_collection(collection_name=collection_name):
                self.log_step("step_2", f"创建集合: {collection_name}")
                self._create_collection(client, collection_name, vector_dim)

            # 4. 插入数据
            self.log_step("step_3", "执行插入")
            self._insert_and_backfill_ids(client, collection_name, chunks)

            # 5. 更新 chunks 状态
            state["chunks"] = chunks

        except MilvusError:
            raise
        except Exception as e:
            raise MilvusError(f"Milvus 操作失败: {e}", node_name=self.name, cause=e)

        return state

    # ================================================================== #
    #                     2. 集合创建                                      #
    # ================================================================== #

    def _create_collection(self, client, collection_name: str, vector_dim: int):
        """创建 Milvus 集合（schema + 索引）。

        Args:
            client: MilvusClient 实例。
            collection_name: 集合名称。
            vector_dim: 稠密向量维度。
        """
        # 1. 构建 schema
        schema = self._build_schema(client, vector_dim)

        # 2. 构建索引
        index_params = self._build_index_params(client)

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

        self.logger.info(f"集合 {collection_name} 创建成功")

    def _build_schema(self, client, vector_dim: int):
        """构建集合 Schema。

        Args:
            client: MilvusClient 实例。
            vector_dim: 稠密向量维度。

        Returns:
            CollectionSchema 对象。
        """

        # 1. 定义 schema
        schema = client.create_schema(enable_dynamic_field=True)

        # 2. 添加主键(自增)
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )

        # 3. 添加标量字段
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # 匹配键：build_item_key(item_name)，下游标量过滤只比它，不比 item_name
        schema.add_field(field_name="item_key", datatype=DataType.VARCHAR, max_length=65535)

        # 4. 添加向量字段
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

        # 5. 返回 schema
        return schema

    def _build_index_params(self, client):
        """构建索引参数。

        Args:
            client: MilvusClient 实例。

        Returns:
            IndexParams 对象。
        """

        # 1. 获取索引对象
        index_params = client.prepare_index_params()

        # 2. 稠密向量添加索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        # 3. 稀疏向量添加索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        return index_params

    # ================================================================== #
    #                  3. 数据插入与 chunk_id 回填                         #
    # ================================================================== #

    @staticmethod
    def _get_vector_dim(chunks: List[Dict]) -> int:
        """从首条 chunk 中获取稠密向量维度。

        Args:
            chunks: 切片数据列表。

        Returns:
            向量维度。

        Raises:
            MilvusError: 首条 chunk 不包含 dense_vector 时抛出。
        """
        dim = len(chunks[0].get("dense_vector", []))
        if dim == 0:
            raise MilvusError("切片数据不包含 dense_vector", node_name="import_milvus")
        return dim

    def _insert_and_backfill_ids(
            self,
            client,
            collection_name: str,
            chunks: List[Dict],
    ):
        """插入数据并将 Milvus 返回的主键回填到每条 chunk。

        Args:
            client: MilvusClient 实例。
            collection_name: 目标集合名称。
            chunks: 切片数据列表（会被原地修改，添加 chunk_id 字段）。
        """
        result = client.insert(collection_name=collection_name, data=chunks)

        insert_count = result.get("insert_count", 0)
        self.logger.info(f"成功插入 {insert_count} 条数据")

        # 回填 chunk_id
        inserted_ids = result.get("ids", [])
        if inserted_ids and len(inserted_ids) == len(chunks):
            for chunk, chunk_id in zip(chunks, inserted_ids):
                chunk["chunk_id"] = str(chunk_id)
        else:
            self.logger.warning(
                f"回填 chunk_id 失败: 返回 {len(inserted_ids)} 个 ID，"
                f"期望 {len(chunks)} 个"
            )

# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

node_import_milvus = ImportMilvusNode()

if __name__ == "__main__":
    """
    独立测试 ImportMilvusNode

    测试流程：
    1. 从上一个节点的输出文件读取状态（带向量的切片数据）
    2. 执行 Milvus 导入
    3. 验证回填的 chunk_id
    4. 将结果保存到临时文件
    """

    setup_logging()

    # ----------------------------------------------------------------
    # Step 1: 配置路径
    # ----------------------------------------------------------------
    temp_dir = r"D:\projects\shopkeeper_brain\knowledge\temp_data\input"

    # 输入：上一个向量化节点处理后的状态
    input_path = os.path.join(temp_dir, "chunks_item_name_vector.json")

    # 输出：导入 Milvus 后的状态（含 chunk_id）
    output_path = os.path.join(temp_dir, "chunks_item_name_vector_ids.json")

    # ----------------------------------------------------------------
    # Step 2: 读取输入数据
    # ----------------------------------------------------------------
    print(f"正在读取输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content.get('chunks', [])
    print(f"读取到 {len(chunks)} 个切片")

    # ----------------------------------------------------------------
    # Step 3: 构建状态并执行处理
    # ----------------------------------------------------------------
    state = {
        "chunks": chunks
    }

    print("\n开始执行 Milvus 导入...")
    result_state = node_import_milvus.process(state)

    # ----------------------------------------------------------------
    # Step 4: 验证回填结果
    # ----------------------------------------------------------------
    output_chunks = result_state.get("chunks", [])
    print(f"\n处理完成，共 {len(output_chunks)} 个切片")

    # 检查 chunk_id 回填情况
    chunks_with_id = sum(1 for c in output_chunks if c.get("chunk_id"))
    chunks_without_id = len(output_chunks) - chunks_with_id

    print(f"\n回填统计:")
    print(f"  - 成功回填 chunk_id: {chunks_with_id} 个")
    print(f"  - 未回填 chunk_id: {chunks_without_id} 个")

    # 打印前 3 个切片的信息
    print("\n前 3 个切片信息:")
    for i, chunk in enumerate(output_chunks[:3]):
        print(f"\n  切片 {i + 1}:")
        print(f"    chunk_id: {chunk.get('chunk_id', '无')}")
        print(f"    title: {chunk.get('title', '')}")
        print(f"    item_name: {chunk.get('item_name', '')}")
        print(f"    content: {chunk.get('content', '')[:50]}...")

    # ----------------------------------------------------------------
    # Step 5: 保存输出文件
    # ----------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"\n已保存到: {output_path}")