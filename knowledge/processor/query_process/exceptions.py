"""查询流程自定义异常类

统一错误处理，提供更清晰的错误信息。
"""


class QueryProcessError(Exception):
    """查询流程基础异常。

    Attributes:
        node_name: 发生异常的节点名称。
        cause: 原始异常对象。
    """

    def __init__(
        self,
        message: str,
        node_name: str = "",
        cause: Exception = None
    ):
        """初始化异常。

        Args:
            message: 错误信息。
            node_name: 节点名称。
            cause: 原始异常。
        """
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        """格式化异常信息。"""
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class ConfigurationError(QueryProcessError):
    """配置错误：环境变量缺失或配置值无效。"""
    pass


class SearchError(QueryProcessError):
    """搜索错误：向量搜索、混合搜索或网络搜索失败。"""
    pass


class EmbeddingError(QueryProcessError):
    """向量化错误：模型调用失败、向量生成异常。"""
    pass


class LLMError(QueryProcessError):
    """LLM 调用错误：API 调用失败、响应解析失败。"""
    pass


class StorageError(QueryProcessError):
    """存储错误：数据库操作失败。"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误。"""
    pass


class Neo4jError(StorageError):
    """Neo4j 存储错误。"""
    pass


class MongoDBError(StorageError):
    """MongoDB 存储错误（用于历史记录）。"""
    pass


class ValidationError(QueryProcessError):
    """数据验证错误：输入数据不符合预期。"""
    pass


class EntityAlignmentError(QueryProcessError):
    """实体对齐错误：知识图谱实体对齐失败。"""
    pass


class RerankError(QueryProcessError):
    """重排序错误：文档重排序失败。"""
    pass


class ItemNameConfirmError(QueryProcessError):
    """商品名称确认错误：识别或确认失败。"""
    pass