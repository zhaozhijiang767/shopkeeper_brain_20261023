"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryConfig:
    """查询流程配置。

    Attributes:
        max_context_chars: 上下文最大字符数限制。
        rerank_max_topk: 重排序动态 TopK 硬上限。
        rerank_min_topk: 重排序最小 TopK。
        rerank_gap_ratio: 重排序断崖阈值（相对）。
        rerank_gap_abs: 重排序断崖阈值（绝对）。
        rrf_k: RRF 算法常数。
        rrf_kg_weight: RRF 融合中知识图谱权重。
        ...
    """

    # ==================== 文本处理配置 ====================
    max_context_chars: int = 12000  # 上下文最大字符数

    # ==================== Rerank 配置 ====================
    rerank_max_topk: int = 10       # 重排序最大返回数
    rerank_min_topk: int = 3        # 重排序最小返回数
    rerank_gap_ratio: float = 0.25  # 断崖检测阈值（相对）
    rerank_gap_abs: float = 0.5     # 断崖检测阈值（绝对）

    # ==================== RRF 配置 ====================
    rrf_k: int = 60                 # RRF 平滑常数
    rrf_kg_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_KG_WEIGHT", "0.7"))
    )
    rrf_max_results: int = 10       # RRF 最大返回结果数

    # ==================== 检索配置 ====================
    embedding_search_limit: int = 10  # 向量搜索返回数量
    hyde_search_limit: int = 5        # HyDE 搜索返回数量

    # ==================== 知识图谱配置 ====================
    kg_entity_align_min_score: Optional[float] = field(
        default_factory=lambda: (
            float(os.getenv("KG_ENTITY_ALIGN_MIN_SCORE"))
            if os.getenv("KG_ENTITY_ALIGN_MIN_SCORE")
            else None
        )
    )
    kg_max_seed_candidates: int = 3   # 每个实体最大种子候选数
    kg_max_total_seeds: int = 30      # 总种子节点上限
    kg_max_triples_per_seed: int = 50 # 每个种子最大三元组数
    kg_max_total_triples: int = 200   # 总三元组上限
    kg_max_total_chunks: int = 200    # 总切片上限

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。"""
        return cls()

    def validate(self, strict: bool = False) -> None:
        """验证配置是否完整。

        Args:
            strict: 是否严格模式，严格模式下缺少配置会抛出异常。
        """
        required_fields = ["milvus_url", "chunks_collection"]
        missing = [k for k in required_fields if not getattr(self, k)]

        if missing:
            msg = f"缺少必要配置: {missing}"
            if strict:
                raise ValueError(msg)
            else:
                print(f"警告: {msg}")


# ==================== 全局单例 ====================
_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。"""
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config


def reset_config() -> None:
    """重置配置（用于测试）。"""
    global _config
    _config = None