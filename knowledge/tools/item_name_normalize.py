"""商品名称标准化：把 LLM 的随意叫法归并为库中已有的唯一规范名。

不再维护任何本地配置表。规范名的来源只有一个：Milvus 的
ITEM_NAME_COLLECTION（kb_item_names）里已经入库的商品名。

处理流程：
    1. 通用清洗：全角转半角、统一连字符、压缩空格、去尾部标点和文档后缀
    2. 与 kb_item_names 中的已有名字比对：
         - 去掉空格/大小写后相同 → 复用
         - 双方型号一致且相似度 >= 0.63 → 复用旧名字
         - 其余 → 视为新商品，用本次名字
    3. 新商品的名字由调用方（item_name_recognition）写入 kb_item_names

匹配键：所有集合另存一个 item_key = build_item_key(item_name)，
下游的 Milvus 标量过滤只比 item_key，绝不比 item_name —— 否则空格、全角半角、
大小写这类纯文本差异会让等值过滤直接漏掉数据。
"""
import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# 型号：RS-12 / 15B+ / FLUKE-87V / RT-1000 / C21
MODEL_RE = re.compile(r"[A-Za-z]{1,6}[- ]?\d{1,5}[A-Za-z+]*")

# 尾部的文档类后缀，不属于商品名
_DOC_SUFFIX_RE = re.compile(
    r"(?:产品说明书|使用说明书|用户手册|用户指南|操作手册|操作指南|使用手册|使用说明|操作说明|使用指南|产品说明|说明书|的使用|手册)$"
)
# 首尾需要去掉的引号/括号/标点
_TRIM_CHARS = " \t\r\n\"'“”‘’《》<>[]【】()（）"
# NFKC 之后仍需要统一的长横线
_DASH_TABLE = str.maketrans({"–": "-", "—": "-", "﹣": "-"})

# 归并阈值：双方型号一致时使用。
# 没型号的商品只认“去空格/大小写后完全相同”，不做相似度归并：
# 实测同品牌不同品类（苏泊尔电磁炉 vs 苏泊尔电饭煲 ≈ 0.63）与正确匹配
# （≈ 0.65）分数几乎重叠，任何阈值都会误合并。
MODEL_MATCH_THRESHOLD = 0.63


def clean_item_name(raw: str) -> str:
    """对所有商品生效的清洗，不依赖任何表。"""
    if not raw:
        return ""

    # 1. 全角字母/数字/标点 → 半角
    s = unicodedata.normalize("NFKC", raw)

    # 2. 统一长横线为半角连字符
    s = s.translate(_DASH_TABLE)

    # 3. 压缩空白、去首尾引号/括号
    s = re.sub(r"\s+", " ", s).strip().strip(_TRIM_CHARS)

    # 4. 去掉尾部的文档后缀（可能叠加，如“使用说明书手册”）
    while True:
        stripped = _DOC_SUFFIX_RE.sub("", s).strip(_TRIM_CHARS)
        if stripped == s:
            break
        s = stripped

    # 5. 去掉尾部标点
    return re.sub(r"[。，,.;；:：]+$", "", s).strip()


def extract_model(name: str) -> Optional[str]:
    found = MODEL_RE.findall(name or "")
    return found[0].upper().replace(" ", "-") if found else None


def _squash(name: str) -> str:
    """用于比较的规范化形式：去掉所有空白并转小写。"""
    return re.sub(r"\s+", "", name or "").lower()


def is_same_name(a: str, b: str) -> bool:
    """比较两个名字是否相同：忽略空格和大小写。"""
    return _squash(a) == _squash(b)


def build_item_key(name: str) -> str:
    """商品名的匹配键：item_name 的纯函数，两侧用同一函数算出必然字节相等。

    用途是替代 item_name 参与 Milvus 标量等值过滤，从而吸收空格、全角半角、
    大小写、长横线等纯文本差异。展示仍用 item_name，永远不要把本返回值给用户看。

    注意：两个不同商品可能算出同一个 key（如“RS-12 万用表”与“RS12万用表”），
    这是期望行为——它们本就是同一种叫法。key 只在入库时算一次并落库，不做查询期函数计算，
    因为 Milvus 的过滤表达式不支持对字段套函数。
    """
    return _squash(clean_item_name(name))


def _search_similar(cleaned: str, top_k: int = 3) -> list:
    """在 ITEM_NAME_COLLECTION 中检索相似商品名，返回 [{item_name, score}]。"""
    from knowledge.processor.import_process.config import get_config
    from knowledge.tools.embedding_utils import (
        get_bge_m3_model,
        generate_hybrid_embeddings,
    )
    from knowledge.tools.milvus_utils import (
        get_milvus_client,
        build_hybrid_search_requests,
        execute_hybrid_search,
    )

    config = get_config()
    collection = config.item_name_collection
    if not collection:
        return []

    client = get_milvus_client()
    if not client or not client.has_collection(collection):
        return []

    model = get_bge_m3_model()
    if not model:
        return []

    embeddings = generate_hybrid_embeddings(model, [cleaned])
    if not embeddings:
        return []

    reqs = build_hybrid_search_requests(
        dense_vector=embeddings["dense"][0],
        sparse_vector=embeddings["sparse"][0],
        dense_params={"metric_type": "IP"},
        sparse_params={"metric_type": "IP"},
        limit=top_k,
    )
    res = execute_hybrid_search(
        milvus_client=client,
        collection_name=collection,
        search_requests=reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=top_k,
        output_fields=["item_name"],
    )
    if not res:
        return []

    return [
        {
            "item_name": (hit.get("entity") or {}).get("item_name", ""),
            "score": hit.get("distance", 0.0),
        }
        for hit in res[0]
    ]


def resolve_item_name(raw: str) -> str:
    """返回唯一规范名：库里有同款就用旧名字，否则返回清洗后的新名字。

    归并规则：
        - 忽略空格/大小写后完全相同：直接复用
        - 双方都有型号：型号必须一致，且相似度 >= MODEL_MATCH_THRESHOLD
        - 双方都没有型号：只认上面那条“完全相同”
        - 一边有型号一边没有：不归并（泛化条目与具体商品不能互相吞并）
    """
    cleaned = clean_item_name(raw)
    if not cleaned:
        return ""

    try:
        candidates = _search_similar(cleaned)
    except Exception as e:
        logger.warning(f"商品名相似检索失败，按新名字处理: {e}")
        candidates = []

    model = extract_model(cleaned)

    for cand in candidates:
        name = (cand.get("item_name") or "").strip()
        if not name:
            continue

        # 去掉空格/大小写后完全相同（“苏泊尔 电磁炉” vs “苏泊尔电磁炉”）
        if _squash(name) == _squash(cleaned):
            logger.info(f"商品名归并(同名): {cleaned!r} -> {name!r}")
            return name

        cand_model = extract_model(name)

        # 双方都有型号且一致
        if model and cand_model:
            if model != cand_model:
                continue
            score = float(cand.get("score") or 0.0)
            if score >= MODEL_MATCH_THRESHOLD:
                logger.info(f"商品名归并(型号): {cleaned!r} -> {name!r} (score={score:.4f})")
                return name

    logger.info(f"商品名入库（新）: {cleaned!r}")
    return cleaned
