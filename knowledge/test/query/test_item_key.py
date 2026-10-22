"""校验 item_key 匹配键是否可用。

检查两个集合的 schema 是否含 item_key 字段，并用「带空格 / 不带空格」两种商品名
走同一条标量过滤，确认命中完全相同的切片。

Milvus v2.5.5 不支持给已有集合加字段（AddCollectionField 未实现），
所以旧集合必须删掉重导入，本脚本用于确认是否已完成迁移。
"""
import os

from dotenv import load_dotenv

load_dotenv()

from knowledge.tools.item_name_normalize import build_item_key
from knowledge.tools.milvus_utils import get_milvus_client, escape_expr_value

# 同一商品的两种写法，只差一个空格
CASES = ["RS-12 数字万用表", "RS-12数字万用表", "rs－12  数字万用表"]


def check_collection(client, collection_name: str) -> bool:
    """检查集合是否已有 item_key 字段。"""
    if not client.has_collection(collection_name):
        print(f"[{collection_name}] 集合不存在，导入流程跑一遍即可自动建新 schema")
        return False

    fields = [f["name"] for f in client.describe_collection(collection_name)["fields"]]
    if "item_key" not in fields:
        print(f"[{collection_name}] 缺少 item_key 字段，需删除集合后重新导入（旧 schema 无法原地加字段）")
        return False

    print(f"[{collection_name}] 已含 item_key 字段")
    return True


def check_filter(client, collection_name: str, name_field: str) -> bool:
    """验证空格差异不影响过滤命中。"""
    counts = {}
    for raw in CASES:
        key = build_item_key(raw)
        expr = f'item_key in ["{escape_expr_value(key)}"]'
        rows = client.query(
            collection_name=collection_name,
            filter=expr,
            output_fields=[name_field],
            limit=100,
        )
        counts[raw] = len(rows)
        print(f"    {raw!r:28} -> key={key!r:24} 命中 {len(rows)} 条")

    hits = set(counts.values())
    ok = len(hits) == 1 and hits != {0}
    print(f"    {'一致' if ok else '不一致或全部为 0'}")
    return ok


if __name__ == "__main__":
    client = get_milvus_client()
    if not client:
        raise SystemExit("无法连接 Milvus，检查 MILVUS_URL")

    chunks = os.getenv("CHUNKS_COLLECTION", "")
    names = os.getenv("ITEM_NAME_COLLECTION", "")

    print("=" * 70)
    print("item_key 迁移校验")
    print("=" * 70)

    schema_ok = all([
        check_collection(client, chunks),
        check_collection(client, names),
    ])

    filter_ok = False
    if schema_ok:
        print(f"\n[{chunks}] 空格差异过滤一致性：")
        filter_ok = check_filter(client, chunks, "item_name")
        print(f"\n[{names}] 空格差异过滤一致性：")
        filter_ok = check_filter(client, names, "item_name") and filter_ok

    print("\n" + "=" * 70)
    print("结论:", "通过" if (schema_ok and filter_ok) else "未通过，按上面的提示处理")
    print("=" * 70)
