"""一次性运维脚本：Milvus 旧数据备份 / 按主键删除。

用法：
    python ops_backup_and_clean_milvus.py backup   # 只读备份，写 knowledge/temp_data/backup_milvus_rs12_<ts>.json
    python ops_backup_and_clean_milvus.py delete   # 按备份文件里记录的主键删除（不碰新写入的数据）
    python ops_backup_and_clean_milvus.py status   # 打印各表行数

恢复：把备份里的 rows 逐条 client.insert 回对应 collection
（sparse_vector 的 key 在 JSON 里被序列化成字符串，插回前需 int() 转回）。
"""
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from pymilvus import DataType, MilvusClient

ITEM_NAME = "RS-12 数字万用表"
COLLECTIONS = ["kb_chunks", "kb_graph_entity_names", "kb_item_names"]
BACKUP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "temp_data"))
LATEST = os.path.join(BACKUP_DIR, "backup_milvus_rs12_latest.json")

client = MilvusClient(uri=os.getenv("MILVUS_URI") or "http://localhost:19530")


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)


def _pk_field(col: str) -> tuple:
    """返回 (主键字段名, 是否字符串类型) —— 决定删除表达式里要不要加引号。

    注意 DataType 是 IntEnum，str() 只会得到数字，必须用枚举本身比较。
    """
    for f in client.describe_collection(col)["fields"]:
        if f.get("is_primary"):
            return f["name"], f["type"] == DataType.VARCHAR
    raise KeyError(f"{col} 没有主键字段")


def counts(col: str) -> tuple:
    mine = client.query(collection_name=col, filter=f'item_name == "{ITEM_NAME}"',
                        output_fields=["count(*)"])[0]["count(*)"]
    total = client.query(collection_name=col, filter=f'item_name != ""',
                         output_fields=["count(*)"])[0]["count(*)"]
    return mine, total


def backup():
    data = {"item_name": ITEM_NAME, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "collections": {}}
    for col in COLLECTIONS:
        pk, pk_is_str = _pk_field(col)
        fields = [f["name"] for f in client.describe_collection(col)["fields"]]
        rows = client.query(collection_name=col, filter=f'item_name == "{ITEM_NAME}"',
                            limit=10000, output_fields=fields)
        data["collections"][col] = {"pk_field": pk, "pk_is_str": pk_is_str, "fields": fields, "rows": rows}
        print(f"[backup] {col}: {len(rows)} 行 (pk={pk}, 字符串主键={pk_is_str})")
    path = os.path.join(BACKUP_DIR, f"backup_milvus_rs12_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, default=_json_default)
    with open(LATEST, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, default=_json_default)
    print(f"[backup] 已写入 {path}\n[backup] 大小={os.path.getsize(path) / 1024:.1f} KB")


def delete():
    with open(LATEST, encoding="utf-8") as fp:
        data = json.load(fp)
    print(f"使用备份: created_at={data['created_at']}")
    for col, blob in data["collections"].items():
        pks = [r[blob["pk_field"]] for r in blob["rows"]]
        if not pks:
            print(f"[delete] {col}: 备份为空，跳过")
            continue
        if blob["pk_is_str"]:
            listed = ", ".join(f'"{p}"' for p in pks)
        else:
            listed = ", ".join(str(int(p)) for p in pks)
        res = client.delete(collection_name=col, filter=f'{blob["pk_field"]} in [{listed}]')
        print(f"[delete] {col}: 按主键删除 {len(pks)} 条, delete_count={res.get('delete_count', res)}")
    status()


def status():
    print("-" * 50)
    for col in COLLECTIONS:
        mine, total = counts(col)
        print(f"[status] {col}: 旧数据匹配行={mine} 全表={total}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"backup": backup, "delete": delete, "status": status}[mode]()
