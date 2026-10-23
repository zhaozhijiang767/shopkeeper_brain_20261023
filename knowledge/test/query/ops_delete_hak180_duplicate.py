"""一次性运维脚本：删除同一份 PDF 被重复导入后写入的那一批数据。

背景：
    hak180使用说明书.pdf 被导入了两次（f327d008 完成、dd8e1735 重复）。
    kb_chunks 的 chunk_id 是 Milvus 自增主键，不做内容去重，
    所以第二次导入会把同一份文档的切片、实体行、Neo4j Chunk 节点全部再写一遍。

判定归属：chunk_id 单调递增 → 保留第一批（小 id），删除 id 大于分界线的新批次。
分界线取重复导入开始前记录的最大 chunk_id。

用法：
    python ops_delete_hak180_duplicate.py                 # 只预览，不写
    python ops_delete_hak180_duplicate.py --apply         # 备份后删除
    python ops_delete_hak180_duplicate.py --boundary 469959628068192600

恢复：备份文件里 rows 逐条 client.insert 回对应 collection
（sparse_vector 的 key 在 JSON 里被序列化成字符串，插回前需 int() 转回）。
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from pymilvus import DataType, MilvusClient

FILE_TITLE = "hak180使用说明书"
ITEM_KEY = "brotherhak180缝纫机"
# 重复导入开始前，kb_chunks 里 hak180 的最大 chunk_id（保留批次与重复批次的分界）
DEFAULT_BOUNDARY = 469959628068192600

CHUNKS = "kb_chunks"
ENTITIES = "kb_graph_entity_names"
NAMES = "kb_item_names"

BACKUP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "temp_data"))

client = MilvusClient(uri=os.getenv("MILVUS_URI") or "http://localhost:19530")


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)


def _pk_field(col: str):
    """返回 (主键字段名, 是否字符串类型) —— 决定删除表达式里要不要加引号。"""
    for f in client.describe_collection(col)["fields"]:
        if f.get("is_primary"):
            return f["name"], f["type"] == DataType.VARCHAR
    raise KeyError(f"{col} 没有主键字段")


def collect(boundary: int) -> dict:
    """按分界线把要删的东西全部挑出来，不动数据。"""
    dup_chunks = client.query(
        collection_name=CHUNKS,
        filter=f'file_title == "{FILE_TITLE}" and chunk_id > {boundary}',
        output_fields=["chunk_id", "content", "item_name", "item_key", "file_title"],
        limit=2000,
    )
    chunk_ids = [int(c["chunk_id"]) for c in dup_chunks]
    if not chunk_ids:
        return {"chunk_ids": [], "dup_chunks": [], "dup_entities": [], "extra_names": []}

    dup_entities = client.query(
        collection_name=ENTITIES,
        filter=f'source_chunk_id in [{", ".join(chr(34) + str(i) + chr(34) for i in chunk_ids)}]',
        output_fields=["pk", "entity_name", "source_chunk_id", "item_name"],
        limit=5000,
    )
    # 商品名集合里同一 item_key 会被写两次：保留最小 pk，其余算重复
    names = client.query(
        collection_name=NAMES,
        filter=f'item_key == "{ITEM_KEY}"',
        output_fields=["pk", "item_name", "item_key", "file_title"],
        limit=100,
    )
    names_sorted = sorted(names, key=lambda n: int(n["pk"]))
    extra_names = names_sorted[1:]

    return {
        "chunk_ids": chunk_ids,
        "dup_chunks": dup_chunks,
        "dup_entities": dup_entities,
        "extra_names": extra_names,
    }


def preview(boundary: int) -> dict:
    info = collect(boundary)
    print(f"保留批次：{FILE_TITLE} 中 chunk_id <= {boundary} 的切片")
    print(f"重复批次切片 : {len(info['dup_chunks'])} 条")
    print(f"重复批次实体行: {len(info['dup_entities'])} 条")
    print(f"重复商品名行  : {len(info['extra_names'])} 条")
    if not info["chunk_ids"]:
        print("\n分界线之上没有任何切片 —— 重复导入还没写进 Milvus，或者分界线填错了。本次不删任何东西。")
        return info
    print(f"\n将删除 chunk_id 范围: {min(info['chunk_ids'])} ~ {max(info['chunk_ids'])}")
    print("Neo4j: 同 chunk_id 的 Chunk 节点会连同关系一起删（Entity 节点按 name+item_key MERGE，两批共用，不删）。")
    return info


def backup(info: dict, boundary: int) -> str:
    path = os.path.join(
        BACKUP_DIR, f"backup_hak180_dup_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "file_title": FILE_TITLE,
                "boundary": boundary,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": {
                    CHUNKS: info["dup_chunks"],
                    ENTITIES: info["dup_entities"],
                    NAMES: info["extra_names"],
                },
            },
            f,
            ensure_ascii=False,
            default=_json_default,
        )
    return path


def delete(info: dict, boundary: int):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    path = backup(info, boundary)
    print(f"已备份到 {path}")

    for col, key, rows in (
        (CHUNKS, "chunk_id", info["dup_chunks"]),
        (ENTITIES, "pk", info["dup_entities"]),
        (NAMES, "pk", info["extra_names"]),
    ):
        pk, pk_is_str = _pk_field(col)
        assert pk == key, f"{col} 主键是 {pk}，脚本预期 {key}"
        ids = [r[pk] for r in rows]
        if not ids:
            print(f"{col}: 无重复行，跳过")
            continue
        expr = ", ".join(f'"{i}"' if pk_is_str else str(i) for i in ids)
        client.delete(collection_name=col, filter=f'{pk} in [{expr}]')
        print(f"{col}: 已删除 {len(ids)} 行")

    _delete_neo4j_chunks(info["chunk_ids"])


def _delete_neo4j_chunks(chunk_ids):
    try:
        from knowledge.tools.neo4j_utils import get_neo4j_driver
    except Exception as e:
        print(f"跳过 Neo4j 清理（驱动不可用）: {e}")
        return
    driver = get_neo4j_driver()
    if driver is None:
        print("跳过 Neo4j 清理：拿不到 driver")
        return
    with driver.session() as session:
        res = session.run(
            "UNWIND $ids AS id "
            "MATCH (c:Chunk {id: toString(id)}) "
            "DETACH DELETE c RETURN count(c) AS deleted",
            ids=[str(i) for i in chunk_ids],
        )
        deleted = res.single()["deleted"]
    driver.close()
    print(f"Neo4j: 已删除 {deleted} 个 Chunk 节点及其关系")


def status():
    # 各表主键类型不同（kb_chunks.chunk_id 是 Int64，kb_item_names.pk 是 VarChar），
    # 全表计数只能按字段类型写表达式，不能用统一的 `pk > 0`
    for col, total_expr, mine_expr in (
        (CHUNKS, "chunk_id > 0", f'file_title == "{FILE_TITLE}"'),
        (ENTITIES, "pk > 0", 'item_name == "Brother HAK 180 缝纫机"'),
        (NAMES, 'pk != ""', f'item_key == "{ITEM_KEY}"'),
    ):
        total = client.query(collection_name=col, filter=total_expr,
                             output_fields=["count(*)"])[0]["count(*)"]
        mine = client.query(collection_name=col, filter=mine_expr,
                            output_fields=["count(*)"])[0]["count(*)"]
        print(f"{col}: 总计 {total}，hak180 {mine}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只预览）")
    parser.add_argument("--boundary", type=int, default=DEFAULT_BOUNDARY)
    parser.add_argument("--status", action="store_true", help="只打印各表行数")
    args = parser.parse_args()

    if args.status:
        status()
        return

    info = collect(args.boundary)
    preview(args.boundary)
    if not args.apply:
        print("\n（预览模式，未改动任何数据。确认无误后加 --apply）")
        return
    if not info["chunk_ids"]:
        print("\n没有可删除的重复批次，终止。")
        sys.exit(1)
    delete(info, args.boundary)
    print("\n删除后：")
    status()


if __name__ == "__main__":
    main()
