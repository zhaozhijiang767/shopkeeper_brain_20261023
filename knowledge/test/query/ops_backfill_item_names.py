"""一次性运维脚本：用 kb_chunks 里已有的 item_name 反补 kb_item_names。

背景：kb_item_names 被清空（或导入时该节点跳过）后，切片仍在但商品名识别不到，
所有查询会在 item_name_confirm 短路返回"无法识别产品名称"。
重导文档会让 import_milvus / knowledge_graph 再写一遍（这两处不删旧数据），
所以只补缺失的商品名行。

用法：
    python ops_backfill_item_names.py          # 预览要补什么，不写
    python ops_backfill_item_names.py --apply  # 实际写入
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from knowledge.processor.query_process.config import get_config
from knowledge.tools.embedding_utils import get_bge_m3_model
from knowledge.tools.item_name_normalize import build_item_key
from knowledge.tools.milvus_utils import get_milvus_client
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector


def _distinct_chunks(client, collection):
    """取切片表里出现过的 (item_name, item_key, file_title) 组合。"""
    rows = client.query(
        collection_name=collection,
        filter='item_key != ""',
        limit=16384,
        output_fields=["item_name", "item_key", "file_title"],
    )
    seen = {}
    for r in rows:
        seen.setdefault(r["item_key"], r)
    return list(seen.values())


def _existing_keys(client, collection):
    rows = client.query(
        collection_name=collection,
        filter='item_key != ""',
        limit=16384,
        output_fields=["item_key"],
    )
    return {r["item_key"] for r in rows}


def _vectors_of(item_name):
    model = get_bge_m3_model()
    vectors = model.encode_documents([item_name])
    dense = vectors["dense"][0].tolist()
    start, end = vectors["sparse"].indptr[0], vectors["sparse"].indptr[1]
    token_ids = vectors["sparse"].indices[start:end].tolist()
    weights = vectors["sparse"].data[start:end].tolist()
    return dense, normalize_sparse_vector(dict(zip(token_ids, weights)))


def main():
    apply = "--apply" in sys.argv
    config = get_config()
    client = get_milvus_client()

    chunk_rows = _distinct_chunks(client, config.chunks_collection)
    have = _existing_keys(client, config.item_name_collection)
    missing = [r for r in chunk_rows if r["item_key"] not in have]

    print(f"切片表商品名 {len(chunk_rows)} 个，名称表已有 {len(have)} 个，待补 {len(missing)} 个")
    for r in missing:
        print(f"  - {r['item_name']}  (key={r['item_key']}, file={r['file_title']})")

    if not missing:
        return
    if not apply:
        print("预览模式，未写入。加 --apply 执行。")
        return

    for r in missing:
        item_key = build_item_key(r["item_name"])
        dense, sparse = _vectors_of(r["item_name"])
        res = client.insert(
            collection_name=config.item_name_collection,
            data=[{
                "file_title": r["file_title"],
                "item_name": r["item_name"],
                "item_key": item_key,
                "dense_vector": dense,
                "sparse_vector": sparse,
            }],
        )
        print(f"已写入 {r['item_name']} -> pk={res['ids'][0]}")

    print("已写入，名称表现在有：", [
        x["item_name"] for x in client.query(
            collection_name=config.item_name_collection,
            filter='item_key != ""',
            output_fields=["item_name"],
            limit=64,
        )
    ])


if __name__ == "__main__":
    main()
