"""一次性运维脚本：走 MD 分支重跑导入流程，重建 Milvus + Neo4j 数据。

PDF 分支需要 mineru.exe（在 .venv/Scripts 里，不在 PATH 上），故直接使用已转换好的 .md。
"""
import os

from dotenv import load_dotenv

# 必须显式指定 .env 路径：knowledge/.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from knowledge.processor.import_process.main_graph import run_import

FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "temp_data", "input"))
MD_PATH = os.path.join(FILE_DIR, "万用表RS-12的使用.md")

print(f"文件目录: {FILE_DIR}")
print(f"导入文件: {MD_PATH}")
print("-" * 50)

result = run_import(FILE_DIR, MD_PATH)

print("-" * 50)
print(f"识别商品: {result.get('item_name')}")
print(f"切片数量: {len(result.get('chunks', []))}")
