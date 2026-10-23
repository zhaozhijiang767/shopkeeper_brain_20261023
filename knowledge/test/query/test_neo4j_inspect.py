"""临时诊断脚本：直连 Neo4j 查看实体/切片/关系数据是否真的写进去了。"""
import os

from dotenv import load_dotenv

load_dotenv()

print("NEO4J_URI    =", os.getenv("NEO4J_URI"))
print("NEO4J_DB     =", os.getenv("NEO4J_DATABASE"))
print("NEO4J_USER   =", os.getenv("NEO4J_USERNAME"))
print("PSET_PWD     =", "yes" if os.getenv("NEO4J_PASSWORD") else "no")
print("-" * 50)

from knowledge.tools.neo4j_utils import get_neo4j_driver

driver = get_neo4j_driver()
if driver is None:
    print("!! Neo4j driver 连接/认证失败，检查 URI 与密码")
    raise SystemExit(1)

db = os.getenv("NEO4J_DATABASE") or "neo4j"
print(f"[{db}] 已连接，开始查询 ...\n")

with driver.session(database=db) as s:

    def show(title, cypher, **params):
        print(f"=== {title} ===")
        try:
            rows = s.run(cypher, **params).data()
        except Exception as e:
            print(f"  查询异常: {e}")
            print()
            return
        if not rows:
            print("  (空 - 没查到数据)")
        for r in rows:
            print("  ", r)
        print()

    show("Entity 节点", "MATCH (e:Entity) RETURN e.name AS name, e.item_name AS item_name, coalesce(e.types,[]) AS types")
    show("Chunk 节点", "MATCH (c:Chunk) RETURN c.id AS id, c.item_name AS item_name")
    show("关系类型统计", "MATCH (a)-[r]->(b) RETURN type(r) AS rel, count(*) AS cnt")
    show("名称含『电压』的实体", "MATCH (e:Entity) WHERE e.name CONTAINS '电压' RETURN e.name AS name, e.item_name AS item_name")
    show("名称含『测试』的实体", "MATCH (e:Entity) WHERE e.name CONTAINS '测试' RETURN e.name AS name, e.item_name AS item_name")
    show("以 Entity 为起点的部分关系示例",
         "MATCH (e:Entity)-[r]->(t) RETURN e.name AS head, type(r) AS rel, t.name AS tail LIMIT 20")

print("done")
