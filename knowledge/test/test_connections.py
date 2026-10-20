#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务连接测试脚本
"""
import os
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 服务启动慢（MongoDB 光建存储引擎就要 17s，Milvus 要 40s+），
# docker compose up 完立刻跑测试会连不上。这里统一做等待重试。
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "60"))  # 每个服务最长等待秒数
CONNECT_RETRY_INTERVAL = 2.0                                 # 重试间隔秒数


def _wait_ready(probe, timeout=None):
    """反复执行 probe() 直到不抛异常，返回 probe 的结果；超时抛最后一次异常。

    用于容忍 docker compose up 之后服务还没初始化完的情况。
    """
    timeout = CONNECT_TIMEOUT if timeout is None else timeout
    deadline = time.time() + timeout
    last_error = None
    while True:
        try:
            return probe()
        except Exception as e:
            last_error = e
            if time.time() >= deadline:
                raise
            time.sleep(CONNECT_RETRY_INTERVAL)


def test_milvus():
    """测试 Milvus 连接"""
    print("测试 Milvus 连接...")
    try:
        from pymilvus import connections, utility

        def _connect():
            connections.connect(
                alias="default",
                uri=os.getenv("MILVUS_URL", "http://localhost:19530")
            )
            return utility.get_server_version()

        version = _wait_ready(_connect)
        print(f"  ✓ Milvus 连接成功，版本: {version}")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"  ✗ Milvus 连接失败: {e}")
        return False

def test_neo4j():
    """测试 Neo4j 连接"""
    print("测试 Neo4j 连接...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            print(f"  ✓ Neo4j 连接成功，测试查询返回: {record['num']}")
        driver.close()
        return True
    except Exception as e:
        print(f"  ✗ Neo4j 连接失败: {e}")
        return False

def test_mongodb():
    """测试 MongoDB 连接"""
    print("测试 MongoDB 连接...")
    try:
        from pymongo import MongoClient

        def _connect():
            client = MongoClient(
                os.getenv("MONGO_URL", "mongodb://localhost:27017"),
                serverSelectionTimeoutMS=5000
            )
            client.admin.command('ping')  # 触发实际连接
            return client

        client = _wait_ready(_connect)
        db_names = client.list_database_names()
        print(f"  ✓ MongoDB 连接成功，数据库列表: {db_names}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ MongoDB 连接失败: {e}")
        return False

def test_rustfs():
    """测试 RustFS 连接（S3 兼容对象存储）"""
    print("测试 RustFS 连接...")
    try:
        import socket
        from urllib.parse import urlparse
        from rustfs import RustFileSystem

        endpoint = os.getenv("RUSTFS_ENDPOINT", "localhost:9000")
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
        parsed = urlparse(endpoint)

        access_key = os.getenv("RUSTFS_ACCESS_KEY", "rustfs")
        secret_key = os.getenv("RUSTFS_SECRET_KEY", "rustfs")
        bucket = os.getenv("RUSTFS_BUCKET_NAME", "knowledge-base")
        region = os.getenv("RUSTFS_REGION", "us-east-1")

        # 先做一次 TCP 探测：服务没起来时快速失败，避免卡在 SDK 的 10 次重试（约 26s）上
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=3):
            pass

        def _connect_buckets():
            fs = RustFileSystem(
                endpoint=endpoint,
                access_key_id=access_key,
                secret_access_key=secret_key,
                bucket=bucket,
                region=region,
                allow_http=(parsed.scheme == "http"),
            )
            return fs.ls("", detail=False) or []

        # TCP 通了不代表 S3 就绪，这里做一次带重试的探测
        entries = _wait_ready(_connect_buckets)
        bucket_names = []
        for entry in entries:
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            name = name.rstrip("/").split("/")[-1]
            if name:
                bucket_names.append(name)

        print(f"  ✓ RustFS 连接成功，存储桶: {bucket_names}")
        if bucket not in bucket_names:
            print(f"  ! 目标存储桶 '{bucket}' 尚未创建，首次上传前需要先建桶")
        return True
    except BaseException as e:
        # 注意：rustfs 底层是 Rust 实现，出错时抛的是继承自 BaseException 的
        # PanicException，普通 except Exception 抓不住，进程会直接崩。
        # 因此这里捕获 BaseException，但放行 Ctrl-C / 主动退出。
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"  ✗ RustFS 连接失败: {type(e).__name__}: {e}")
        return False

def main():
    print("=" * 50)
    print("掌柜问数 - 服务连接测试")
    print("=" * 50)

    results = {
        "Milvus": test_milvus(),
        "Neo4j": test_neo4j(),
        "MongoDB": test_mongodb(),
        "RustFS": test_rustfs(),
    }

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    all_passed = True
    for service, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {service}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("所有服务连接正常！")
    else:
        print("存在服务连接失败，请检查配置。")

    return all_passed

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)