#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RustFS 对象存储测试

按 RustFS 官方文档（boto3 + botocore）编写，覆盖：
    1. 连接        2. 建桶        3. 上传        4. 下载
    5. 列出对象    6. URL 访问    7. 清理

连接要点（来自官方文档）：
    - endpoint_url           指向 RustFS，本地不带 http:// 时会自动补全
    - signature_version      必须 's3v4'，RustFS 支持 v4 签名
    - region_name            RustFS 不校验 region，填任意值即可

凭据从 .env 读取（RUSTFS_ENDPOINT / RUSTFS_ACCESS_KEY / RUSTFS_SECRET_KEY / RUSTFS_BUCKET_NAME）。
"""
import os
import sys
import json
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 测试用本地目录：knowledge/temp_data/rustfs_test
TEST_DIR = Path(__file__).resolve().parent.parent / "temp_data" / "rustfs_test"

# 公开读策略模板：让桶内对象可以通过普通 URL 直接访问（不依赖 presigned URL）
PUBLIC_READ_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::{bucket}/*"],  # {bucket} 会被替换成实际桶名
        }
    ]
}


def _public_read_policy(bucket_name):
    """把模板里的 {bucket} 占位符替换成实际桶名，返回策略 dict"""
    import copy
    policy = copy.deepcopy(PUBLIC_READ_POLICY)
    policy["Statement"][0]["Resource"] = [
        res.format(bucket=bucket_name)
        for res in policy["Statement"][0]["Resource"]
    ]
    return policy


class RustFSTest:
    """RustFS 上传 / 下载 / URL 访问测试类"""

    def __init__(self, bucket_name=None):
        self.endpoint = self._normalize_endpoint(
            os.getenv("RUSTFS_ENDPOINT", "localhost:9000")
        )
        self.access_key = os.getenv("RUSTFS_ACCESS_KEY", "rustfs")
        self.secret_key = os.getenv("RUSTFS_SECRET_KEY", "rustfs")
        self.region = os.getenv("RUSTFS_REGION", "us-east-1")
        self.bucket_name = bucket_name or os.getenv("RUSTFS_BUCKET_NAME", "knowledge-base")

        self.client = self._build_client()
        self.results = []

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_endpoint(endpoint):
        """RustFS 地址补上协议头，'localhost:9000' -> 'http://localhost:9000'"""
        if not endpoint.startswith(("http://", "https://")):
            return f"http://{endpoint}"
        return endpoint

    def _build_client(self):
        """按官方文档构建 boto3 客户端"""
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=self.region,
        )

    def _record(self, name, ok, detail=""):
        self.results.append((name, ok, detail))
        print(f"  {'✓' if ok else '✗'} {name}" + (f": {detail}" if detail else ""))
        return ok

    # ------------------------------------------------------------------ #
    # 1. 连接
    # ------------------------------------------------------------------ #
    def test_connection(self):
        """测试与 RustFS 的连接（列出所有桶）"""
        print("1. 测试连接 ...")
        try:
            resp = self.client.list_buckets()
            buckets = [b["Name"] for b in resp.get("Buckets", [])]
            self._record("连接成功", True, f"已有桶: {buckets}")
            return True
        except Exception as e:
            self._record("连接失败", False, f"{type(e).__name__}: {e}")
            return False

    # ------------------------------------------------------------------ #
    # 2. 建桶
    # ------------------------------------------------------------------ #
    def ensure_bucket(self):
        """确保桶存在，不存在则创建（官方示例的 BucketAlreadyOwnedByYou 处理）"""
        print(f"2. 确保桶存在: {self.bucket_name} ...")
        try:
            self.client.create_bucket(Bucket=self.bucket_name)
            self._record("创建桶", True, self.bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou",
                                               "BucketAlreadyExists"):
                self._record("桶已存在", True, self.bucket_name)
            else:
                self._record("建桶失败", False, str(e))
                return False
        return True

    # ------------------------------------------------------------------ #
    # 3. 上传
    # ------------------------------------------------------------------ #
    def test_upload(self, local_path, object_name=None, content_type=None):
        """上传本地文件，返回对象名"""
        print("3. 测试上传 ...")

        local = Path(local_path).resolve()  # 必须用绝对路径
        if not local.exists():
            self._record("上传失败", False, f"本地文件不存在: {local}")
            return None

        obj = object_name or local.name
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        elif local.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp",
                                      ".pdf", ".txt", ".md", ".json"):
            # rustfs/rustfs 包的 put() 不支持 MIME，boto3 可以通过 ExtraArgs 指定
            extra["ContentType"] = self._guess_content_type(local.suffix.lower())

        try:
            self.client.upload_file(
                str(local),
                self.bucket_name,
                obj,
                ExtraArgs=extra or None,
            )
            self._record("上传成功", True,
                         f"{local.name} -> {self.bucket_name}/{obj}"
                         + (f" (ContentType={extra['ContentType']})" if extra else ""))
            return obj
        except Exception as e:
            self._record("上传失败", False, f"{type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 4. 下载
    # ------------------------------------------------------------------ #
    def test_download(self, object_name, save_as=None):
        """下载对象到本地，返回保存路径"""
        print("4. 测试下载 ...")

        TEST_DIR.mkdir(parents=True, exist_ok=True)
        target = Path(save_as) if save_as else TEST_DIR / f"downloaded_{Path(object_name).name}"
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(self.bucket_name, object_name, str(target.resolve()))
            size = target.stat().st_size
            self._record("下载成功", True, f"保存到 {target} ({size} 字节)")
            return target
        except Exception as e:
            self._record("下载失败", False, f"{type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 5. 列出对象
    # ------------------------------------------------------------------ #
    def test_list(self, prefix=""):
        """列出桶内对象（官方示例的 list_objects_v2）"""
        print("5. 测试列出对象 ...")
        try:
            resp = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            contents = resp.get("Contents", [])
            for obj in contents:
                print(f"    - {obj['Key']} ({obj['Size']} bytes)")
            self._record("列出成功", True, f"共 {len(contents)} 个对象")
            return contents
        except Exception as e:
            self._record("列出失败", False, f"{type(e).__name__}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # 6. URL 访问
    # ------------------------------------------------------------------ #
    def get_public_url(self, object_name):
        """普通 URL：桶设为公开读时可直接访问，无需签名"""
        return f"{self.endpoint}/{self.bucket_name}/{object_name}"

    def get_presigned_url(self, object_name, expires_in=3600):
        """预签名 URL：带鉴权信息，打开即可下载，expires_in 秒后失效。

        桶不是公开读时用这个，一定能访问；官方文档未演示，属 boto3 通用能力。
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": object_name},
            ExpiresIn=expires_in,
        )

    def test_url_access(self, object_name):
        """验证 URL 访问：优先普通 URL，失败则退回预签名 URL"""
        print("6. 测试 URL 访问 ...")

        # 先给桶挂公开读策略，保证普通 URL 能直接访问
        if self.set_public_read_policy():
            time.sleep(2)  # 策略生效有短暂延迟

        import urllib.request
        for kind, url in (("普通 URL", self.get_public_url(object_name)),
                          ("预签名 URL", self.get_presigned_url(object_name))):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = resp.read()
                self._record(f"{kind} 可访问", True,
                             f"HTTP {resp.status}, {len(data)} 字节 -> {url}")
                return url
            except Exception as e:
                self._record(f"{kind} 不可访问", False, f"{type(e).__name__}: {e}")

        print("    ! 两种 URL 都失败。普通 URL 需要桶策略允许公开读；"
              "若项目在容器网络内，请用容器名而非 localhost。")
        return None

    def set_public_read_policy(self):
        """给桶设置公开读策略，使普通 URL 可直接访问"""
        print("6.1 设置桶公开读策略 ...")
        try:
            self.client.put_bucket_policy(Bucket=self.bucket_name, Policy=json.dumps(
                _public_read_policy(self.bucket_name)
            ))
            self._record("策略已设置", True)
            return True
        except Exception as e:
            self._record("策略设置失败", False, f"{type(e).__name__}: {e}")
            return False

    # ------------------------------------------------------------------ #
    # 7. 清理
    # ------------------------------------------------------------------ #
    def cleanup(self, object_name=None):
        """删除测试对象（可选），再删空桶"""
        print("7. 清理测试数据 ...")
        if object_name:
            try:
                self.client.delete_object(Bucket=self.bucket_name, Key=object_name)
                self._record("删除对象", True, object_name)
            except Exception as e:
                self._record("删除对象失败", False, str(e))

    # ------------------------------------------------------------------ #
    # 完整流程
    # ------------------------------------------------------------------ #
    def run(self, local_path=None, keep_bucket=True):
        """按顺序跑完整流程：连接 -> 建桶 -> 上传 -> 下载 -> 列出 -> URL"""
        print("=" * 60)
        print("RustFS 对象存储测试")
        print("=" * 60)
        print(f"端点: {self.endpoint}  桶: {self.bucket_name}")

        # 准备一个测试文件
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        upload_file = Path(local_path).resolve() if local_path else TEST_DIR / "hello.txt"
        if not local_path:
            upload_file.write_text("hello from RustFS test\n", encoding="utf-8")

        print(f"上传文件: {upload_file}\n")

        if not self.test_connection():
            print("! 连接失败，后续步骤跳过。请确认 RustFS 容器已启动。")
            return False
        self.ensure_bucket()

        obj = self.test_upload(upload_file, object_name=f"test/{upload_file.name}")
        if obj:
            self.test_download(obj)
            self.test_list()
            self.test_url_access(obj)

            print(f"\n对象访问地址:")
            print(f"  普通 URL   : {self.get_public_url(obj)}")
            print(f"  预签名 URL : {self.get_presigned_url(obj)[:110]}...")

        # 汇总
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        passed = sum(1 for _, ok, _ in self.results if ok)
        for name, ok, detail in self.results:
            print(f"  {'✓' if ok else '✗'} {name}")
        print("=" * 60)
        print(f"通过 {passed}/{len(self.results)} 项")

        if not keep_bucket:
            # 桶非空时删不掉，需要先清空
            self.cleanup(obj)
            try:
                self.client.delete_bucket(Bucket=self.bucket_name)
                print(f"已删除桶 {self.bucket_name}")
            except Exception as e:
                print(f"删除桶失败（桶可能非空）: {e}")

        return passed == len(self.results)

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _guess_content_type(suffix):
        return {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp", ".pdf": "application/pdf",
            ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
        }.get(suffix, "application/octet-stream")


if __name__ == "__main__":
    # 可直接运行：.venv\Scripts\python.exe test\test_rustfs.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dotenv import load_dotenv as _ld

    _ld()

    tester = RustFSTest()

    # 想测自己的文件就传路径，例如 tester.run(r"D:\pic\photo.jpg")
    ok = tester.run(local_path=r"D:\projects\shopkeeper_brain\knowledge\temp_data\input\1.jpg")
    sys.exit(0 if ok else 1)
