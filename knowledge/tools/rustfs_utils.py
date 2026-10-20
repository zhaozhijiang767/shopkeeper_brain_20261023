import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# RustFS 配置（支持环境变量覆盖）
RUSTFS_ENDPOINT = os.getenv("RUSTFS_ENDPOINT", "localhost:9000")
RUSTFS_ACCESS_KEY = os.getenv("RUSTFS_ACCESS_KEY", "rustfs")
RUSTFS_SECRET_KEY = os.getenv("RUSTFS_SECRET_KEY", "rustfs")
RUSTFS_BUCKET_NAME = os.getenv("RUSTFS_BUCKET_NAME", "knowledge-base")
RUSTFS_REGION = os.getenv("RUSTFS_REGION", "us-east-1")


def _normalize_endpoint(endpoint):
    """补上协议头，'localhost:9000' -> 'http://localhost:9000'

    否则 boto3 会把 endpoint_url 当成非法值直接报错。
    """
    if not endpoint.startswith(("http://", "https://")):
        return f"http://{endpoint}"
    return endpoint


# 初始化 RustFS 客户端（单例）
try:
    rustfs_client = boto3.client(
        "s3",
        endpoint_url=_normalize_endpoint(RUSTFS_ENDPOINT),
        aws_access_key_id=RUSTFS_ACCESS_KEY,
        aws_secret_access_key=RUSTFS_SECRET_KEY,
        # signature_version 必须是 s3v4，RustFS 唯一支持的签名版本
        config=Config(signature_version="s3v4"),
        region_name=RUSTFS_REGION,  # RustFS 不校验 region，填任意值即可
    )

    # 确保 Bucket 存在
    try:
        rustfs_client.create_bucket(Bucket=RUSTFS_BUCKET_NAME)
        print(f"Bucket {RUSTFS_BUCKET_NAME} created.")
    except ClientError as e:
        # 桶已存在时 RustFS 返回 BucketAlreadyOwnedByYou，属正常情况
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou",
                                           "BucketAlreadyExists"):
            print(f"Bucket {RUSTFS_BUCKET_NAME} already exists.")
        else:
            raise

except Exception as e:
    print(f"RustFS initialization failed: {e}")
    rustfs_client = None


def get_rustfs_client():
    """获取 RustFS 客户端单例"""
    return rustfs_client


def upload_file(local_path, object_name=None, content_type=None):
    """上传本地文件，返回对象名（不含桶名）

    Args:
        local_path:   本地文件路径（Windows 原生绝对路径）
        object_name:  对象名，可带路径前缀，如 "images/photo.jpg"；默认用文件名
        content_type: MIME 类型，如 "image/jpeg"；不传则按扩展名自动推断
    """
    if rustfs_client is None:
        raise RuntimeError("RustFS 客户端未初始化，请检查 .env 与 RustFS 服务状态")

    from pathlib import Path
    local = Path(local_path).resolve()  # 必须是绝对路径
    if not local.exists():
        raise FileNotFoundError(f"本地文件不存在: {local}")

    key = object_name or local.name
    ctype = content_type or _guess_content_type(local.suffix.lower())

    rustfs_client.upload_file(str(local), RUSTFS_BUCKET_NAME, key,
                              ExtraArgs={"ContentType": ctype})
    return key


def download_file(object_name, save_path):
    """下载对象到本地，返回保存路径（Path 对象）"""
    if rustfs_client is None:
        raise RuntimeError("RustFS 客户端未初始化，请检查 .env 与 RustFS 服务状态")

    from pathlib import Path
    target = Path(save_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    rustfs_client.download_file(RUSTFS_BUCKET_NAME, object_name, str(target))
    return target


def get_object_url(object_name, expires_in=3600):
    """获取对象访问 URL，默认返回预签名 URL

    预签名 URL 自带鉴权，无需桶公开读，expires_in 秒后失效（默认 1 小时）。
    若桶已设公开读策略，用 get_public_url() 得到不限时效的普通 URL。
    """
    if rustfs_client is None:
        raise RuntimeError("RustFS 客户端未初始化，请检查 .env 与 RustFS 服务状态")

    return rustfs_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": RUSTFS_BUCKET_NAME, "Key": object_name},
        ExpiresIn=expires_in,
    )


def get_public_url(object_name):
    """普通 URL，需桶已设置公开读策略才能直接访问"""
    return f"{_normalize_endpoint(RUSTFS_ENDPOINT)}/{RUSTFS_BUCKET_NAME}/{object_name}"


def list_objects(prefix=""):
    """列出桶内对象，返回 [(key, size), ...]"""
    if rustfs_client is None:
        raise RuntimeError("RustFS 客户端未初始化，请检查 .env 与 RustFS 服务状态")

    resp = rustfs_client.list_objects_v2(Bucket=RUSTFS_BUCKET_NAME, Prefix=prefix)
    return [(o["Key"], o["Size"]) for o in resp.get("Contents", [])]


def delete_object(object_name):
    """删除对象"""
    if rustfs_client is None:
        raise RuntimeError("RustFS 客户端未初始化，请检查 .env 与 RustFS 服务状态")

    rustfs_client.delete_object(Bucket=RUSTFS_BUCKET_NAME, Key=object_name)


def _guess_content_type(suffix):
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".pdf": "application/pdf",
        ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
    }.get(suffix, "application/octet-stream")
