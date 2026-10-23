"""项目路径工具"""

import os

KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件归档根目录
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端静态页面目录
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")


def get_local_base_dir() -> str:
    """本地文件存储基础目录。"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """前端静态页面目录。"""
    return FRONT_PAGE_DIR
