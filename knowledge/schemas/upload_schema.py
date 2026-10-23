"""上传相关 Schema 定义"""

from typing import List

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """文件上传响应"""
    message: str = Field(..., description="响应消息")
    task_ids: List[str] = Field(..., description="每个文件对应的任务ID")
