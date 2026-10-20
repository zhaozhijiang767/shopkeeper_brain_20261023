# 项目目录结构说明

```
shopkeeper_brain/
├── main.py                               # 项目入口
│
└── knowledge/                            # 知识库核心模块
    ├── api/                              # API 路由层
    │   ├── query_router.py               # 查询服务路由 (port 8001)
    │   │   ├── POST /query               # 发起查询
    │   │   ├── GET /stream/{session_id}  # SSE 流式获取
    │   │   ├── GET /history/{session_id} # 获取历史
    │   │   └── DELETE /history/...       # 清除历史
    │   └── import_router.py              # 导入服务路由 (port 8000)
    │       ├── POST /upload              # 上传文件
    │       └── GET /status/{task_id}     # 查询任务状态
    │
    ├── core/                             # 核心配置
    │   ├── deps.py                       # 依赖注入（单例管理）
    │   └── paths.py                      # 路径常量配置
    │
    ├── processor/                        # 业务处理流程（LangGraph）
    │   ├── import_process/               # 导入流程
    │   │   ├── main_graph.py             # 导入流程图定义
    │   │   ├── state.py                  # 状态类型定义
    │   │   └── nodes/                    # 处理节点
    │   │       ├── entry.py              # 入口节点
    │   │       ├── pdf_to_md.py          # PDF 转 MD
    │   │       ├── md_img.py             # 图片处理
    │   │       ├── document_split.py     # 文档切分
    │   │       ├── item_name_recognition.py  # 商品识别
    │   │       ├── bge_embedding.py      # 向量嵌入
    │   │       ├── import_milvus.py      # Milvus 存储
    │   │       └── knowledge_graph.py    # 知识图谱构建
    │   │
    │   └── query_process/                # 查询流程
    │       ├── main_graph.py             # 查询流程图定义
    │       ├── state.py                  # 状态类型定义
    │       ├── prompt.py                 # 提示词模板
    │       └── nodes/                    # 处理节点
    │           ├── item_name_confirm.py  # 商品确认
    │           ├── search_embedding.py   # 向量检索
    │           ├── search_embedding_hyde.py  # HyDE 检索
    │           ├── query_kg.py           # 知识图谱查询
    │           ├── web_search_mcp.py     # Web 搜索
    │           ├── rrf.py                # RRF 融合
    │           ├── rerank.py             # 重排序
    │           └── answer_output.py      # 答案生成
    │
    ├── schemas/                          # 数据模型定义
    │   ├── query_schema.py               # 查询请求/响应模型
    │   ├── upload_schema.py              # 上传响应模型
    │   └── task_schema.py                # 任务状态模型
    │
    ├── services/                         # 业务服务层
    │   ├── file_import_service.py        # 文件导入服务
    │   └── task_service.py               # 任务管理服务
    │
    ├── tools/                            # 工具函数库
    │   ├── milvus_utils.py               # Milvus 向量库操作
    │   ├── neo4j_utils.py                # Neo4j 图数据库操作
    │   ├── embedding_utils.py            # 向量嵌入工具
    │   ├── llm_utils.py                  # LLM 客户端封装
    │   ├── reranker_utils.py             # Reranker 模型
    │   ├── mongo_history_utils.py        # MongoDB 历史记录
    │   ├── sse_utils.py                  # SSE 流式推送
    │   ├── task_utils.py                 # 任务状态管理
    │   ├── minio_utils.py                # 对象存储（S3 兼容）
    │   └── normalize_sparse_vector.py    # 稀疏向量规范化
    │
    ├── front/                            # 前端页面
    │   ├── chat.html                     # 聊天界面
    │   └── import.html                   # 导入界面
    │
    ├── configs/                          # 服务配置文件
    │   └── milvus.yaml                  # Milvus S3 对象存储（RustFS）连接配置
    │
    ├── test/                             # 测试代码
    ├── docs/                             # 文档目录
    ├── temp_data/                        # 临时数据目录
    ├── volumes/                          # Docker 数据持久化目录
    │   ├── etcd/                         # etcd 元数据
    │   ├── milvus/                       # Milvus 本地数据（RocksMQ 日志等）
    │   ├── mongo/                        # MongoDB 数据
    │   └── rustfs/                       # RustFS 对象存储数据
    │
    ├── docker-compose.yml                # Docker Compose 编排（etcd / rustfs / milvus / attu / neo4j / mongo）
    ├── .env                              # 环境配置（API 密钥、数据库连接等）
    ├── requirements.txt                  # Python 依赖声明
    └── requirements.lock.txt             # 锁定版本依赖
```

## 各层职责速查

| 层 | 目录 | 职责 |
|----|------|------|
| 路由层 | `api/` | 暴露 HTTP 接口（查询 :8001 / 导入 :8000） |
| 流程层 | `processor/` | 基于 LangGraph 的 DAG 编排（导入 / 查询两条主线） |
| 服务层 | `services/` | 业务逻辑封装，被路由层调用 |
| 工具层 | `tools/` | 各外部组件（Milvus / Neo4j / LLM / S3 等）的客户端封装 |
| 模型层 | `schemas/` | Pydantic 数据模型，定义请求/响应结构 |
| 前端 | `front/` | 静态 HTML 页面（聊天 + 导入） |
| 配置 | `configs/` + `.env` | 服务连接参数、API 密钥 |
| 编排 | `docker-compose.yml` | 一键拉起 etcd / RustFS / Milvus / Attu / Neo4j / MongoDB |
