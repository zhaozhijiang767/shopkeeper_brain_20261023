"""
导入流程主图

使用 LangGraph 构建文档导入工作流
"""

from dotenv import load_dotenv

from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.md_img import MdImgNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode

# 加载环境变量
load_dotenv()
from langgraph.graph import StateGraph, END

from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def route_after_entry(state: ImportGraphState) -> str:
    """
    入口节点后的路由逻辑

    根据文件类型决定走 PDF 转换分支还是直接处理 MD 分支

    Args:
        state: 当前图状态

    Returns:
        下一个节点名称
    """
    if state.get("is_md_read_enabled"):
        return "md_img"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md"
    return END


def create_import_graph() -> StateGraph:
    """
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry
          │
          ├── (PDF) ──> pdf_to_md ──┐
          │                         │
          └── (MD) ────────────────>├──> md_img
                                    │
                                    v
                            document_split
                                    │
                                    v
                        item_name_recognition
                                    │
                                    v
                            bge_embedding
                                    │
                                    v
                            import_milvus
                                    │
                                    v
                          knowledge_graph
                                    │
                                    v
                                   END
    """

    # 1. 定义图状态的工作流
    workflow = StateGraph(ImportGraphState)

    # 2. 实例化各个节点
    nodes = {
        "entry": EntryNode(),
        "pdf_to_md": PdfToMdNode(),
        "md_img": MdImgNode(),
        "document_split": "DocumentSplitNode()",
        "item_name_recognition": "ItemNameRecognitionNode()",
        "bge_embedding": "BgeEmbeddingNode()",
        "import_milvus": "ImportMilvusNode()",
        "knowledge_graph": "KnowledgeGraphNode()",
    }

    # 3. 添加节点到图状态的工作流中
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 4. 设置图状态工作流的入口节点
    workflow.set_entry_point("entry")

    # 5. 添加条件边：入口节点后根据文件类型路由
    workflow.add_conditional_edges(
        "entry",
        route_after_entry,
        {
            "md_img": "md_img",
            "pdf_to_md": "pdf_to_md",
            END: END
        }
    )

    # 6. 添加顺序边
    workflow.add_edge("pdf_to_md", "md_img")
    workflow.add_edge("md_img", "document_split")
    workflow.add_edge("document_split", "item_name_recognition")
    workflow.add_edge("item_name_recognition", "bge_embedding")
    workflow.add_edge("bge_embedding", "import_milvus")
    workflow.add_edge("import_milvus", "knowledge_graph")
    workflow.add_edge("knowledge_graph", END)

    # 7. 返回构建好的图工作流
    return workflow.compile()


# 创建全局图实例
kb_import_app = create_import_graph()


def run_import(file_dir: str = "", import_file_path: str = "") -> dict:
    """
    便捷函数：运行导入流程

    Args:
        file_dir: 本地工作目录
        import_file_path: 输入文件路径（PDF 或 MD）

    Returns:
        最终状态字典
    """

    # 1. 创建初始状态
    initial_state = create_default_state(
        file_dir=file_dir,
        import_file_path=import_file_path
    )
    final_state = None

    # 2. 运行图的工作流节点
    # stream() 方法逐步获取每个节点的执行结果
    for event in kb_import_app.stream(initial_state):
        for key, value in event.items():
            print(f"节点: {key}")
            final_state = value

    # 3. 返回运行后的图状态
    return final_state or initial_state
