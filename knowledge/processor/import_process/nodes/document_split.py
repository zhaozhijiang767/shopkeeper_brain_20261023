"""
文档切分节点

按 Markdown 标题切分文档，支持二次切分和短内容合并
"""
import re
import os
import json
from typing import List, Tuple, Optional

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import DocumentSplitError


class DocumentSplitNode(BaseNode):
    """
    文档切分节点

    处理流程：
    1. 读取 MD 内容
    2. 按 Markdown 标题进行一级切分（title 与 body 分离存储）
    3. 处理无标题情况
    4. 对超长章节进行二次切分
    5. 合并过短地相邻章节
    6. 组装最终 content = title + body
    """

    name = "document_split"

    # ------------------------------------------------------------------ #
    #                           主流程                                     #
    # ------------------------------------------------------------------ #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # Step 1: 获取输入
        content, file_title, max_length = self._get_inputs(state, config)
        if not content:
            raise DocumentSplitError("md_content 为空", node_name=self.name)

        # Step 2: 按标题一级切分
        sections, has_title = self._split_by_headings(content, file_title)

        # Step 3: 处理全文无标题的情况
        if not has_title:
            sections = [{"title": "无标题", "body": content, "file_title": file_title}]
            self.logger.info("全文无标题，作为单个 chunk 处理")

        # Step 4: 二次切分 + 合并短章节
        sections = self._split_and_merge(
            sections, max_length, config.min_content_length
        )

        # Step 5: 组装最终 content（title + body），清理内部字段
        sections = self._assemble_content(sections)

        # Step 6: 日志统计
        self._log_summary(content, sections, max_length)

        # Step 7: 备份
        state["chunks"] = sections
        self._backup_chunks(state, sections)

        return state

    # ------------------------------------------------------------------ #
    #                       Step 1: 获取输入                               #
    # ------------------------------------------------------------------ #

    def _get_inputs(
        self, state: ImportGraphState, config
    ) -> Tuple[Optional[str], Optional[str], int]:
        """获取输入参数并预处理"""
        self.log_step("step_1", "获取输入")

        content = state.get("md_content", "")
        if content:
            # 统一换行符
            content = content.replace("\r\n", "\n").replace("\r", "\n")

        file_title = state.get("file_title", "")
        max_length = config.max_content_length

        return content, file_title, max_length

    # ------------------------------------------------------------------ #
    #                  Step 2: 按标题一级切分                               #
    # ------------------------------------------------------------------ #

    def _split_by_headings(
        self, content: str, file_title: str
    ) -> Tuple[List[dict], bool]:
        """
        按 Markdown 标题行切分，title 与 body 分开存储。

        Returns:
            sections: [{"title": "# xxx", "body": "正文...", "file_title": ...}, ...]
            has_title: 文档中是否存在标题
        """
        self.log_step("step_2", "按标题切分")

        heading_re = re.compile(r"^\s*#{1,6}\s+.+")
        lines = content.split("\n")

        sections: List[dict] = [] # 存储切分结果
        current_title = "" # 当前章节标题
        body_lines: List[str] = [] # 当前章节正文行列表
        has_title = False # 标记文档是否有标题
        in_fence = False  # 是否在代码围栏内

        def _flush():
            """将当前积累的内容保存为一个 section"""
            body = "\n".join(body_lines).strip()
            if current_title or body:
                sections.append({
                    "title": current_title,
                    "body": body,
                    "file_title": file_title,
                })

        for line in lines:
            # 检测代码围栏（``` 或 ~~~）
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence

            is_heading = (not in_fence) and heading_re.match(line)

            if is_heading:
                has_title = True
                _flush()
                current_title = line.strip()
                body_lines = []
            else:
                body_lines.append(line)

        _flush()

        return sections, has_title

    # ------------------------------------------------------------------ #
    #                Step 4: 二次切分 + 合并短章节                          #
    # ------------------------------------------------------------------ #
    def _split_and_merge(
            self,
            sections: List[dict],
            max_length: int,
            min_length: int,
    ) -> List[dict]:
        """二次切分超长章节，合并过短章节"""
        self.log_step("step_4", "二次切分和合并")

        if max_length <= 0:
            return sections

        # 4a: 对超长章节做二次切分
        split_result: List[dict] = []
        for section in sections:
            split_result.extend(self._split_long_section(section, max_length))

        # 4b: 合并过短地相邻章节（仅限同一父标题下的子片段）
        return self._merge_short_sections(split_result, min_length)

    def _split_long_section(self, section: dict, max_length: int) -> List[dict]:
        """
        将超长章节按段落 → 句子逐级切分。

        最终每个子片段的 content 长度 = len(title_prefix) + len(body_piece) <= max_length
        """
        title = section.get("title", "")
        body = section.get("body", "")
        file_title = section.get("file_title", "")

        # title 作为前缀会占用一部分空间
        title_prefix = f"{title}\n\n" if title else ""
        total = len(title_prefix) + len(body)

        if total <= max_length:
            return [section]

        available = max_length - len(title_prefix)
        if available <= 0:
            return [section]

        # 按段落切分 body
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

        pieces: List[str] = []
        buf = ""

        for para in paragraphs:
            # 单个段落就超长 → 按句子装箱
            if len(para) > available:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.extend(self._pack_by_sentences(para, available))
                continue

            # 正常段落拼接
            new_len = len(buf) + (2 if buf else 0) + len(para)
            if new_len <= available:
                buf += ("\n\n" if buf else "") + para
            else:
                if buf:
                    pieces.append(buf)
                buf = para

        if buf:
            pieces.append(buf)

        # 只有一片，无需编号
        if len(pieces) <= 1:
            return [section]

        # 生成子片段
        return [
            {
                "title": f"{title}-{i + 1}" if title else f"chunk-{i + 1}",
                "body": piece,
                "file_title": file_title,
                "parent_title": title, # 记录父标题
                "part": i + 1, # 记录片段编号
            }
            for i, piece in enumerate(pieces)
        ]

    def _pack_by_sentences(self, para: str, max_len: int) -> List[str]:
        """将一个超长段落按句子边界装箱"""
        sentences = re.split(r"(?<=[。！？；.!?;])\s*", para)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        buf = ""

        for sent in sentences:
            if len(buf) + len(sent) <= max_len:
                buf += sent
            else:
                if buf:
                    chunks.append(buf)
                buf = sent

        if buf:
            chunks.append(buf)

        return chunks

    def _merge_short_sections(
            self, sections: List[dict], min_length: int
    ) -> List[dict]:
        """
        合并过短的相邻子片段（仅限同一 parent_title 下的片段）。

        合并条件：
        - 当前片段 body 长度 < min_length
        - 当前片段与下一片段拥有相同的 parent_title
        """
        if not sections:
            return []

        merged: List[dict] = []
        current = sections[0]

        for next_sec in sections[1:]:
            cur_body_len = len(current.get("body", ""))
            same_parent = (
                    current.get("parent_title")
                    and current["parent_title"] == next_sec.get("parent_title")
            )

            if cur_body_len < min_length and same_parent:
                # 合并: 将 next_sec 的 body 追加到 current
                current["body"] = (
                        current.get("body", "").rstrip()
                        + "\n\n"
                        + next_sec.get("body", "").lstrip()
                ).strip()
                # 标题回退为父标题
                current["title"] = current.get("parent_title", current.get("title", ""))
                # 更新 part 编号
                if "part" in next_sec:
                    current["part"] = next_sec["part"]
            else:
                merged.append(current)
                current = next_sec

        merged.append(current)
        return merged

    # ------------------------------------------------------------------ #
    #               Step 5: 组装最终 content                               #
    # ------------------------------------------------------------------ #
    def _assemble_content(self, sections: List[dict]) -> List[dict]:
        """
        将 title + body 组装为最终的 content 字段，
        清理内部临时字段 body，保留 parent_title 和 part 供下游使用。
        """
        self.log_step("step_5", "组装 content")

        result: List[dict] = []
        for sec in sections:
            title = sec.get("title", "")
            body = sec.get("body", "")

            # 组装: title 在最前面，body 紧随其后
            if title and body:
                content = f"{title}\n\n{body}"
            else:
                content = title or body

            chunk = {
                "title": title,
                "content": content.strip(),
                "file_title": sec.get("file_title", ""),
            }

            # 保留二次切分产生的字段，供下游合并/溯源使用
            if "parent_title" in sec:
                chunk["parent_title"] = sec["parent_title"]
            if "part" in sec:
                chunk["part"] = sec["part"]

            result.append(chunk)

        return result
    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, sections: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step_6", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(sections)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if sections:
            self.logger.info("章节预览:")
            for i, sec in enumerate(sections[:5]):
                title = sec.get("title", "")[:50]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(sections) > 5:
                self.logger.info(f"  ... 还有 {len(sections) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step_7", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

# 兼容原有调用方式
node_document_split = DocumentSplitNode()

if __name__ == '__main__':
    """
    文档切分节点测试

    测试不同场景下的切分逻辑
    """
    import json
    import os

    from knowledge.processor.import_process.base import setup_logging
    from knowledge.processor.import_process.nodes.document_split import node_document_split

    # 1. 开启日志
    setup_logging()

    print("=" * 60)
    print("DocumentSplitNode 节点测试")
    print("=" * 60)

    # -------------------- 测试用例 1: 正常文档 -------------------- #
    print("\n--- 测试用例 1: 正常 Markdown 文档 ---")

    sample_md = """# 第一章 万用表概述

万用表是一种多功能测量仪器，可以测量电压、电流、电阻等。

## 1.1 基本组成

万用表主要由以下部分组成：
- 显示屏
- 旋钮
- 测量端口

## 1.2 工作原理

万用表内部电路根据选择的测量模式切换不同的测量电路。

## 1.3 注意事项

使用时注意安全。

## 1.4 保养方法

定期清洁。
"""

    state = {
        "file_title": "万用表的使用",
        "md_content": sample_md,
        "file_dir": r"D:\test_output"
    }

    result = node_document_split.process(state)

    print(f"\n切分结果: {len(result['chunks'])} 个 chunks")
    for i, chunk in enumerate(result['chunks']):
        print(f"\n--- Chunk {i+1} ---")
        print(f"标题: {chunk['title']}")
        print(f"内容长度: {len(chunk['content'])} 字符")
        print(f"内容预览: {chunk['content'][:100]}...")

    # -------------------- 测试用例 2: 无标题文档 -------------------- #
    print("\n\n--- 测试用例 2: 无标题文档 ---")

    no_title_md = """这是一段没有标题的文档。

它包含多个段落，但没有使用 Markdown 标题格式。

这是第三段内容。
"""

    state_no_title = {
        "file_title": "无标题测试",
        "md_content": no_title_md,
    }

    result_no_title = node_document_split.process(state_no_title)
    print(f"切分结果: {len(result_no_title['chunks'])} 个 chunks")
    print(f"标题: {result_no_title['chunks'][0]['title']}")

    # -------------------- 测试用例 3: 代码块中的 # -------------------- #
    # -------------------- 测试用例 3: 代码块中的 # -------------------- #
    print("\n\n--- 测试用例 3: 代码块中的 # 不应被识别为标题 ---")

    code_block_md = """# 真正的标题

    下面是一段 Python 代码：

    ​```python
    # 这是注释，不是标题
    def hello():
        print("Hello")
    """