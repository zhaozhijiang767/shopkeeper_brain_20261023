import subprocess

# 基础用法：run() - 同步执行，等待完成
result = subprocess.run(
    ["mineru", "-p", "input.pdf", "-o", "output/"],
    capture_output=True,  # 捕获输出
    text=True,            # 文本模式
    check=True            # 非零返回码抛出异常
)
print(result.stdout)

# 高级用法：Popen() - 实时获取输出
proc = subprocess.Popen(
    ["mineru", "-p", "input.pdf"],
    stdout=subprocess.PIPE,    # 捕获标准输出
    stderr=subprocess.STDOUT,  # 合并错误到标准输出
    text=True,
    bufsize=1  # 行缓冲，实时输出
)

# 逐行读取输出
for line in proc.stdout:
    print(line.rstrip())

# 等待完成
return_code = proc.wait()


"""
关键参数说明：**

| 参数  | 说明  |
| --- | --- |
| `stdout=subprocess.PIPE` | 捕获标准输出 |
| `stderr=subprocess.STDOUT` | 将错误流合并到标准输出 |
| `text=True` | 文本模式（自动解码为字符串） |
| `encoding="utf-8"` | 指定编码 |
| `errors="replace"` | 遇到无法解码的字符时替换（避免崩溃） |
| `env=os.environ.copy()` | 传递当前环境变量 |
| `bufsize=1` | 行缓冲，每行立即输出 |
"""