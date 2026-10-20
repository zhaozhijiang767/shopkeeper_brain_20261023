import os

# 获取CUDA_PATH环境变量
cuda_path = os.getenv('CUDA_PATH')

# 使用assert检查CUDA_PATH是否已设置
assert cuda_path is not None, 'Can not find $env:CUDA_PATH'
print(f"cuda的地址:{cuda_path}")