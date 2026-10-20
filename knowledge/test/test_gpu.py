import torch
print(f"是否支持GPU: {torch.cuda.is_available()}")
print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


"""
$env:MODELSCOPE_CACHE = "D:\ai_models\modelscope_cache"
下载mineru模型 mineru版本3.4.5
mineru-models-download
满足gpu条件
mineru -p <input.path> -o <output.path> --source local

不满足gpu条件，使用pipeline模式
mineru -p <input.path> -o <output.path> --source local -b pipeline 
mineru -p D:\\projects\\shopkeeper_brain\\knowledge\temp_data\\input\\万用表RS-12的使用.pdf -o D:\\projects\\shopkeeper_brain\\knowledge\temp_data\\output --source local -b pipeline 


"""