import torch
print("="*50)
print("PyTorch CUDA 支持检查")
print("="*50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
else:
    print("CUDA 不可用 - 使用 CPU 训练")
print("="*50)