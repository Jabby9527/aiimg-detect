"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/5
"""
import torch
# 创建两个示例张量，这里以简单的一维整型张量为例，你可以根据需求替换维度和数据类型等
tensor1 = torch.tensor([1, 0, 1, 1], dtype=torch.int)
tensor2 = torch.tensor([1, 1, 0, 1], dtype=torch.int)

# 确定运行设备，优先选择CUDA设备（如果可用），其次看MPS设备（如果可用），最后是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 将张量移动到选定的设备上
tensor1 = tensor1.to(device)
tensor2 = tensor2.to(device)

# 执行按位与逻辑操作
result = tensor1 & tensor2
print(torch.__version__)

print("在设备", device, "上执行操作的结果：")
print(result)

best_accu = 999.2222
str = f'macos_net_epoch_best_{int(best_accu)}.pth'
print(str)