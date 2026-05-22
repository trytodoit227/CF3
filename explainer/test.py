import torch
import torch.nn as nn

# 定义两个二维张量
input = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
target = torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])

# 将张量转换为概率分布
input_prob = nn.functional.softmax(input, dim=1)
target_prob = nn.functional.softmax(target, dim=1)

# 创建KLDivLoss实例
criterion = nn.KLDivLoss(reduction='batchmean')

# 计算KL散度损失
loss = criterion(torch.log(input_prob), target_prob)

# 打印结果
print(loss.item())

print('ok!')