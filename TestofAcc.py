import torch

ts = torch.tensor(([1, 3, 5],
                    [0, 4, 1],
                    [2, 2, 2]))

# 返回tensor中每一行中最大值的列号
# 如果argmax中的参数是0，那就是返回tensor中每一列最大值的行号
pred = ts.argmax(1)
print(pred)
labels = torch.tensor(([0, 1, 2]))
# 计算出预测结果和label相等的个数
print(pred == labels)
print((pred == labels).sum())





