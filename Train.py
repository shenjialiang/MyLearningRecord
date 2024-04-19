import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='dataset_CIFAR10', train=True, download=True,
                                              transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='dataset_CIFAR10', train=False, download=True,
                                             transform=torchvision.transforms.ToTensor())
# 查看数据集的长度
train_len = len(train_data)
test_len = len(test_data)
print('训练集的长度为：{}'.format(train_len))
print('测试集的长度为：{}'.format(test_len))

# 准备加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建网络模型
class M_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = M_CIFAR10()
model = model.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter(log_dir='logs_model')
start_time = time.time()

for i in range(epoch):
    print('----------第{}轮训练开始----------'.format(i+1))
    # 训练步骤开始
    model.train()
    for data in train_loader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # 优化器优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print('训练时间:{}'.format(end_time - start_time))
            print('训练次数：{}, Loss：{}'.format(total_train_step, loss.item()))   # .item()的作用：把tensor数据类型转换为真实的数字
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            test_loss = loss_fn(outputs, labels)
            total_test_loss += test_loss.item()
            total_accuracy += (outputs.argmax(1) == labels).sum()
        total_test_loss /= len(test_loader)
    print('整体测试集上的loss为：{}'.format(total_test_loss))
    print('整体测试集上的正确率为：{}'.format(total_accuracy/len(test_loader)))
    writer.add_scalar('train_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/len(test_loader), total_test_step)
    total_test_step += 1

    # 保存每一轮训练的结果
    torch.save(model, 'model_epoch{}.pth'.format(i))
    # torch.save(model.state_dict(), 'model_epoch{}.pth'.format(i))
    print('模型已保存')

writer.close()



