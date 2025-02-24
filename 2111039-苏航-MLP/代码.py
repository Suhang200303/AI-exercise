import torchvision
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307, ), (0.3081, )) 
])

# 加载MNIST训练集和测试集
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, transform=transform, download=True)


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
def train(model, train_loader, optimizer, criterion, device):
    # 将模型设置为训练模式
    model.train()  
    train_loss = 0
    correct = 0
    total = 0
    # 遍历训练集
    for inputs, labels in train_loader:
        # 将输入数据和标签移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()  
        # 前向传播计算输出
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播更新参数
        loss.backward()
        optimizer.step()
        # 统计总损失、预测正确的样本数和总样本数
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = correct / total
    return train_loss, acc
def test(model, test_loader, criterion, device):
    # 将模型设置为评估模式
    model.eval()
    test_loss = 0  
    correct = 0    
    total = 0       
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算损失
            
            test_loss += loss.item()  # 累加测试集上的总损失
            _, predicted = outputs.max(1)  # 获取预测结果中概率最大的类别
            total += labels.size(0)  # 累加样本数
            correct += predicted.eq(labels).sum().item()  # 统计预测正确的样本数   
    acc = correct / total  
    return test_loss, acc
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 绘制损失和准确度曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    return train_losses, test_losses, train_accs, test_accs
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256) #前一个是输入 后一个是输出
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU() # 这里选取的是ReLU函数作为激活函数
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # 将输入展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0015)
criterion = nn.CrossEntropyLoss()
train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10)

class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
model = DeepMLP().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10)

class wide_MLP(nn.Module):
    def __init__(self):
        super(wide_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512) #前一个是输入 后一个是输出
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU() # 这里选取的是ReLU函数作为激活函数
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # 将输入展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = wide_MLP().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10)

class ResMLP(nn.Module):
    def __init__(self):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)  # 确保输入输出维度相同以便残差连接
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 64)  # 确保输入输出维度相同以便残差连接
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入   
        # 第一层
        out = self.relu(self.fc1(x))
        # 第二层 + 残差连接
        residual = out
        out = self.relu(self.fc2(out))
        out = out + residual 
        # 第三层
        out = self.relu(self.fc3(out))
        # 第四层 + 残差连接
        residual = out
        out = self.relu(self.fc4(out))
        out = out + residual# 添加残差
        # 输出层
        out = self.fc5(out)
        return out
model = ResMLP().to(device)
# 输出网络结构
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10)