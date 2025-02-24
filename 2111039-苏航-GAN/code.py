import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from PIL import Image
class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.size(0), 784) # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out) # range [-1, 1]
        # convert to image
        out = out.view(out.size(0), 1, 28, 28)
        return out
class CNN_Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(CNN_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 100, 1, 1)  # (batch_size, z_dim, 1, 1)
        return self.main(x)
class CNN_Discriminator(nn.Module):
    def __init__(self):
        super(CNN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]),
                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def show_imgs(x, title="None", new_fig=True, save_path=None):
    x = x.cpu()
    if new_fig:
        plt.figure()
    grid_img = torchvision.utils.make_grid(x, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')  # 隐藏坐标轴
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
D = Discriminator().to(device)
G = Generator().to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, 100, device=device)
def train_and_generate_images(G, D, num_rounds=200):
    # 标签定义
    lab_real = torch.ones(64, 1, device=device)
    lab_fake = torch.zeros(64, 1, device=device)
    # 储存损失值
    loss_D_list = []
    loss_G_list = []
    # 将数据加载器转换为迭代器
    dataloader_iter = iter(dataloader)
    model_count = 1  # 模型保存次数
    for epoch in range(num_rounds):  
        for i, data in enumerate(dataloader, 0):
            # STEP 1: 判别器优化步骤
            try:
                x_real, _ = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                x_real, _ = next(dataloader_iter)

            x_real = x_real.to(device)
            batch_size = x_real.size(0)  # 动态获取批量大小
            lab_real = torch.ones(batch_size, 1, device=device)
            lab_fake = torch.zeros(batch_size, 1, device=device)
            optimizerD.zero_grad()
            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real)
            z = torch.randn(batch_size, 100, device=device)  # 随机噪声，64个样本，z_dim=100
            x_gen = G(z).detach()
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake)
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()
            # STEP 2: 生成器优化步骤
            optimizerG.zero_grad()
            z = torch.randn(batch_size, 100, device=device)  # 随机噪声，64个样本，z_dim=100
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real)  # -log D(G(z))
            lossG.backward()
            optimizerG.step()
        loss_D_list.append(lossD.item())
        loss_G_list.append(lossG.item())
        print(epoch, lossD.item(), lossG.item())
        # 保存模型
        if epoch % 10 == 0:
            D_path = os.path.join(save_dir, f'discriminator_{model_count}.pth')
            G_path = os.path.join(save_dir, f'generator_{model_count}.pth')
            torch.save(D.state_dict(), D_path)
            torch.save(G.state_dict(), G_path)

        model_count += 1

    return loss_D_list, loss_G_list
loss_D_list, loss_G_list = train_and_generate_images(G, D)
plt.plot(loss_D_list, label="Discriminator Loss")
plt.plot(loss_G_list, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.show()
target_indices = [22, 27, 32, 36, 47, 50, 59, 94]
for i in target_indices:
    G = CNN_Generator()
    # 使用字符串格式化方法插入变量值
    G.load_state_dict(torch.load(fr'.\CNN_GAN\generator_{i}.pth', map_location=device))
    G.eval()
    with torch.no_grad():
        x_gen = G(fixed_noise)
    show_imgs(x_gen, title=f'Generator {i}', save_path=fr'.\test_1\generator_{i}.png')
im = Image.open('./test_1/generator_22.png')
w, h = im.size
image_row = 4
image_column = 2
names = os.listdir('./test_1')
new_img = Image.new('RGB', (image_column * w, image_row * h))
for y in range(image_row):
    for x in range(image_column):
        o_img = Image.open('./test_1/' + names[image_column * y + x])
        new_img.paste(o_img, (x * w, y * h))
new_img.save('随机噪声.jpg')


