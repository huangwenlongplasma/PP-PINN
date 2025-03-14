import scipy
import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time


# 定义PINN模型
class PINN(torch.nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = torch.nn.Linear(2, 70)
        self.layer2 = torch.nn.Linear(70, 70)
        self.layer3 = torch.nn.Linear(70, 70)
        self.layer4 = torch.nn.Linear(70, 70)
        self.layer5 = torch.nn.Linear(70, 70)
        self.layer6 = torch.nn.Linear(70, 70)
        self.layer7 = torch.nn.Linear(70, 70)
        self.layer8 = torch.nn.Linear(70, 70)
        self.layer9 = torch.nn.Linear(70, 1)


    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        x = torch.tanh(self.layer1(inputs))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))
        x = torch.tanh(self.layer6(x))
        x = torch.tanh(self.layer7(x))
        x = torch.tanh(self.layer8(x))


        return self.layer9(x)


# 定义损失函数
def compute_loss(model, x, t, rho=1):
    u = model(x, t)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    f = u_t - rho * u * (1 - u)
    loss_f = torch.mean(f ** 2)

    # Initial condition loss
    u0 = model(x, torch.zeros_like(t))
    loss_ic = torch.mean((u0 - torch.exp(-(x - torch.pi)**2 / (0.2*torch.pi**2))) ** 2)

    # Boundary condition loss
    u_left = model(torch.zeros_like(x), t)
    u_right = model(2 * np.pi * torch.ones_like(x), t)
    loss_bc = torch.mean((u_left - u_right) ** 2)

    loss = loss_f + loss_ic + loss_bc

    return loss, loss_f, loss_ic, loss_bc


# 主函数
if __name__ == '__main__':
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机训练数据集
    num_samples = 1000
    x_train = (2 * np.pi * torch.rand(num_samples, 1)).to(device)
    x_train.requires_grad = True
    t_train = (1 * torch.rand(num_samples, 1)).to(device)
    t_train.requires_grad = True

    # Adam优化器
    model = PINN().to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 6000

    start_time = time.time()
    # 使用Adam训练模型
    losses = []
    pde_losses = []
    ic_losses = []
    bc_losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss, pde_loss, ic_loss, bc_loss = compute_loss(model, x_train, t_train)
        loss.backward()
        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        ic_losses.append(ic_loss.item())
        bc_losses.append(bc_loss.item())
        optimizer.step()
        if epoch % 100 == 0:
            print(
                f'Epoch {epoch}, Loss: {loss.item()}, PDE Loss: {pde_loss.item()}, IC Loss: {ic_loss.item()}, BC Loss: {bc_loss.item()}')

    # LBFGS优化器
    lbfgs_losses = []
    lbfgs_loss_f = []
    lbfgs_loss_ic = []
    lbfgs_loss_bc = []
    iteration = 0  # 初始化迭代计数


    def closure():
        global iteration  # 声明iteration为全局变量，这样我们可以在函数内部修改它
        optimizer_lbfgs.zero_grad()
        loss, loss_f, loss_ic, loss_bc = compute_loss(model, x_train, t_train)
        lbfgs_losses.append(loss.item())
        lbfgs_loss_f.append(loss_f.item())
        lbfgs_loss_ic.append(loss_ic.item())
        lbfgs_loss_bc.append(loss_bc.item())
        print(
            f"Iteration: {iteration} | Total Loss: {loss.item()}| Loss_f: {loss_f.item()} | Loss_ic: {loss_ic.item()} | Loss_bc: {loss_bc.item()}")

        loss.backward()
        iteration += 1
        return loss


    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=4000, max_eval=None, tolerance_grad=0,
                                        tolerance_change=0, history_size=100, line_search_fn=None)
    optimizer_lbfgs.step(closure)
    end_time = time.time()
    print(f"Time elapsed for training: {(end_time - start_time) / 60} min")

    Total_losses = losses + lbfgs_losses
    PDE_losses = pde_losses + lbfgs_loss_f
    IC_losses = ic_losses + lbfgs_loss_ic
    BC_losses = bc_losses + lbfgs_loss_bc
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.plot(Total_losses, label='Total Loss')
    plt.plot(PDE_losses, label='PDE Loss')
    plt.plot(IC_losses, label='IC Loss')
    plt.plot(BC_losses, label='BC Loss')
    plt.axvline(x=num_epochs, color='grey', linestyle='--', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Training with Adam and L-BFGS')
    plt.legend()
    plt.savefig('fig/loss1_1tanh.png')
    plt.savefig('fig/loss1_1tanh.pdf')
    plt.show()
    # 测试模型
    model.eval()
    x = torch.linspace(0, 2 * torch.pi, 1000)
    t = torch.linspace(0, 1, 1000)
    X, T = torch.meshgrid(x, t)
    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    X = X.to(device, dtype=torch.float32)
    T = T.to(device, dtype=torch.float32)
    rho = 1
    h_x = torch.exp(-(X - torch.pi) ** 2 / (0.2*torch.pi ** 2))
    numerator = h_x * torch.exp(rho * T)
    denominator = h_x * torch.exp(rho * T) + 1 - h_x
    real_u = numerator / denominator
    uxt = model(X, T)
    real_u = real_u.cpu().detach().numpy()
    uxt = uxt.cpu().detach().numpy()
    L2_error = np.linalg.norm((real_u - uxt).flatten(), 2) / np.linalg.norm(real_u.flatten(), 2)
    print('相对误差: %e' % (L2_error))
    Error = np.mean(np.abs(real_u - uxt))
    print('绝对误差: %e' % (Error))
    
    relative_error = (real_u - uxt) / real_u
    relative_error = relative_error.reshape(1000, 1000)
    
    plt.figure(figsize=(10, 6))
    cp = plt.contourf(T.cpu().detach().numpy().reshape(1000, 1000), X.cpu().detach().numpy().reshape(1000, 1000), np.abs(relative_error), 50, cmap='jet')
    plt.colorbar(cp)  # 显示颜色条
    plt.title('Relative Error')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig('fig/lcr_error_1_1tanh.png')
    plt.savefig('fig/lcr_error_1_1tanh.pdf')
    plt.show()

    # 保存数据
    x = torch.linspace(0, 2 * torch.pi, 1000).to(device)
    t = torch.linspace(0, 1, 1000).to(device)
    x = x.unsqueeze(1)
    t = t.unsqueeze(1)
    solution = model(x, t)
    scipy.io.savemat('data/lcr_pred1_1tanh.mat',
                     {'x': x.cpu().detach().numpy().reshape(-1, 1), 't': t.cpu().detach().numpy().reshape(-1, 1),
                      'u': solution.cpu().detach().numpy().reshape(-1, 1)})
    t_fp = torch.ones_like(x)
    u_fp = model(x, t_fp)
    scipy.io.savemat('data/lcr_fp1_1tanh.mat',
                     {'x': x.cpu().detach().numpy().reshape(-1, 1), 't': t_fp.cpu().detach().numpy().reshape(-1, 1),
                      'u': u_fp.cpu().detach().numpy().reshape(-1, 1)})

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.cpu().detach().numpy().reshape(1000, 1000), T.cpu().detach().numpy().reshape(1000, 1000),uxt.reshape(1000, 1000), cmap='jet')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('pred')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.savefig('fig/lcr_pred_1_1tanh.png')
    plt.savefig('fig/lcr_real_1_1tanh.pdf')
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.cpu().detach().numpy().reshape(1000, 1000), T.cpu().detach().numpy().reshape(1000, 1000),real_u.reshape(1000, 1000), cmap='jet')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('real')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.savefig('fig/lcr_real_1_1tanh.png')
    plt.savefig('fig/lcr_real_1_1tanh.pdf')
    plt.show()

    save_path = 'model/lcr1_1tanh.pth'
    torch.save(model.state_dict(), save_path)
