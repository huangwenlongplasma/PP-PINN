import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from scipy.interpolate import griddata
from matplotlib.ticker import FuncFormatter


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
def compute_loss(model, x, t, x_data, t_data, real, x_fp, t_fp,  previous_solution, rho=1):
    u = model(x, t)
#     u = u / 0.1
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    f = u_t - rho * u * (1 - 0.1*u)
    loss_f = torch.mean(f ** 2)

    # Initial condition loss
    u0 = model(x, torch.zeros_like(t))
    loss_ic = torch.mean((u0 - torch.exp(-(x - torch.pi)**2 / (0.2*torch.pi**2))) ** 2)

    # Boundary condition loss
    u_left = model(torch.zeros_like(x), t)
    u_right = model(2 * np.pi * torch.ones_like(x), t)
    loss_bc = torch.mean((u_left - u_right) ** 2)

    # data loss
    u_data = model(x_data, t_data)
    loss_data = torch.mean((u_data - real) ** 2)

    # fp loss
    u_fp = model(x_fp, t_fp)
    loss_fp = torch.mean((u_fp - previous_solution) ** 2)

    loss = loss_f + loss_ic + loss_bc + loss_data + loss_fp

    return loss, loss_f, loss_ic, loss_bc, loss_data, loss_fp

# 主函数
if __name__ == '__main__':
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = torch.load('model/lcr0.1_2.pth')
    # 创建随机训练数据集
    num_samples = 1000
    x_train = (2 * np.pi * torch.rand(num_samples, 1)).to(device)
    x_train.requires_grad = True
    t_train = (10 * torch.rand(num_samples, 1)).to(device)
    t_train.requires_grad = True

    # Adam优化器
    model = PINN()
    model.load_state_dict(model_path)
    model.train()
    model.to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 8000

    data = scipy.io.loadmat('data/lcr_pred10_20.mat')
    real = data['u']
    x = data['x'].flatten()[:, None]
    t = data['t'].flatten()[:, None]
    real = torch.from_numpy(real)
    X = torch.from_numpy(x)
    T = torch.from_numpy(t)
    x_data = X.to(device, dtype=torch.float32)
    t_data = T.to(device, dtype=torch.float32)
    real = real.to(device, dtype=torch.float32)


    data = scipy.io.loadmat('data/lcr_fp10_20.mat')
    x_fp = data['x']
    t_fp = data['t']
    previous_solution = data['u']
    previous_solution = previous_solution.reshape(-1, 1)
    previous_solution = torch.from_numpy(previous_solution)
    previous_solution = previous_solution.to(device, dtype=torch.float32)
    x_fp = x_fp.reshape(-1, 1)
    t_fp = t_fp.reshape(-1, 1)
    x_fp = torch.from_numpy(x_fp)
    t_fp = torch.from_numpy(t_fp)
    x_fp = x_fp.to(device)
    t_fp = t_fp.to(device)
    x_fp = x_fp.requires_grad_(True)
    t_fp = t_fp.requires_grad_(True)

    start_time = time.time()
    # 使用Adam训练模型
    losses = []
    pde_losses = []
    ic_losses = []
    bc_losses = []
    data_losses = []
    fp_losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss, pde_loss, ic_loss, bc_loss, loss_data, loss_fp = compute_loss(model, x_train, t_train, x_data, t_data, real, x_fp, t_fp,  previous_solution)
        loss.backward()
        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        ic_losses.append(ic_loss.item())
        bc_losses.append(bc_loss.item())
        data_losses.append(loss_data.item())
        fp_losses.append(loss_fp.item())
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, PDE Loss: {pde_loss.item()}, IC Loss: {ic_loss.item()}, BC Loss: {bc_loss.item()}, Data Loss: {loss_data.item()}, FP Loss: {loss_fp.item()}')


    # LBFGS优化器
    lbfgs_losses = []
    lbfgs_loss_f = []
    lbfgs_loss_ic = []
    lbfgs_loss_bc = []
    lbfgs_loss_data = []
    lbfgs_loss_fp = []
    iteration = 0  # 初始化迭代计数
    def closure():
        global iteration  # 声明iteration为全局变量，这样我们可以在函数内部修改它
        optimizer_lbfgs.zero_grad()
        loss, pde_loss, ic_loss, bc_loss, loss_data, loss_fp = compute_loss(model, x_train, t_train, x_data, t_data,
                                                                            real, x_fp, t_fp, previous_solution)
        lbfgs_losses.append(loss.item())
        lbfgs_loss_f.append(pde_loss.item())
        lbfgs_loss_ic.append(ic_loss.item())
        lbfgs_loss_bc.append(bc_loss.item())
        lbfgs_loss_data.append(loss_data.item())
        lbfgs_loss_fp.append(loss_fp.item())
        print(
            f"Iteration: {iteration} | Total Loss: {loss.item()} | PDE loss: {pde_loss.item()} | IC loss: {ic_loss.item()} | BC loss: {bc_loss.item()} | Data loss: {loss_data.item()} | FP loss: {loss_fp.item()} ")
        loss.backward()
        iteration += 1  # 在每次迭代结束后增加迭代计数
        return loss


    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),  max_iter=4000, max_eval=None, tolerance_grad=0,
                                        tolerance_change=0, history_size=100, line_search_fn=None)
    optimizer_lbfgs.step(closure)
    end_time = time.time()
    print(f"Time elapsed for training: {(end_time - start_time) / 60} min")
    
    Total_losses = losses + lbfgs_losses
    PDE_losses = pde_losses + lbfgs_loss_f
    IC_losses = ic_losses + lbfgs_loss_ic
    BC_losses = bc_losses + lbfgs_loss_bc
    DATA_losses = data_losses + lbfgs_loss_data
    FP_losses = fp_losses + lbfgs_loss_fp
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.plot(Total_losses, label='Total Loss')
    plt.plot(PDE_losses, label='PDE Loss')
    plt.plot(IC_losses, label='IC Loss')
    plt.plot(BC_losses, label='BC Loss')
    plt.plot(DATA_losses, label='Data Loss')
    plt.plot(FP_losses, label='FP Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Training with Adam and L-BFGS')
    plt.legend()
    plt.savefig('fig/loss0.1_10.png')
#     plt.savefig('fig/loss10_20.pdf')
    plt.show()
    # 测试模型
    model.eval()
    x = torch.linspace(0, 2 * torch.pi, 1000)
    t = torch.linspace(0, 10, 1000)
    X, T = torch.meshgrid(x, t)
    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    X = X.to(device, dtype=torch.float32)
    T = T.to(device, dtype=torch.float32)
    rho = 1
    h_x = 0.1*torch.exp(-(X - torch.pi) ** 2 / (0.2*torch.pi ** 2))
    numerator = h_x * torch.exp(rho * T)
    denominator = h_x * torch.exp(rho * T) + 1 - h_x
    real_u = numerator / denominator
    uxt = model(X, T)
    real_u = real_u.cpu().detach().numpy()
    print(real_u.shape)
    print(X.shape)
    print(T.shape)
    uxt = 0.1*uxt.cpu().detach().numpy()
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
#     plt.savefig('fig/lcr_error_10_20.png')
#     plt.savefig('fig/lcr_error_10_20.pdf')
    plt.show()

#     保存数据
    x = torch.linspace(0, 2 * torch.pi, 1000).to(device)
    t = torch.linspace(0, 10, 1000).to(device)
    x = x.unsqueeze(1)
    t = t.unsqueeze(1)
    solution = model(x, t)
    scipy.io.savemat('data/lcr_pred10_3.mat',
                     {'x': x.cpu().detach().numpy().reshape(-1, 1), 't': t.cpu().detach().numpy().reshape(-1, 1),
                      'u': solution.cpu().detach().numpy().reshape(-1, 1)})
    t_fp = torch.ones_like(x) * 10
    u_fp = model(x, t_fp)
    scipy.io.savemat('data/lcr_fp10_3.mat',
                     {'x': x.cpu().detach().numpy().reshape(-1, 1), 't': t_fp.cpu().detach().numpy().reshape(-1, 1),
                      'u': u_fp.cpu().detach().numpy().reshape(-1, 1)})




    uxt_flat = uxt.flatten()
    real_u = real_u.flatten()
    T_scaled = (T - T.min()) / (T.max() - T.min())
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.cpu().detach().numpy().reshape(1000, 1000), T_scaled.cpu().detach().numpy().reshape(1000, 1000),uxt.reshape(1000, 1000), cmap='jet')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_yticks([0.0, 0.5, 1.0])
#     ax.set_zticks([uxt_flat.min(), (uxt_flat.max() + uxt_flat.min()) / 2, uxt_flat.max()]) # z 轴分为三个刻度
#     z_ticks = [uxt_flat.min(), (uxt_flat.max() + uxt_flat.min()) / 2, uxt_flat.max()]
#     ax.set_zticklabels([f'{z:.1f}' for z in z_ticks])
    ax.set_zticks([0.0,  0.05,0.10])
    ax.set_title('Ours', fontsize=23)
    ax.set_xlabel('x', fontsize=23)
    ax.set_xticks([0, 3, 6])
    ax.set_ylabel('t', fontsize=23)
#     ax.set_zlabel('u', fontsize=16)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    for t in ax.zaxis.get_major_ticks():
        t.set_pad(10)

    # 自动调整子图参数
    plt.tight_layout()
#     ax.set_zscale('log')/
#     ax.set_zmargin(0.2)  # 增加 z 轴的边距
    plt.savefig('fig/lcr_pred_0.1_3.png')
#     plt.savefig('fig/lcr_pred_10_20.pdf')
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.cpu().detach().numpy().reshape(1000, 1000), T_scaled.cpu().detach().numpy().reshape(1000, 1000),real_u.reshape(1000, 1000), cmap='jet')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Exact Solution', fontsize=23)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.05, 0.1])
    ax.set_xlabel('x', fontsize=23)
    ax.set_xticks([0, 3, 6])
    ax.set_ylabel('t', fontsize=23)
#     ax.set_zlabel('u', fontsize=16)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    for t in ax.zaxis.get_major_ticks():
        t.set_pad(10)

    # 自动调整子图参数
    plt.tight_layout()
    plt.savefig('fig/lcr_real_0.1_3.png')
#     plt.savefig('fig/lcr_real_10_20.pdf')
    plt.show()

    save_path = 'model/lcr0.1_3.pth'
    torch.save(model.state_dict(), save_path)
