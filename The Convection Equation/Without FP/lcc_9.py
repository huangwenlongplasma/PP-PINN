import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from scipy.interpolate import griddata



# 定义PINN模型
class PINN(torch.nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = torch.nn.Linear(2, 60)
        self.layer2 = torch.nn.Linear(60, 60)
        self.layer3 = torch.nn.Linear(60, 60)
        self.layer4 = torch.nn.Linear(60, 60)
        self.layer5 = torch.nn.Linear(60, 60)
        self.layer6 = torch.nn.Linear(60, 60)
        self.layer7 = torch.nn.Linear(60, 60)
        self.layer8 = torch.nn.Linear(60, 60)
        self.layer9 = torch.nn.Linear(60, 1)

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
def compute_loss(model, x, t, x_data, t_data, real,  beta=20):
    u = model(x, t)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    f = u_t + beta * u_x
    loss_f = torch.mean(f ** 2)

    # Initial condition loss
    u0 = model(x, torch.zeros_like(t))
    loss_ic = torch.mean((u0 - torch.sin(x)) ** 2)

    # Boundary condition loss
    u_left = model(torch.zeros_like(x), t)
    u_right = model(2 * np.pi * torch.ones_like(x), t)
    loss_bc = torch.mean((u_left - u_right) ** 2)

    # data loss
    u_data = model(x_data, t_data)
    loss_data = torch.mean((u_data - real) ** 2)

    # fp loss
#     u_fp = model(x_fp, t_fp)
#     loss_fp = torch.mean((u_fp - previous_solution) ** 2)

    loss = loss_f + loss_ic + loss_bc + loss_data 

    return loss, loss_f, loss_ic, loss_bc, loss_data
# 主函数
if __name__ == '__main__':
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = torch.load('model/lcctanh20_8.pth')
    # 创建随机训练数据集
    num_samples = 1000
    x_train = (2 * np.pi * torch.rand(num_samples, 1)).to(device)
    x_train.requires_grad = True
    t_train = (9 * torch.rand(num_samples, 1)).to(device)
    t_train.requires_grad = True

    # Adam优化器
    model = PINN()
    model.load_state_dict(model_path)
    model.train()
    model.to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 30000

    data = scipy.io.loadmat('data/lcctanh20_pred8.mat')
    real = data['u']
    x = data['x'].flatten()[:, None]
    t = data['t'].flatten()[:, None]
    real = torch.from_numpy(real)
    X = torch.from_numpy(x)
    T = torch.from_numpy(t)
    x_data = X.to(device, dtype=torch.float32)
    t_data = T.to(device, dtype=torch.float32)
    real = real.to(device, dtype=torch.float32)


#     data = scipy.io.loadmat('data/lcctanh20_fp6.mat')
#     x_fp = data['x']
#     t_fp = data['t']
#     previous_solution = data['u']
#     previous_solution = previous_solution.reshape(-1, 1)
#     previous_solution = torch.from_numpy(previous_solution)
#     previous_solution = previous_solution.to(device, dtype=torch.float32)
#     x_fp = x_fp.reshape(-1, 1)
#     t_fp = t_fp.reshape(-1, 1)
#     x_fp = torch.from_numpy(x_fp)
#     t_fp = torch.from_numpy(t_fp)
#     x_fp = x_fp.to(device)
#     t_fp = t_fp.to(device)
#     x_fp = x_fp.requires_grad_(True)
#     t_fp = t_fp.requires_grad_(True)

    start_time = time.time()
    # 使用Adam训练模型
    losses = []
    pde_losses = []
    ic_losses = []
    bc_losses = []
    data_losses = []
#     fp_losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss, pde_loss, ic_loss, bc_loss, loss_data = compute_loss(model, x_train, t_train, x_data, t_data, real)
        loss.backward()
        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        ic_losses.append(ic_loss.item())
        bc_losses.append(bc_loss.item())
        data_losses.append(loss_data.item())
#         fp_losses.append(loss_fp.item())
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, PDE Loss: {pde_loss.item()}, IC Loss: {ic_loss.item()}, BC Loss: {bc_loss.item()}, Data Loss: {loss_data.item()}')


    # LBFGS优化器
    lbfgs_losses = []
    lbfgs_loss_f = []
    lbfgs_loss_ic = []
    lbfgs_loss_bc = []
    lbfgs_loss_data = []

    iteration = 0  # 初始化迭代计数
    def closure():
        global iteration  # 声明iteration为全局变量，这样我们可以在函数内部修改它
        optimizer_lbfgs.zero_grad()
        loss, pde_loss, ic_loss, bc_loss, loss_data = compute_loss(model, x_train, t_train, x_data, t_data,
                                                                            real)
        lbfgs_losses.append(loss.item())
        lbfgs_loss_f.append(pde_loss.item())
        lbfgs_loss_ic.append(ic_loss.item())
        lbfgs_loss_bc.append(bc_loss.item())
        lbfgs_loss_data.append(loss_data.item())
#         lbfgs_loss_fp.append(loss_fp.item())
        print(
            f"Iteration: {iteration} | Total Loss: {loss.item()} | PDE loss: {pde_loss.item()} | IC loss: {ic_loss.item()} | BC loss: {bc_loss.item()} | Data loss: {loss_data.item()} ")
        loss.backward()
        iteration += 1  # 在每次迭代结束后增加迭代计数
        return loss


    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),  max_iter=65000, max_eval=None, tolerance_grad=0,
                                        tolerance_change=0, history_size=100, line_search_fn=None)
    optimizer_lbfgs.step(closure)
    end_time = time.time()
    print(f"Time elapsed for training: {(end_time - start_time) / 60} min")
    
    Total_losses = losses + lbfgs_losses
    PDE_losses = pde_losses + lbfgs_loss_f
    IC_losses = ic_losses + lbfgs_loss_ic
    BC_losses = bc_losses + lbfgs_loss_bc
    DATA_losses = data_losses + lbfgs_loss_data
#     FP_losses = fp_losses + lbfgs_loss_fp
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.plot(Total_losses, label='Total Loss')
    plt.plot(PDE_losses, label='PDE Loss')
    plt.plot(IC_losses, label='IC Loss')
    plt.plot(BC_losses, label='BC Loss')
    plt.plot(DATA_losses, label='Data Loss')
#     plt.plot(FP_losses, label='FP Loss')
    plt.axvline(x=num_epochs, color='grey', linestyle='--', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Training with Adam and L-BFGS')
    plt.legend()
    plt.savefig('fig/losstanh20_9.png')
    plt.savefig('fig/losstanh20_9.pdf')
    plt.show()
    # 测试模型
    model.eval()
    x = torch.linspace(0, 2 * torch.pi, 1000)
    t = torch.linspace(0, 9, 1000)
    X, T = torch.meshgrid(x, t)
    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    X = X.to(device, dtype=torch.float32)
    T = T.to(device, dtype=torch.float32)
    real_u = torch.sin(X - 20*T)
    uxt = model(X, T)
    real_u = real_u.cpu().detach().numpy()
    uxt = uxt.cpu().detach().numpy()
    L2_error = np.linalg.norm((real_u - uxt).flatten(), 2) / np.linalg.norm(real_u.flatten(), 2)
    print('相对误差: %e' % (L2_error))
    Error = np.mean(np.abs(real_u - uxt))
    print('绝对误差: %e' % (Error))

    # 保存数据
    x = torch.linspace(0, 2 * torch.pi, 1000).to(device)
    t = torch.linspace(0, 9, 1000).to(device)
    x = x.unsqueeze(1)
    t = t.unsqueeze(1)
    solution = model(x, t)
    scipy.io.savemat('data/lcctanh20_pred9.mat',
                     {'x': x.cpu().detach().numpy().reshape(-1, 1), 't': t.cpu().detach().numpy().reshape(-1, 1),
                      'u': solution.cpu().detach().numpy().reshape(-1, 1)})
    t_fp = torch.ones_like(x) * 9
    u_fp = model(x, t_fp)
    scipy.io.savemat('data/lcctanh20_fp9.mat',
                     {'x': x.cpu().detach().numpy().reshape(-1, 1), 't': t_fp.cpu().detach().numpy().reshape(-1, 1),
                      'u': u_fp.cpu().detach().numpy().reshape(-1, 1)})





    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T.cpu().detach().numpy().reshape(1000, 1000), X.cpu().detach().numpy().reshape(1000, 1000),
                   uxt.reshape(1000, 1000), shading='auto', cmap='rainbow')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    # plt.ylim(0, 2*np.pi)
    plt.title('pred')
    plt.savefig('fig/lcctanh_pred_9.png')
    plt.savefig('fig/lcctanh_pred_9.pdf')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T.cpu().detach().numpy().reshape(1000, 1000), X.cpu().detach().numpy().reshape(1000, 1000),
                   real_u.reshape(1000, 1000), shading='auto', cmap='rainbow')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('real')
    plt.savefig('fig/lcctanh_real_9.png')
    plt.savefig('fig/lcctanh_real_9.pdf')
    plt.show()

    save_path = 'model/lcctanh20_9.pth'
    torch.save(model.state_dict(), save_path)

