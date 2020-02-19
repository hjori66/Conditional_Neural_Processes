from model import Encoder, Decoder, CNPs, NLLloss
from data import generate_data, draw_graph, generate_sinoid_data

import argparse
import numpy as np
import torch

# from torchviz import make_dot


SEED = 1234
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    # x_train, y_train, x_test, y_test = generate_sinoid_data(num_data=batch_size)
    # x_torch, y_torch = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
    # x_test_torch = torch.Tensor(x_test).to(device)
    
    # --------- ARGS --------- 
    epochs = args.epochs
    batch_size = args.batch_size
    ctx_size = args.ctx_size
    lr = args.lr
    print_step = args.print_step

    epsilon = 0.01

    # --------- MODEL --------- 
    x_dim = 1
    y_dim = 1
    r_dim = 128  # 128

    encoder = Encoder(x_dim, y_dim, r_dim).to(device)
    decoder = Decoder(x_dim, r_dim, y_dim).to(device)
    net = CNPs(encoder=encoder, decoder=decoder).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 0.01 and 10000 epochs!
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # 0.01 and 10000 epochs!

    # --------- TRAIN --------- 
    for epoch in range(epochs):
        x_train, y_train, x_test, y_test = generate_sinoid_data(num_data=batch_size)
        x_torch, y_torch = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)

        optimizer.zero_grad()
        
        mean, std = net(x_torch[:ctx_size], y_torch[:ctx_size], x_torch)
        # mean, variance = net(x_torch[:ctx_size], y_torch[:ctx_size], x_torch)
        nll_loss = NLLloss(y_torch, mean, std*std)

        if epoch % print_step == 0:
            print('Epoch', epoch, ': nll loss', nll_loss.item())
        # dot = make_dot(mean)
        # dot.render("model.png")

        nll_loss.backward()
        optimizer.step()

    print("final loss : nll loss", nll_loss.item())
    for i in range(3):
        x_train, y_train, x_test, y_test = generate_sinoid_data(num_data=batch_size)
        x_torch, y_torch = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
        x_test_torch = torch.Tensor(x_test).to(device)

        result_mean, result_std = net(x_torch, y_torch, x_test_torch)
        # result_mean, result_var = net(x_torch, y_torch, x_test_torch)
        result_mean, result_std = result_mean.cpu().detach().numpy(), result_std.cpu().detach().numpy()
        # result_mean, result_var = result_mean.cpu().detach().numpy(), result_var.cpu().detach().numpy()
        print(result_mean, result_std)
        draw_graph(x_test, y_test, x_train, y_train, result_mean, np.sqrt(result_std*result_std), pic_name=str(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ctx_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--print_step', type=int, default=500)

    args = parser.parse_args()
    main(args)
