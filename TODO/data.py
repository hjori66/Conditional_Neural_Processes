import torch
import random
import numpy as np
import matplotlib.pyplot as plt


#-------------------------- 1D DATA ----------------------------

def generate_data(num_data=20, x_range=(-3, 3), std=3.):
    # train data    
    x_train = [[np.random.uniform(*x_range)] for _ in range(num_data)]
    y_train = [[x[0]**3 + np.random.normal(0, std)] for x in x_train]

    # test data
    x_test = np.linspace(-6, 6, 100).reshape(100, 1)  # test data for regression
    y_test = x_test**3

    return x_train, y_train, x_test, y_test


def draw_graph(x, y, x_set, y_set, mean_predict, std, pic_name="result.png"):  # x-s
    plt.plot(x, y, 'b-', label="Ground Truth")
    plt.plot(x_set, y_set, 'ro', label='data points')
    plt.plot(x, mean_predict, label='MLPs (MSE)', color='grey')
    plt.fill_between(x.reshape(-1), (mean_predict-3*std).reshape(100,), (mean_predict+3*std).reshape(100,),color='grey',alpha=0.3)

    plt.legend()
    # plt.savefig('CNP_5shot_100k_' + pic_name)
    plt.show()


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_sinoid_data(num_data=20, x_range=(-4.5, 4.5), std=3.):
    # train data
    amplitude = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0.0, 2*np.pi)
    x_train = np.array([[np.random.uniform(*x_range)] for _ in range(num_data)])
    y_train = amplitude * np.sin(x_train + phase)

    # test data
    x_test = np.array(np.linspace(-6, 6, 100).reshape(100, 1))  # test data for regression
    y_test = amplitude * np.sin(x_test + phase)

    # return torch.Tensor(x).unsqueeze(1), torch.Tensor(y).unsqueeze(1)
    return x_train, y_train, x_test, y_test


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
