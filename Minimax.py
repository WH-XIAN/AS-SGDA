import argparse
import os
from sklearn.datasets import load_svmlight_file
import numpy as np
import time

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Adaptive Stepsize Minimax Algorithm')
parser.add_argument('--dataset', default='a9a', type=str, help='name of dataset')
parser.add_argument('--lambda2', default=0.001, type=float, help='coefficient of regularization')
parser.add_argument('--alpha', default=10.0, type=float, help='coefficient in regularization')
parser.add_argument('--sigma', default=1.0, type=float, help='standard deviation of initialization')
parser.add_argument('--adaptive', action='store_true', help='whether to use adaptive stepsize')
parser.add_argument('--lr_x', default=0.01, type=float, help='learning rate for x')
parser.add_argument('--lr_y', default=0.1, type=float, help='learning rate for y')
parser.add_argument('--bx', default=50, type=int, help='mini batch size for x')
parser.add_argument('--by', default=50, type=int, help='mini batch size for y')
parser.add_argument('--num_epochs', default=50000, type=int, help='number of epochs to train')
parser.add_argument('--print_freq', default=100, type=int, help='frequency to print train stats')
parser.add_argument('--out_fname', default='', type=str, help='path of output file')
# --------------------------------------------------------------------------- #

libsvm_root = '/nfshomes/wxian1/data/libsvm/binary/'

libsvm_description = {
    'a9a': {'name': 'a9a', 'feature': 123, 'sample': 32561, 'label': [-1, 1]},
    'covtype': {'name': 'covtype_scale', 'feature': 54, 'sample': 581012, 'label': [1, 2]},
    'diabetes': {'name': 'diabetes_scale', 'feature': 8, 'sample': 768, 'label': [-1, 1]},
    'german': {'name': 'german.numer_scale', 'feature': 24, 'sample': 1000, 'label': [-1, 1]},
    'gisette': {'name': 'gisette_scale', 'feature': 5000, 'sample': 6000, 'label': [-1, 1]},
    'ijcnn1': {'name': 'ijcnn1', 'feature': 22, 'sample': 141691, 'label': [-1, 1]},
    'mushrooms': {'name': 'mushrooms', 'feature': 112, 'sample': 8124, 'label': [1, 2]},
    'phishing': {'name': 'phishing', 'feature': 68, 'sample': 11055, 'label': [0, 1]},
    'real-sim': {'name': 'real-sim', 'feature': 20958, 'sample': 72309, 'label': [-1, 1]},
    'w8a': {'name': 'w8a', 'feature': 300, 'sample': 49749, 'label': [-1, 1]},
    'webspam_u': {'name': 'webspam_u', 'feature': 254, 'sample': 350000, 'label': [-1, 1]}
}

def libsvm_loader(filename):
    source = os.path.join(libsvm_root, libsvm_description[filename]['name'])
    data = load_svmlight_file(source)
    x_raw = data[0]
    y = np.array(data[1])
    # labels should be +1 or -1
    if libsvm_description[filename]['label'][0] == 1 and libsvm_description[filename]['label'][1] == 2:
        y = 2 * y - 3
    if libsvm_description[filename]['label'][0] == 0 and libsvm_description[filename]['label'][1] == 1:
        y = 2 * y - 1
    # add bias
    x = np.ones([x_raw.shape[0], x_raw.shape[1] + 1])
    x[:, :-1] = x_raw.todense()
    return x, y


def cal_gradient_x(data, label, x, y, n, lambda2, alpha):
    term1 = np.exp(- np.sum(data * x, axis=1) * label)
    term2 = - y * label * term1 / (1 + term1)
    # grad1 = np.matmul(term2, data) / data.shape[0]
    grad1 = np.matmul(term2, data) * n / len(y)
    denominator = (1 + alpha * x * x) * (1 + alpha * x * x)
    numerator = 2 * lambda2 * alpha * x
    grad2 = numerator / denominator
    return grad1 + grad2


def cal_gradient_y(data, label, x, y, idx, n):
    logistic_loss = np.log(1 + np.exp(- np.sum(data * x, axis=1) * label)) * n / len(idx)
    grad = np.ones(n) * 1.0 / n
    grad[idx] += logistic_loss
    return grad - y


def projection(y, n):
    y_sort = np.sort(y)
    y_sort = y_sort[::-1]
    sum_y = 0
    t = 0
    for i in range(n):
        sum_y = sum_y + y_sort[i]
        t = (sum_y - 1.0) / (i + 1)
        if i < n - 1 and y_sort[i + 1] <= t < y_sort[i]:
            break
    return np.maximum(y - t, 0)


def cal_phi(data, label, x, n, lambda2, alpha):
    logistic_loss = np.log(1 + np.exp(- np.sum(data * x, axis=1) * label))
    un = np.ones(n) * 1.0 / n
    y_star = projection(logistic_loss + un, n)
    phi = np.inner(y_star, logistic_loss) - 0.5 * np.sum((y_star - un) * (y_star - un)) + lambda2 * \
          np.sum(alpha * x * x / (alpha * x * x + 1))
    grad = cal_gradient_x(data, label, x, y_star, n, lambda2, alpha)
    grad_norm = np.sqrt(np.sum(grad * grad))
    return phi, grad_norm


def main():
    args = parser.parse_args()
    data, label = libsvm_loader(args.dataset)
    n, d = data.shape
    alpha = args.alpha
    lambda2 = args.lambda2

    # x = np.random.normal(0, args.sigma, d)
    x = np.ones(d) * args.sigma
    y = np.ones(n) * 1.0 / n

    if args.adaptive:
        out_fname = 'result_' + args.dataset + '_adaptive.csv'
    else:
        lrstr = str(args.lr_x)[2:]
        out_fname = 'result_' + args.dataset + '_' + lrstr + '.csv'

    phi, grad_norm = cal_phi(data, label, x, n, lambda2, alpha)
    if not os.path.exists(out_fname):
        with open(out_fname, 'w') as f:
            print('{ep:d},{t:.3f},{ifo:d},{phi:.5f},{grad:.5f}'.format(ep=0, t=0, ifo=0, phi=phi, grad=grad_norm), file=f)

    elapsed_time = 0.0
    oracle = 0
    scale = 1
    for epoch in range(args.num_epochs):
        t_begin = time.time()

        idx = np.random.randint(0, n, args.bx)
        idy = np.random.randint(0, n, args.by)
        gx = cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha)
        gy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n)
        oracle = oracle + args.bx + args.by

        if args.adaptive:
            scale = max(np.sqrt(np.sum(gx * gx)), 0.01)

        x = x - (args.lr_x / scale) * gx
        y = y + args.lr_y * gy
        y = projection(y, n)

        t_end = time.time()
        elapsed_time += (t_end - t_begin)
        if (epoch + 1) % args.print_freq == 0:
            phi, grad_norm = cal_phi(data, label, x, n, lambda2, alpha)
            with open(out_fname, '+a') as f:
                print('{ep:d},{t:.3f},{ifo:d},{phi:.5f},{grad:.5f}'
                      .format(ep=epoch + 1, t=elapsed_time, ifo=oracle, phi=float(phi), grad=grad_norm), file=f)


if __name__ == '__main__':
    main()
