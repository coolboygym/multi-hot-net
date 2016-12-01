import numpy as np
import matplotlib.pyplot as plt

# plot train auc and test auc from log file
def plot_result(filename):
    p = []
    q1 = []
    q2 = []
    index = 0
    with open(filename) as fin:
        for line in fin:
            index = (index + 1) % 2
            if index == 1:
                nums = line.strip().split()
                p.append(int(nums[2]))
                q1.append(float(nums[5]))
            else:
                nums = line.replace('[', ' ').replace(']', ' ').split()
                q2.append(float(nums[5]))
    print p
    print q1
    print q2
    p = np.array(p)
    q1 = np.array(q1)
    q2 = np.array(q2)
    return p, q1, q2


if __name__ == '__main__':
    iters, train_loss, valid_loss = plot_result('../log/score')
    plt.figure(1)
    plt.plot(iters, train_loss)
    plt.plot(iters, valid_loss)
    plt.show()