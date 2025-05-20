import torch

def cifar10_std_and_mean(dataset):
    X = torch.stack([sample[1] for sample in dataset])
    print(X.shape) # should be 50000, 3, 32, 32
    means = torch.mean(X, dim=(0, 2, 3))
    print(means.shape) # should be 3
    stds = torch.std(X, dim=(0, 2, 3))
    print(stds.shape) # should be 3
    return means, stds

cifar10_means = [0.4914, 0.4822, 0.4465]
cifar10_stds = [0.2470, 0.2435, 0.2616]

cifar_label2name = {0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}