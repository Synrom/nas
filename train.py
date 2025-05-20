import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.simple import Lenet5
from dataset.utils import cifar10_means, cifar10_stds, cifar10_std_and_mean
from dataset.wrapper import cifar10
from monitor import Monitor
from utils import clone_model, compare_models

# set random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 2

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar10_means, std=cifar10_stds)])
eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar10_means, std=cifar10_stds)])

train_dataset = cifar10(train=True, transform=train_transform, target_transform=None, download=True)
test_dataset = cifar10(train=False, transform=eval_transform, target_transform=None, download=True)

batch_size = 16
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Lenet5()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
monitor = Monitor(model, test_dataset=test_dataset, device=device, criterion=criterion, debug=True)

def validate_model(model: nn.Module, criterion: nn.Module, monitor: Monitor):
    model.eval()
    losses, accs, topk_accs = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(valid_data_loader):
            imgs, X, Y = batch
            X,Y = X.to(device), Y.to(device)
            out = model(X) 
            
            # loss
            loss = criterion(out, Y)
            assert out.shape == torch.Size([batch_size, 10])
            losses.append(loss)

            # accuracy
            preds = out.argmax(dim=1)
            assert preds.shape == Y.shape
            acc = torch.sum(preds == Y) / Y.shape[0]
            accs.append(acc)

            # topk-accuracy
            _, topk_preds = out.topk(k=5, dim=1)
            assert topk_preds.shape == torch.Size([batch_size, 5])
            topk_acc = 0.0
            for i, p in enumerate(topk_preds):
                topk_acc += float(p in Y[i])
            topk_acc /= Y.shape[0]
            topk_accs.append(topk_acc)
    mean_loss = np.array(losses).mean()
    mean_acc = np.array(accs).mean()
    mean_topk_acc = np.array(topk_accs).mean()
    monitor.add_validation_loss(mean_loss, mean_acc, mean_topk_acc)
    return mean_loss, mean_acc, mean_topk_acc

def train(model: nn.Module, criterion: nn.Module, monitor: Monitor):
    model.eval()
    validate_model(model, criterion, monitor)
    model.train()
    for epoch in range(num_epochs):
        for idx, batch in enumerate(data_loader):
            imgs, X, Y = batch
            assert Y.shape == torch.Size([batch_size])
            X,Y = X.to(device), Y.to(device)
            out = model(X) 
            assert out.shape == torch.Size([batch_size, 10])
            loss = criterion(out, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == 0 and idx == 0:
                monitor.first_batch(imgs, X, Y, loss.item())
                monitor.test_data_sharing_inbetween_batch(model, X)
                bf_model, bf_criterion = clone_model(model), clone_model(criterion)
                monitor.overfit_single_batch(model, X, Y, criterion, optimizer, 100)
                assert compare_models(bf_model, model)
                assert compare_models(bf_criterion, criterion)
                del bf_model
                del bf_criterion
            monitor.add_training_loss(loss.item())
            if (idx+1) % 780 == 0: # 4 times per epoch
                model.eval()
                monitor.eval_test_batch(model, f"After {epoch} epochs and {idx} steps") 
                validate_model(model, criterion, monitor)
                model.train()
        monitor.end_epoch()
        model.eval()
        monitor.input_dependent_baseline(model, criterion)
        model.train()

train(model, criterion, monitor)
validate_model(model, criterion, monitor)