import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import time
from model import TransferModel, ResNet50Fc, TransferNet
from loss import MMD_loss, CORAL
from torchvision import models

torch.cuda.set_device(1)

# Data
data_folder = 'dataset'
batch_size = 32
n_class = 31
domain_src, domain_tar = 'dslr', 'amazon'

# Data loader
def load_data(root_path, domain, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=os.path.join(root_path, domain), transform=transform_dict[phase])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4)
    return data_loader

src_loader = load_data(data_folder, domain_src, batch_size, phase='src')
tar_loader = load_data(data_folder, domain_tar, batch_size, phase='tar')

model = TransferModel().cuda()
RAND_TENSOR = torch.randn(1, 3, 224, 224).cuda()
output = model(RAND_TENSOR)

dataloaders = {'src': src_loader,
               'val': tar_loader,
               'tar': tar_loader}
n_epoch = 100
criterion = nn.CrossEntropyLoss()
early_stop = 100

# Test function
def test(model, target_test_loader):
    model.eval()
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = correct.double() / len(target_test_loader.dataset)
    return acc

# KL Divergence calculation function
def calculate_kl_divergence(mu1, log_var1, mu2, log_var2):
    kl_div = -0.5 * torch.sum(1 + log_var1 - log_var2 - mu1.pow(2) - log_var1.exp() / log_var2.exp() - (mu1 - mu2).pow(2) / log_var2.exp())
    return kl_div

transfer_loss = 'mmd'
learning_rate = 0.0001
beta = 0.0005  # Beta value for KL divergence loss
transfer_model = TransferNet(n_class, transfer_loss=transfer_loss, base_net='resnet50').cuda()
optimizer = torch.optim.SGD([
    {'params': transfer_model.base_network.parameters()},
    {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * learning_rate},
    {'params': transfer_model.classifier_layer.parameters(), 'lr': 10 * learning_rate},
], lr=learning_rate, momentum=0.9, weight_decay=5e-4)
lamb = 0.5 # Weight for transfer loss, it is a hyperparameter that needs to be tuned

# Training function
def train(dataloaders, model, optimizer):
    source_loader, target_train_loader, target_test_loader = dataloaders['src'], dataloaders['val'], dataloaders['tar']
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    best_acc = 0
    stop = 0
    n_batch = min(len_source_loader, len_target_loader)
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_kl, train_loss_total = 0, 0, 0, 0
        model.train()
        for (src, tar) in zip(source_loader, target_train_loader):
            data_source, label_source = src
            data_target, _ = tar
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()

            optimizer.zero_grad()
            label_source_pred, transfer_loss, mu_input, log_var_input, mu_bottleneck, log_var_bottleneck = model(data_source, data_target, return_gaussian_params=True) # Domain loss
            clf_loss = criterion(label_source_pred, label_source) # Source classification loss
            kl_loss_input_to_bottleneck = calculate_kl_divergence(mu_input, log_var_input, mu_bottleneck, log_var_bottleneck) # KL Divergence loss
            loss = clf_loss + lamb * transfer_loss + beta * kl_loss_input_to_bottleneck # Total loss function
            loss.backward()
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() + train_loss_clf
            train_loss_transfer = transfer_loss.detach().item() + train_loss_transfer
            train_loss_kl = kl_loss_input_to_bottleneck.detach().item() + train_loss_kl
            train_loss_total = loss.detach().item() + train_loss_total
        
        acc = test(model, target_test_loader)
        print(f'Epoch: [{e:2d}/{n_epoch}], cls_loss: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, kl_loss: {train_loss_kl/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), 'trans_model.txt')
            stop = 0
        if stop >= early_stop:
            break

if __name__ == "__main__":
    train(dataloaders, transfer_model, optimizer)
