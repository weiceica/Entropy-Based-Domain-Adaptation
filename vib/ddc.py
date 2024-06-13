import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import time
from model import TransferModel, TransferNet
from loss import MMD_loss, CORAL, VIB_loss
from torchvision import models

torch.cuda.set_device(1)

# Data
data_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
batch_size = 32
n_class = 31
domain_src, domain_tar = 'webcam', 'amazon'

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

# Initialize model
transfer_model = TransferNet(n_class, transfer_loss='mmd', base_net='resnet50').cuda()

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
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = correct.double() / len(target_test_loader.dataset)
    return acc

learning_rate = 0.0001
beta = 0.0005  # Beta value for VIB loss
optimizer = torch.optim.SGD([
    {'params': transfer_model.base_network.parameters()},
    {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * learning_rate}
], lr=learning_rate, momentum=0.91, weight_decay=5e-4)
lamb = 0.5 # Weight for transfer loss, it is a hyperparameter that needs to be tuned

# Training function
def train(dataloaders, model, optimizer, beta=0.0005, lamb=0.5):
    source_loader, target_train_loader, target_test_loader = dataloaders['src'], dataloaders['val'], dataloaders['tar']
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    best_acc = 0
    stop = 0
    n_batch = min(len_source_loader, len_target_loader)
    vib_loss_fn = VIB_loss(beta=beta)
    
    for e in range(n_epoch):
        stop += 1
        train_loss_transfer, train_loss_kl, train_loss_total, train_loss_vib = 0, 0, 0, 0
        model.train()
        for (src, tar) in zip(source_loader, target_train_loader):
            data_source, label_source = src
            data_target, _ = tar
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()

            optimizer.zero_grad()
            source_bottleneck, transfer_loss, mu, log_var = model(data_source, data_target, return_gaussian_params=True)
            vib_loss, _, kl_divergence = vib_loss_fn(mu, log_var, data_source, label_source, source_bottleneck)
            loss = vib_loss + lamb * transfer_loss
            loss.backward()
            optimizer.step()
            
            train_loss_transfer += transfer_loss.detach().item() if transfer_loss else 0
            train_loss_kl += kl_divergence.detach().item()
            train_loss_vib += vib_loss.detach().item()
            train_loss_total += loss.detach().item()
        
        acc = test(model, target_test_loader)
        print(f'Epoch: [{e:2d}/{n_epoch}], transfer_loss: {train_loss_transfer/n_batch:.4f}, kl_loss: {train_loss_kl/n_batch:.4f}, vib_loss: {train_loss_vib/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        if best_acc < acc:
            best_acc = acc
            stop = 0
        if stop >= early_stop:
            break

if __name__ == "__main__":
    train(dataloaders, transfer_model, optimizer)
