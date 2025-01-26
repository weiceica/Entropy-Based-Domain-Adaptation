import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model import TransferNet
from loss import MMD_loss, CORAL
from torchvision import datasets, transforms

torch.cuda.set_device(1)

# Path to your dataset (adjust according to your actual dataset path)
data_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
domain_src = 'webcam'  # Source domain
domain_tar = 'amazon'  # Target domain

# Load Data
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
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4)

# Train model with logging for each epoch
def train(dataloaders, model, optimizer, lamb=0.5):
    source_loader = dataloaders['src']
    target_loader = dataloaders['tar']
    best_acc = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss, total_clf_loss, total_transfer_loss = 0, 0, 0
        for data_source, label_source in source_loader:
            data_target, _ = next(iter(target_loader))  # Load target data as well
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()

            optimizer.zero_grad()
            # Pass both source and target data to the model
            source_bottleneck, transfer_loss = model(data_source, data_target)

            # Classification loss
            clf_loss = F.cross_entropy(source_bottleneck, label_source)
            
            # Handle the case where transfer_loss is None
            if transfer_loss is not None:
                loss = clf_loss + lamb * transfer_loss
                total_transfer_loss += transfer_loss.item()
            else:
                loss = clf_loss  # In case transfer_loss is not calculated

            loss.backward()
            optimizer.step()

            total_clf_loss += clf_loss.item()
            total_loss += loss.item()

        # Average losses for the epoch
        avg_loss = total_loss / len(source_loader)
        avg_clf_loss = total_clf_loss / len(source_loader)
        avg_transfer_loss = total_transfer_loss / len(source_loader)

        # Testing accuracy on the target domain
        acc = test(model, target_loader)

        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Clf Loss: {avg_clf_loss:.4f}, Transfer Loss: {avg_transfer_loss:.4f}, Accuracy: {acc:.4f}')

        # Check for the best accuracy
        if acc > best_acc:
            best_acc = acc
            print(f'Best Accuracy improved to: {best_acc:.4f}')


# Test model on clean data
def test(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            # Since the model returns a tuple (source_bottleneck, transfer_loss), we only need source_bottleneck during testing
            s_output, _ = model(data)
            pred = s_output.max(1, keepdim=True)[1]  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
    acc = correct / len(dataloader.dataset)
    return acc


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    n_class = 31
    n_epochs = 100
    learning_rate = 0.0001
    lamb = 0.5  # Weight for transfer loss (MMD/CORAL)

    # Initialize model, optimizer, and dataloaders
    transfer_model = TransferNet(num_class=n_class, transfer_loss='mmd', base_net='resnet50').cuda()
    optimizer = torch.optim.SGD(transfer_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Load your dataset (data path retained)
    dataloaders = {
        'src': load_data(data_folder, domain_src, batch_size, phase='src'),
        'tar': load_data(data_folder, domain_tar, batch_size, phase='tar')
    }

    # Train the model on the original dataset
    print("Training the model...")
    train(dataloaders, transfer_model, optimizer)

    # Save the trained model after training
    model_path = './trained_model_original_original.pth'  # Specify the path where you want to save the model
    torch.save(transfer_model.state_dict(), model_path)
    print(f"Model has been saved to '{model_path}'.")
