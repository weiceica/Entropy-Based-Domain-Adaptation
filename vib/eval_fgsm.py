import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import TransferNet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms (no resizing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the adversarial datasets based on the model (VIB or Original)
def load_adversarial_dataset(domain, epsilon, model_type):
    if model_type == 'vib':
        dataset_path = os.path.abspath(f'./{domain}-fgsm/{epsilon}')
    elif model_type == 'original':
        dataset_path = os.path.abspath(f'./{domain}-fgsm-orig/{epsilon}')
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Load the original datasets
def load_original_dataset(domain):
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset', domain))
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Test model on a dataset
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)  # Assuming model returns a tuple
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Evaluate accuracy on original and adversarial datasets
def evaluate_accuracy(model, model_name, domain, model_type):
    # Evaluate on the original dataset
    print(f"\n{model_name} on original {domain} dataset...")
    original_loader = load_original_dataset(domain)
    original_acc = test_model(model, original_loader)
    print(f"Accuracy on original {domain}: {original_acc:.4f}")

    # Evaluate on adversarial datasets based on model type
    epsilons = [0, 0.1, 0.2]
    for epsilon in epsilons:
        epsilon_str = str(epsilon)
        print(f"\n{model_name} on {domain} dataset with epsilon = {epsilon}...")
        adv_loader = load_adversarial_dataset(domain, epsilon_str, model_type)
        adv_acc = test_model(model, adv_loader)
        print(f"Accuracy on {domain} dataset with epsilon = {epsilon}: {adv_acc:.4f}")

# Load the models
def load_model(model_path):
    model = TransferNet(num_class=31, transfer_loss='mmd', base_net='resnet50').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

if __name__ == "__main__":
    # Load both models
    vib_model_path = "trained_model_vib.pth"
    original_model_path = "trained_model_original.pth"
    
    vib_model = load_model(vib_model_path)
    original_model = load_model(original_model_path)

    # Evaluate VIB Model on its respective adversarial datasets
    print("\nEvaluating VIB Model...")
    evaluate_accuracy(vib_model, "VIB Model", "webcam", "vib")
    evaluate_accuracy(vib_model, "VIB Model", "amazon", "vib")

    # Evaluate Original Model on its respective adversarial datasets
    print("\nEvaluating Original Model...")
    evaluate_accuracy(original_model, "Original Model", "webcam", "original")
    evaluate_accuracy(original_model, "Original Model", "amazon", "original")
