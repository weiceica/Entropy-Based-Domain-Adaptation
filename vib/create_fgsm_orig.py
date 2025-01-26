import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import TransferNet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
model = TransferNet(num_class=31, transfer_loss='mmd', base_net='resnet50')
model.load_state_dict(torch.load("trained_model_original.pth", map_location=device))
model = model.to(device)
model.eval()

# Define data transforms (ensure no resizing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the datasets
data_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
amazon_loader = datasets.ImageFolder(root=os.path.join(data_folder, 'amazon'), transform=transform)
webcam_loader = datasets.ImageFolder(root=os.path.join(data_folder, 'webcam'), transform=transform)

# Dataloader setup
def get_loader(dataset, batch_size=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

amazon_loader = get_loader(amazon_loader)
webcam_loader = get_loader(webcam_loader)

# FGSM attack function
def fgsm_attack(model, images, labels, eps):
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    # Forward pass
    outputs, _ = model(images)

    # Calculate loss
    loss = F.cross_entropy(outputs, labels)
    
    # Zero the gradients
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Apply FGSM attack
    perturbed_images = images + eps * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images

# Save perturbed images in respective folders
def save_adv_images(perturbed_data, original_paths, epsilon_str, domain):
    base_save_dir = f"./{domain}-fgsm-orig/{epsilon_str}/"
    for idx, image in enumerate(perturbed_data):
        original_path = original_paths[idx]
        label_folder = os.path.dirname(original_path).split(os.sep)[-1]  # Keep folder structure intact
        img_name = os.path.basename(original_path)  # Original image name
        save_folder = os.path.join(base_save_dir, label_folder)
        os.makedirs(save_folder, exist_ok=True)
        save_image(image, os.path.join(save_folder, img_name))

# Apply FGSM attack to the dataset
def apply_fgsm_and_save(loader, epsilon, domain, epsilon_str):
    for batch_idx, (images, labels) in enumerate(loader):
        perturbed_data = fgsm_attack(model, images, labels, epsilon)
        
        # Get the original image paths to preserve folder structure
        image_paths = [loader.dataset.samples[i][0] for i in range(batch_idx * loader.batch_size, (batch_idx + 1) * loader.batch_size)]
        
        # Save perturbed images in respective folder
        save_adv_images(perturbed_data, image_paths, epsilon_str, domain)

# Run FGSM for different epsilons
epsilons = [0, 0.1, 0.2]
for epsilon in epsilons:
    epsilon_str = str(epsilon)
    print(f"Applying FGSM with epsilon = {epsilon} on webcam dataset")
    apply_fgsm_and_save(webcam_loader, epsilon, domain='webcam', epsilon_str=epsilon_str)

    print(f"Applying FGSM with epsilon = {epsilon} on amazon dataset")
    apply_fgsm_and_save(amazon_loader, epsilon, domain='amazon', epsilon_str=epsilon_str)

print("FGSM attack completed and images saved.")
