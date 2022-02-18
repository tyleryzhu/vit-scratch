import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim
import argparse
from tqdm import tqdm

from vit import ViT

train_transform = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

val_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 16
data_dir = "data/"

train_set = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_transform
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=val_transform
)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Parameters for CIFAR-10
patch_size = 8
img_size = [32, 32]
num_layers = 4
embedding_dim = 48
num_heads = 4
attention_mlp_hidden = 48
classify_mlp_hidden = 100
dropout_rate = 0.1
num_classes = 10
epochs = 20

# Derived parameters
assert img_size[0] % patch_size == 0
assert img_size[1] % patch_size == 0
num_patches = img_size[0] * img_size[1] // patch_size ** 2
input_dim = patch_size ** 2 * 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ViT(
    num_layers,
    num_patches,
    input_dim,
    embedding_dim,
    num_heads,
    dropout_rate,
    attention_mlp_hidden,
    classify_mlp_hidden,
    num_classes,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()


def main():
    print(model)
    best_loss = float("inf")
    with open("results/vit_stats.txt", "w") as f:
        f.write(f"epoch\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc\n")
    for epoch in range(epochs):
        # Training
        print(f"...... Training Epoch {epoch+1} ........")
        total_train_loss = 0
        total_train_acc = 0
        train_len = 0
        model.train()
        # model.apply(deactivate_batchnorm)
        for i, data in enumerate(tqdm(train_loader, dynamic_ncols=True), 0):
            inputs, labels = data
            # Reshape inputs into tensor form.
            inputs = inputs.unfold(2, patch_size, patch_size).unfold(
                3, patch_size, patch_size
            )
            inputs = inputs.permute(0, 2, 3, 4, 5, 1)
            inputs = inputs.contiguous().view(inputs.size(0), num_patches, input_dim)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # write losses, account for batch difference
            b = labels.shape[0]
            total_train_loss += loss.item() * b
            total_train_acc += torch.sum((torch.argmax(output, dim=1) == labels)).item()
            train_len += b
        total_train_loss /= train_len  # dataset size
        total_train_acc /= train_len

        # Testing
        total_test_loss = 0
        total_test_acc = 0
        test_len = 0
        model.eval()
        for i, data in enumerate(tqdm(test_loader, dynamic_ncols=True)):
            inputs, labels = data
            # Reshape
            inputs = inputs.unfold(2, patch_size, patch_size).unfold(
                3, patch_size, patch_size
            )
            inputs = inputs.permute(0, 2, 3, 4, 5, 1)
            inputs = inputs.contiguous().view(inputs.size(0), num_patches, input_dim)
            optimizer.zero_grad()
            with torch.no_grad():
                output = model(inputs)
            loss = criterion(output, labels)

            b = labels.shape[0]
            total_test_loss += loss.item() * b
            total_test_acc += torch.sum((torch.argmax(output, dim=1) == labels)).item()
            test_len += b
        total_test_loss /= test_len  # dataset size
        total_test_acc /= test_len  # dataset size

        if total_test_loss < best_loss:
            best_loss = total_test_loss
            torch.save(model.state_dict(), "cifar_best_model.pt")

        print(
            f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_train_loss:.4f}, Testing Loss: {total_test_loss:.4f},"
            f" Training Accuracy: {total_train_acc:.4f}, Testing Accuracy: {total_test_acc:.4f}"
        )
        with open("results/vit_stats.txt", "a") as f:
            f.write(
                f"{epoch+1}\t{total_train_loss:.4f}\t{total_train_acc:.4f}\t{total_test_loss:.4f}\t{total_test_acc:.4f}\n"
            )


if __name__ == "__main__":
    main()

