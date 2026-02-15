import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src.dataset import get_dataloaders
from src.model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main(epochs=2, batch_size=64, lr=1e-3):
    """
    Simple but professional baseline training with MLflow tracking.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = get_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("cifar10_resnet18")

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("device", device)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            print(f"Epoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Save model weights locally
        torch.save(model.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")


if __name__ == "__main__":
    main()
