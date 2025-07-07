from torch import nn
import os
import torch
import data, engine, model, utils
from torchvision import transforms
from tqdm.auto import tqdm


NUM_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = "./data/train"
test_dir = "./data/test"
device = "cuda" if torch.cuda.is_available() else "cpu"


data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])



def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    
    model.to(device)
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=-1), dim=-1)
        train_acc += (y_pred_class == y).sum().item() / len(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc /= len(dataloader)
    train_loss /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.to(device)
    model.eval()

    test_loss, test_acc = 0 , 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=-1), dim=-1)
            test_acc += (test_pred_labels == y).sum().item() / len(y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device
) -> dict[str, list]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        print(
            f"Epoch: {epoch}"
            f"train_loss: {train_loss}"
            f"train_acc: {train_acc}"
            f"test_loss: {test_loss}"
            f"test_acc: {test_acc}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results



if __name__ == "__main__":
    train_loader, test_loader, class_names = data.create_dataloaders(
        train_dir, test_dir, data_transforms,
        batch_size=BATCH_SIZE
    )

    model0 = model.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model0.parameters(), lr=LEARNING_RATE)
    train(
        model=model0,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        optimizer=optimizer
    )
    utils.save_model(
        model=model0,
        target_dir="./models",
        model_name="going_modular_tiny_vgg_model.pth"
    )