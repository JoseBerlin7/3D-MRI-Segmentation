# train.py

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, decollate_batch

from model_code import get_model
from data_loaders import get_dataloaders

import argparse
import os


def train_swinunetr(model, train_loader, val_loader, num_classes, n_epochs=100, lr=1e-4, weight_decay=1e-5, val_interval=2, device="cpu", ckpt_path=None):

    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    writer = SummaryWriter()

    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch+1}/{n_epochs}")
        print("-" * 30)
        model.train()

        epoch_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["mask"].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                output = model(images)
                loss = loss_fn(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_loss}")

        # Validation
        if ((epoch + 1) % val_interval) == 0:
            model.eval()
            dice_scores = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["mask"].to(device)

                    with autocast(device_type=device.type):
                        output = model(images)

                    output = [post_pred(i) for i in decollate_batch(output)]
                    labels = [post_label(i) for i in decollate_batch(labels)]

                    dice_metric(y_pred=output, y=labels)
                    dice_scores.append(dice_metric.aggregate().item())
                    dice_metric.reset()

            mean_dice = sum(dice_scores) / len(dice_scores)
            writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Mean Dice: {mean_dice}")
            print("---" * 30, end="\n")

            if ckpt_path:
                ckpt_name = f"/{ckpt_path}/epoch_{epoch+1}_dice_{mean_dice:.4f}.pth"
                torch.save(model.state_dict(), ckpt_name)

    writer.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/data/brats")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader = get_dataloaders(args.data_path)

    model = get_model().to(device)

    train_swinunetr(
        model,
        train_loader,
        val_loader,
        num_classes=4,
        n_epochs=args.epochs,
        val_interval=2,
        device=device,
        ckpt_path="weights"
    )
