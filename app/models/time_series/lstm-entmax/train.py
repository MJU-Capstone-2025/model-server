import torch
import torch.nn as nn
from config import DEVICE

def weighted_mse_loss(y_pred, y_true, temp=5.0):
    errors = (y_pred - y_true) ** 2
    weights = torch.softmax(errors * temp, dim=0)
    return torch.sum(weights * errors)

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    loss_fn = weighted_mse_loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_seq, x_static, y in train_loader:
            x_seq, x_static, y = x_seq.to(DEVICE), x_static.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_seq, x_static)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_seq, x_static, y in val_loader:
                x_seq, x_static, y = x_seq.to(DEVICE), x_static.to(DEVICE), y.to(DEVICE)
                out = model(x_seq, x_static)
                val_loss += nn.MSELoss()(out, y).item()

        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")