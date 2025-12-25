import torch
from torch.utils.data import DataLoader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def train_model(model, dataset, criterion, epochs=500, batch_size=256,
                lr=1e-3, device='cpu', verbose=True, log_every=50):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        losses.append(loss)

        if verbose and (epoch + 1) % log_every == 0:
            print(f'Epoch {epoch+1:3d}/{epochs}, Loss: {loss:.4f}')
    
    return losses

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model