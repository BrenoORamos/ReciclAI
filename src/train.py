import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets, transforms
import os

# Caminho dos dados e hiperparâmetros
DATA_DIR = "processed_data/"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001  # Reduzindo a taxa de aprendizado
PATIENCE = 10

# Inicialize o SummaryWriter
writer = SummaryWriter("logs/treinamento")

# Transformações para pré-processamento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=DATA_DIR + "/train", transform=transform)
val_dataset = datasets.ImageFolder(root=DATA_DIR + "/val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Treinamento em:", device)

def criar_modelo():
    model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model.to(device)

def train_model(model, optimizer, criterion, train_loader, val_loader):
    global best_val_loss, patience_counter

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss de Treinamento: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Treinamento", avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Loss de Validação: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Validação", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }
            torch.save(checkpoint, "models/reciclai_model_best.pth")
            print("Modelo melhorado e salvo!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping aplicado.")
                break

    writer.close()

if __name__ == "__main__":
    while True:
        best_val_loss = float('inf')
        patience_counter = 0

        if os.path.exists("models/reciclai_model_best.pth"):
            model = criar_modelo()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            checkpoint = torch.load("models/reciclai_model_best.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Modelo carregado do último ponto de verificação.")
        else:
            model = criar_modelo()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        criterion = nn.CrossEntropyLoss()
        train_model(model, optimizer, criterion, train_loader, val_loader)
        print("Treinamento concluído. Iniciando novo treinamento...")
