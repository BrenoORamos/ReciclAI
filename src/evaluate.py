# src/evaluate.py
import torch
from torchvision import models
import torch.nn as nn
from data_loader import get_data_loaders
from sklearn.metrics import classification_report

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    # Carrega apenas o `state_dict` do modelo sem os metadados
    checkpoint = torch.load("models/reciclai_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Use apenas a parte do `state_dict`
    
    model = model.to(device)
    model.eval()

    _, test_loader = get_data_loaders()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model()
