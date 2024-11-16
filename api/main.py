from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Definindo o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo treinado com o ajuste para evitar erros de carregamento
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Ajustando para 4 classes
    checkpoint = torch.load("models/reciclai_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Definindo as classes de rótulo
CLASSES = ["metal", "papel", "plastico", "vidro"]

# Transformações para pré-processamento da imagem
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    # Ler a imagem carregada
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Pré-processamento da imagem
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Fazer a inferência
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = CLASSES[predicted.item()]

    return JSONResponse(content={"class": class_name})

# Endpoint de status para verificar se a API está ativa
@app.get("/")
def read_root():
    return {"status": "API está funcionando"}

