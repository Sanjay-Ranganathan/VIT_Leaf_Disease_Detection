from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import timm
from torchvision import transforms as T
import io
from PIL import Image
classes = ['Brinjal Tobacco mosaic virus', 'Lotus Rotting tubers', 'Lotus nutrient deficiency  and rotting tubers', 'brinjal _cercospora leaf spot', 'naval_anthracnose', 'naval_healthy', 'naval_leaf_galls']
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(classes))
model.load_state_dict(torch.load('child_wound_best_model.pth', map_location='cpu'))
model.eval()
tfs = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
app = FastAPI()

@app.post("/upload/predict")
async def predict(file:UploadFile=File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message" : "Only images"})
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    ip = tfs(img).unsqueeze(0)
    with torch.no_grad():
        op = model(ip)
        pc = torch.argmax(op, dim=1).item()
        f = classes[pc]
        return {"message" : "Image successfully received",
                "Predicted" : f"{f}"}


