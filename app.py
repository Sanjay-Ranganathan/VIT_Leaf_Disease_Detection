import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms as T
import json

def load_model():
    with open('config.json','r') as f:
        cfg = json.load(f)
    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]

    model = timm.create_model(model_name,pretrained=False,num_classes=num_classes)
    model.load_state_dict(torch.load('leaf_vit.pth',map_location='cpu'))
    model.eval()

    return model,cfg

model, cfg = load_model()
class_names=cfg["class_names"]

mean,std = [0.485,0.456, 0.406], [0.229, 0.224, 0.225]
tfs = T.Compose([
    T.Resize([224,224]),
    T.ToTensor(),
    T.Normalize(mean,std)
])

# -------------------------- UI --------------------------------

st.title('Leaf Disease Detection - ViT')
st.write('Upload a leaf image to classify.')
st.write('NOTE : This model is trained with 147 images due to less GPU power.')
img = st.file_uploader("Upload Image",type=['jpg','jpeg','png'])

if img:
    image = Image.open(img).convert('RGB')
    st.image(image=image, caption="Uploaded Image", use_container_width=True)

    if st.button('Predict'):
        x = tfs(image).unsqueeze(0)
        with torch.no_grad():
            log = model(x)
            probs = torch.softmax(log, dim=1)[0]
        
        ti = probs.argmax().item()

        st.success(f'Prediction : {class_names[ti]}')
        st.write("Probabilities")
        for i,p in enumerate(probs):
            st.write(f"- **{class_names[i]}**: {float(p):.4f}")