import streamlit as st
from PIL import Image
import requests

st.title("Leaf Disease Prediction")
image = st.file_uploader('Upload Image',type=['jpg','jpeg','png'])
if image is not None:
    img = Image.open(image)
    st.image(image, caption='uploaded image', use_container_width=True)
    files = {
        "file" : (image.name, image, image.type)
    }
    response = requests.post("http://127.0.0.1:8000//upload/predict", files=files)
    if response.status_code == 200 :
        st.success(response.json().get("Predicted"))
    else:
        st.error(f"error : {response.text}")