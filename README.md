# VIT_Leaf_Disease_Detection
This Repo Uses Vision Transformer with custom dataset of only 147 images and achieves a 99% accuracy.
Steps to Run the model :
1. Upload the dataset in gdrive.
2. Generate a token in Hugging Face.
3. Create the .env file and store the token
4. Modify the path and token fetching in code.
5. You're ready to train the model.
6. Try with Multiple datsets and explore more...

If you are done with this steps, Download the best performed model(.pth file) from saved model in colab.

# Streamlit App
1. After downloaded the best_model.pth, Move it to repo.
2. Now You can install necessary packages and start the model_api.py
  -> It is FastAPI, You can start this using this command
      uvicorn model_api:app --reload
3. The backend is ready, Now start the Frontend App, which is App.py
  -> It is streamlit web application, You can start this using this command
       streamlit run app.py
4. It automatically opens your web application, upload the leaf diseased photo and predict it.
