import os
import torch
import torch.nn as nn
import librosa
from fastapi import FastAPI, UploadFile
from io import BytesIO
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to save the fitted StandardScaler
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# Configure CORS to allow requests from your Flutter app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to only allow your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the path to your fitted StandardScaler
scaler_path = "app/utils/scaler.pkl"



def extract_features(audio_path, mfcc=True, chroma=True, mel=True):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=1)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate), axis=1)
        features.extend(mel)
    return features


def preprocess_audio(audio_path):
    audio_data, _ = librosa.load(audio_path, sr=None)
    features = extract_features(audio_path)
    # print(features)
    scaler = StandardScaler()
    features = np.array(features).reshape(1, -1)

    scaler.fit(features)  # Fit with the reshaped features
    joblib.dump(scaler, scaler_path)

    return torch.tensor(features, dtype=torch.float32)


@app.get("/")
def read_root():
    return {"Welcome to Kubus Suara Version 1.0"}

@app.post("/upload_audio")
async def upload_audio_file(file: UploadFile):
    if not file:
        return JSONResponse(content={"error": "No file provided"}, status_code=422)

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    # Save the uploaded audio file.
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    if file.filename.endswith((".mp3", ".wav", ".ogg")):
        # Define the RNNModel class
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(RNNModel, self).__init__()
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                out, _ = self.rnn(x)
                out = self.fc(out[:, -1, :])
                return out
        
        # Define the path to your trained PyTorch model
        model_path = "app/utils/semangka_model.pth"
        
        with open("app/utils/labels.txt", "r") as file:
            class_labels = [line.strip() for line in file]
        # Load the trained model
        input_size = 153
        hidden_size = 512
        num_classes = len(class_labels)

        model = RNNModel(input_size, hidden_size, num_classes)
       
        model.load_state_dict(torch.load(model_path))
        model.eval()
            # Load the scaler
        scaler = joblib.load(scaler_path)
        audio_features = preprocess_audio(file_path)
        audio_features = scaler.transform(audio_features)
        print(audio_features)
        # Convert the features to a PyTorch tensor
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32)

        # Reshape the tensor to match the model's input size
        audio_tensor = audio_tensor.view(1, 1, audio_tensor.size(1))
        # print(audio_tensor)
        import torch.nn.functional as F
        with torch.no_grad():
            output = model(audio_tensor)
            probabilities = F.softmax(output, dim=1)

        predicted_class_index = output.argmax().item()
        predicted_class = class_labels[predicted_class_index]

    
    return JSONResponse(content={"predicted_label": predicted_class})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)