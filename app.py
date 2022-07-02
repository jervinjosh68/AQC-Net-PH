from model import AQC_NET
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr
import requests
import os

def get_file(url,path,filename, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as downloaded:
        for chunk in r.iter_content(chunk_size=chunk_size):
            downloaded.write(chunk)


def predict(image_name):
    model.eval()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = preprocess(image_name)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs.unsqueeze(0))
        values, indices = torch.topk(outputs, k=3) 
        print(values,indices)
    return {i.item(): v.item() for i, v in zip(indices[0], values.detach()[0])}
def preprocess(image_name):
    transforms = T.Compose([
        T.Resize((256,256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transforms(image_name)
    return image

def run_gradio():
    
    title = "AQC_NET PH"
    description = "trial AQC_NET"
    examples = ["test_image.jpg","test_img.jpg"]
    inputs = [
        gr.inputs.Image(type="pil", label="Input Image")
    ]


    gr.Interface(
        predict,
        inputs,
        outputs = 'label',
        title=title,
        description=description,
        examples=examples,
        theme="huggingface",
    ).launch(debug=True, enable_queue=True)

model = AQC_NET(pretrain=True, num_label=2)
if not os.path.exists('weight.pth'):
    print("weight.pth does not exist. Downloading...")
    get_file("https://github.com/Kaldr4/EEE-199/releases/download/v1/weight.pth", 'weight.pth',"weight.pth")
    print("weight.pth downloaded")
else:
    print('Specified file (weight.pth) already downloaded. Skipping this step.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('weight.pth', map_location=torch.device(device))
model.load_state_dict(state_dict)

run_gradio()