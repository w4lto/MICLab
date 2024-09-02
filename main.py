import torchxrayvision as xrv
import os
import requests
import yaml
import skimage, torch, torchvision

with open('config.yaml') as cfg:
    config = yaml.safe_load(cfg)

ORTHAN_URL = config['OrthanC_Url']
DICOM_DIR = config['Dicom_Files_Directory']
RESULTS = config['Results_File']

print("Starting predicting process")
print(f"OrthanC url: {ORTHAN_URL}")
print(f"OrthanC url: {ORTHAN_URL}")
print(f"OrthanC url: {ORTHAN_URL}")

# Inicializando modelo de classificacao
model = xrv.models.DenseNet(weights="densenet121-res224-all")

for file in os.listdir(DICOM_DIR):
    with open(os.path.join(DICOM_DIR, file), 'rb') as f:
        response = requests.post(ORTHAN_URL, files={'file':f})
        print(f"Uploaded file: {file}: status: {response.status_code}")

    img = xrv.utils.read_xray_dcm(os.path.join(DICOM_DIR, file))
    img = xrv.datasets.normalize(img, 255)
    img = img.mean(2)[None,:,:]
    preds = model(img)
    print(f"Predictions for file: {file}: {preds}")
    
print(f"Process finished")