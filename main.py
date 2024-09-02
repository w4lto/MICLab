import torchxrayvision as xrv
import os
import requests
import yaml
import json, torch, torchvision

with open('config.yaml') as cfg:
    config = yaml.safe_load(cfg)

ORTHAN_URL = config['OrthanC_Url']
DICOM_DIR = config['Dicom_Files_Directory']
RESULTS_JSON = config['Results_File']

print("Iniciando processo")
print(f"OrthanC url: {ORTHAN_URL}")
print(f"Dicom files: {DICOM_DIR}")
print(f"Results file: {RESULTS_JSON}")

# Inicializando modelo de classificacao
model = xrv.models.DenseNet(weights="densenet121-res224-all")

for file in os.listdir(DICOM_DIR):
    # Enviando arquivos DICOM para o ORTHANC
    with open(os.path.join(DICOM_DIR, file), 'rb') as f:
        response = requests.post(ORTHAN_URL, files={'file':f})
        print(f"Enviado arquivo: {file} para url: {ORTHAN_URL}: status: {response.status_code}")

    # Preparando a imagem
    img = xrv.utils.read_xray_dcm(os.path.join(DICOM_DIR, file))
    img = xrv.datasets.normalize(img, 255)
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img = torch.from_numpy(img)
    
    # Processando imagem
    outputs = model(img[None,...])
    print(f"Predictions for file: {file}: {outputs}")
    
    print(f"Salvando resultados no arquivo: {RESULTS_JSON}")
    with open(RESULTS_JSON, 'w') as f:
        json.dump(outputs, f)
    
print("Processo finalizado")