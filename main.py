import torchxrayvision as xrv
import os
import requests
from requests.auth import HTTPBasicAuth
import yaml
import json, torch, torchvision, skimage
import uuid

with open('config.yaml') as cfg:
    config = yaml.safe_load(cfg)

ORTHAN_URL = os.getenv('ORTHANC_URL')
DICOM_DIR = config['Dicom_Files_Directory']
RESULTS_DIR = config['Results_Dir']

print("Iniciando processo")
print(f"OrthanC url: {ORTHAN_URL}")
print(f"Dicom files: {DICOM_DIR}")
print(f"Results dir: {RESULTS_DIR}")

# Inicializando modelo de classificacao
model = xrv.models.DenseNet(weights="densenet121-res224-all")

for file in os.listdir(DICOM_DIR):
    # Enviando arquivos DICOM para o ORTHANC
    file_path = os.path.join(DICOM_DIR, file)
    print(f"Arquivo: {file_path}")
    with open(file_path, 'rb') as f:
        requestUrl = ORTHAN_URL + "/instances"
        response = requests.post(requestUrl, files={'file':f}, auth=HTTPBasicAuth('admin','admin'))
        print(f"Enviado arquivo: {file} para url: {ORTHAN_URL}: status: {response.status_code}")

        # Preparando a imagem
        img = skimage.io.imread(file_path)
 
        if img.ndim == 2:
            # Caso a imagem ja esteja na escala de cinza vamos adicionar um eixo a matriz
            img = img[None,...]
        elif img.ndim == 3:
            # Caso esteja em RGB vamos converter para uma escala monocromatica
            img = img.mean(2,keepdims=True)
            
        # Normalizando e transformando a imagem
        img = xrv.datasets.normalize(img, img.max())
        
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img = torch.from_numpy(img)
        
        # Processando imagem
        outputs = model(img[None,...])
        output_list = outputs.tolist()
         
        sr_json = {
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.88.22",
            "SOPInstanceUID": str(uuid.uuid4()),
            "StudyInstanceUID": str(uuid.uuid4()),
            "SeriesInstanceUID": str(uuid.uuid4()),
            "ContentSequence": [
                {
                    "ValueType": "TEXT",
                    "TextValue": "Model output scores for the DICOM image."
                },
                {
                    "ValueType": "NUMERIC",
                    "NumericValues": output_list
                }
            ],
            "Conclusion": "Model processed the DICOM image and generated prediction scores."
        }
        
        results_filename = f"{os.path.splitext(file)[0]}_results.json"
        results_path = os.path.join(RESULTS_DIR,results_filename)
         
        print(f"Salvando resultados no arquivo: {results_filename}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(results_path, 'w') as r:
            json.dump(sr_json, r, indent=4)
    
print("Processo finalizado")