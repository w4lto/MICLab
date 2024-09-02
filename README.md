# MICLab

Projeto para processo seletivo do projecto de IC do MICLab UNICAMP
- Desenvolvido um programa em python que realiza as seguintes operacoes:
  - Envio de arquivos DCM para o OrthanC via requisicao http
  - Utilizacao do modelo pre-treinado para analise e classificacao dos arquivos DCM utilizando o TorchXRay
  - Criando arquivos DICOM SR (Structured Report) para cada arquivo DICOM analisado
Foram encontradas algumas dificuldades durante o processo de desenvolvimento, dentre elas:
  - Normalizacao diferenciada para imagens monocromaticas e em formato RGB
