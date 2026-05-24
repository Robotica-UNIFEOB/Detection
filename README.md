# Visão Computacional — Detecção de Formas Geométricas

Sistema de detecção de formas geométricas em tempo real usando câmera, desenvolvido para o projeto do robô com garra.

## Como funciona

A câmera do PC captura o vídeo em tempo real e identifica formas geométricas (Círculo, Quadrado e Triângulo) usando um modelo treinado no Roboflow com 98.7% de precisão. O sistema destaca a forma mais próxima do centro da câmera como a próxima a ser pega pelo robô.

## Arquivos

- `detectar.py` — script principal de detecção em tempo real
- `PI2.ipynb` — notebook do Google Colab usado para treinar o modelo de classificação

## Requisitos

- Python 3.11 — download em https://python.org/downloads/release/python-3119/
- Durante a instalação marcar a opção **Add Python to PATH**

## Instalação

Abra o terminal na pasta do projeto e rode:

```
py -3.11 -m pip install inference-sdk opencv-python
```

## Como rodar

```
py -3.11 detectar.py
```

A janela da câmera abrirá automaticamente. Pressione **Q** para fechar.

## Como usar na apresentação

1. Liga o robô e conecta no WiFi
2. Abre o browser no IP do ESP32 para controle manual
3. Roda o `detectar.py` no PC
4. Coloca as formas na frente da câmera
5. O sistema mostra na tela qual forma está mais próxima
6. Operador controla o robô manualmente para pegar a forma indicada

## Cores na tela

| Forma | Cor da caixa |
|---|---|
| Círculo | Laranja |
| Quadrado | Verde |
| Triângulo | Azul |

## Treinamento

O modelo foi treinado com 104 imagens das 3 formas geométricas usando YOLOv11 Nano via Roboflow, atingindo 98.7% de mAP@50.
