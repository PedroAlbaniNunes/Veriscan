import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ==========================================
# CONFIGURAÇÕES BÁSICAS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Real', 'IA Gerada']  # Ordem exata das classes no seu DatasetFolder
MODEL_PATH = 'densenet121_custom.pth'
NUM_CLASSES = 2

# Variáveis globais para capturar ativações e gradientes do Grad-CAM
activated_features = None
gradients = None

def save_activation(module, input, output):
    """Hook para capturar a saída da última camada convolucional interna."""
    global activated_features
    activated_features = output.detach()

def save_gradient(module, grad_input, grad_output):
    """Hook para capturar os gradientes da última camada convolucional interna."""
    global gradients
    gradients = grad_output[0].detach()

def get_densenet_model_with_hooks(num_classes):
    """Carrega a DenseNet-121 com a estrutura de Transfer Learning e Dropout."""
    model = models.densenet121(weights=None)
    
    
    # CORREÇÃO: Usar a mesma estrutura com Dropout usada no novo treinamento
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Modelo carregado com sucesso de '{MODEL_PATH}'.")
    else:
        print(f"Aviso: Arquivo '{MODEL_PATH}' não encontrado.")
        
    model.to(DEVICE)
    model.eval()
    
    # Ponto de Hook para o Grad-CAM
    target_layer = model.features.denseblock4.denselayer16
    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)
    
    return model

def generate_gradcam(image_path, model):
    """Executa a inferência e calcula o mapa de calor Grad-CAM sem avisos de backward."""
    global activated_features, gradients
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    
    # CORREÇÃO DO WARNING: Forçar o tensor de entrada a exigir gradientes
    input_tensor.requires_grad_(True)

    # 1. Forward Pass
    model.zero_grad()
    output = model(input_tensor)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, preds = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[preds.item()]
    conf_percentage = confidence.item() * 100

    print(f"Predição: {predicted_class} ({conf_percentage:.2f}% de confiança)")

    # 2. Backward Pass para a classe predita
    class_idx = preds.item()
    score = output[0, class_idx]
    score.backward()

    # 3. Processamento espacial do Grad-CAM
    if gradients is not None and activated_features is not None:
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = activated_features.squeeze(0)
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
    else:
        heatmap = np.ones((7, 7), dtype=np.float32)

    return image_pil, heatmap, predicted_class, conf_percentage

def plot_analysis_panel(image_path, save_path='densenet_heatmap_resultado.png'):
    """Cria e salva o painel visual comparativo."""
    model = get_densenet_model_with_hooks(NUM_CLASSES)
    image_pil, heatmap, pred_class, conf = generate_gradcam(image_path, model)
    
    img_resized = image_pil.resize((224, 224))
    img_np = np.array(img_resized)

    # Redimensionar o heatmap para 224x224 com interpolação suave
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Gerar o mapa colorido e sobrepor na imagem
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    # Plotar o painel analítico (3 colunas)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_resized)
    axes[0].set_title("1. Imagem Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap_colored)
    axes[1].set_title("2. Mapa Grad-CAM (Refinado)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title(f"3. Detecção\n[{pred_class} - {conf:.1f}%]", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Painel Grad-CAM salvo com sucesso em '{save_path}'!")
    plt.show()

# ==========================================
# EXECUÇÃO
# ==========================================
if __name__ == '__main__':
    caminho_da_imagem = 'test_image.jpg'  # Altere para o caminho da sua imagem de teste
    plot_analysis_panel(caminho_da_imagem)