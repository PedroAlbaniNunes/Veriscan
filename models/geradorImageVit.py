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
CLASS_NAMES = ['Real', 'IA Gerada']  # Ordem das classes no treino
MODEL_PATH = 'vit_ai_detector.pth'

# Variável global para capturar os pesos de atenção
attention_weights = None

def patch_vit_attention(model):
    """
    Modifica temporariamente o método forward da camada de auto-atenção 
    do último bloco do ViT para forçar 'need_weights=True'.
    """
    target_layer = model.encoder.layers[-1].self_attention
    
    # Salvamos o forward original caso precise
    old_forward = target_layer.forward
    
    def new_forward(self, input, key, value, need_weights=True, attn_mask=None, average_attn_weights=True):
        global attention_weights
        # Forçamos need_weights=True para capturar a matriz
        x, weights = self._-orig_forward(input, key, value, need_weights=True, attn_mask=attn_mask, average_attn_weights=average_attn_weights) if hasattr(self, '_-orig_forward') else super(type(self), self).forward(input, key, value, need_weights=True, attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        attention_weights = weights.detach()
        return x, weights

    # Forma limpa de sobrescrever o forward da instância específica no PyTorch
    def custom_forward(query, key, value, need_weights=False, **kwargs):
        global attention_weights
        # Chamamos a MultiheadAttention nativa passando need_weights=True obrigatoriamente
        out, weights = target_layer._old_forward(query, key, value, need_weights=True, **kwargs)
        attention_weights = weights.detach()
        return out, weights

    target_layer._old_forward = target_layer.forward
    target_layer.forward = custom_forward

def get_vit_model(num_classes):
    """Carrega o modelo, aplica o patch de atenção e retorna."""
    model = models.vit_b_16(weights=None)
    
    # Ajustar o classificador final igual ao treino
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Carregar pesos treinados
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Modelo carregado com sucesso de '{MODEL_PATH}'.")
    else:
        print(f"Aviso: Modelo '{MODEL_PATH}' não encontrado. Usando pesos aleatórios.")
        
    model.to(DEVICE)
    model.eval()
    
    # Aplicar o patch para capturar a atenção da última camada
    patch_vit_attention(model)
    
    return model

def generate_attention_heatmap(image_path, model):
    """Processa a imagem e gera o mapa de calor baseado na atenção do ViT de forma segura."""
    global attention_weights
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[preds.item()]
    conf_percentage = confidence.item() * 100

    print(f"Predição: {predicted_class} ({conf_percentage:.2f}% de confiança)")

    if attention_weights is not None:
        attn = attention_weights.detach().cpu()
        
        # Tratamento dinâmico de dimensões para evitar erros de IndexError
        if attn.ndim == 4:
            # Formato: (batch, num_heads, q_len, k_len) -> tira a média das cabeças e remove o batch
            attn = attn.mean(dim=1).squeeze(0)
        elif attn.ndim == 3:
            # Formato: (batch, q_len, k_len) -> remove o batch
            attn = attn.squeeze(0)
            
        # Garante que temos uma matriz 2D (q_len, k_len)
        if attn.ndim == 2:
            # Atenção do token [CLS] (índice 0) para os patches de imagem (índices 1 em diante)
            cls_attention = attn[0, 1:]
        else:
            cls_attention = attn.flatten()

        grid_size = int(np.sqrt(cls_attention.shape[0]))
        mask = cls_attention.reshape(grid_size, grid_size).numpy()
        
        # Normalização segura da máscara entre 0 e 1
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    else:
        mask = np.ones((14, 14), dtype=np.float32)

    return image_pil, mask, predicted_class, conf_percentage

def plot_analysis_panel(image_path, save_path='heatmap_resultado.png'):
    """Cria e salva o painel visual comparativo."""
    model = get_vit_model(len(CLASS_NAMES))
    image_pil, mask, pred_class, conf = generate_attention_heatmap(image_path, model)
    
    img_resized = image_pil.resize((224, 224))
    img_np = np.array(img_resized)

    # Em vez de um redimensionamento simples, usamos interpolação cúbica para suavizar
    mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Aplicar um leve desfoque gaussiano para eliminar o aspecto pixelado/quadriculado
    mask_resized = cv2.GaussianBlur(mask_resized, (15, 15), 0)

    # Normalizar novamente após o filtro
    mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)  

    # Criar o mapa de calor colorido (colormap JET)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Sobrepor o heatmap na imagem original (transparência de 40%)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Plotar os três painéis lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_resized)
    axes[0].set_title("1. Imagem Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title("2. Mapa de Atenção (Heatmap)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title(f"3. Detecção\n[{pred_class} - {conf:.1f}%]", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Painel com mapa de calor salvo com sucesso em '{save_path}'!")
    plt.show()

# ==========================================
# EXECUÇÃO
# ==========================================
if __name__ == '__main__':
    caminho_da_imagem = 'test_image.jpg'  # Altere para o caminho da imagem de teste
    plot_analysis_panel(caminho_da_imagem)