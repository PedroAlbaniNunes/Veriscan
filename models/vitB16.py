import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import os

# ==========================================
# 0. AJEITANDO O PATH DO DATASET
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '../dataset')  # Pasta com subpastas por classe
# Caminhos para as pastas de treino e teste (ajuste conforme seu diretório)
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
TEST_DIR = os.path.join(DATASET_PATH, 'test')

# ==========================================
# 1. CONFIGURAÇÕES E HIPERPARÂMETROS
# ==========================================

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 2 # Ex: [0] Real, [1] IA Gerada
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {DEVICE}")

# ==========================================
# 2. PREPARAÇÃO DOS DADOS (DATA AUGMENTATION)
# ==========================================
# Aumento de dados agressivo para evitar que o modelo memorize a base de treino
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # Adições para evitar overfitting:
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Carregando os datasets
image_datasets = {
    'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
    'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(f"Classes identificadas: {class_names}")

# ==========================================
# 3. DEFINIÇÃO DE VISION TRANSFORMER (ViT)
# ==========================================
def build_model(num_classes):
    # Inicializando ViT com patches 16x16
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    
    # Congelar as camadas de extração de características (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False
        
    # Substituir o classificador final
    # O Dropout (0.5) é crucial aqui para evitar que a rede dependa de caminhos específicos
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model.to(DEVICE)

model = build_model(NUM_CLASSES)

# ==========================================
# 4. FUNÇÃO DE PERDA E OTIMIZADOR
# ==========================================
criterion = nn.CrossEntropyLoss()

# Apenas os parâmetros do "head" (classificador final) serão atualizados
optimizer = optim.Adam(model.heads.head.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. LOOP DE TREINAMENTO (COM MÉTRICAS)
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Criando o dicionário history que a sua função de gráficos exige
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'train_f1': [], 'test_f1': [],
        'train_prec': [], 'test_prec': [],
        'train_rec': [], 'test_rec': []
    }

    for epoch in range(num_epochs):
        print(f'Época {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Listas para guardar todas as previsões e rótulos da época
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Salvar previsões para calcular as métricas do Sklearn depois
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().data.numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]
            
            # Calculando F1, Precision e Recall (usando macro average)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            # Alimentando o dicionário history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_f1'].append(epoch_f1)
            history[f'{phase}_prec'].append(epoch_prec)
            history[f'{phase}_rec'].append(epoch_rec)

            print(f'{phase.capitalize()} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Treinamento concluído em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Melhor Acurácia de Teste: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# 6. PAINEL DE GRÁFICOS COMPLETO
# ==========================================
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(16, 10))
    
    # Gráfico 1: Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Treino', marker='o', color='blue')
    plt.plot(epochs, history['test_loss'], label='Teste', marker='o', color='red')
    plt.title('Evolução da Perda (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 2: Acurácia
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Treino', marker='o', color='blue')
    plt.plot(epochs, history['test_acc'], label='Teste', marker='o', color='red')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 3: F1-Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_f1'], label='Treino', marker='o', color='blue')
    plt.plot(epochs, history['test_f1'], label='Teste', marker='o', color='red')
    plt.title('Evolução do F1-Score (Macro)')
    plt.xlabel('Épocas')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 4: Precisão e Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['test_prec'], label='Precisão (Teste)', marker='s', color='purple')
    plt.plot(epochs, history['test_rec'], label='Recall (Teste)', marker='^', color='orange')
    plt.title('Precisão vs Recall (Validação)')
    plt.xlabel('Épocas')
    plt.ylabel('Métrica')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('grafico_treinamento_vit16.png', dpi=300)
    print("\nGráfico detalhado salvo como 'grafico_treinamento_vit16.png'!")
    plt.show()

# ==========================================
# 7. EXECUÇÃO
# ==========================================
if __name__ == '__main__':
    # Agora a função retorna o modelo E o histórico
    model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
    
    # Salvar o modelo treinado
    torch.save(model.state_dict(), 'vit_ai_detector.pth')
    print("Modelo salvo como 'vit_ai_detector.pth'")
    
    # Chamar a sua função para gerar os gráficos
    plot_history(history)