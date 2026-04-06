import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt 
import os


# ==========================================
# 0. AJEITANDO O PATH DO DATASET
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '../dataset')  # Pasta com subpastas por classe

# ==========================================
# 1. CONFIGURAÇÕES E HIPERPARAMETROS
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 60
NUM_CLASSES = 2  # Altere para o seu número de categorias
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Executando em: {DEVICE}")

# ==========================================
# 2. PREPARAÇÃO DOS DADOS (DATA AUGMENTATION)
# ==========================================
# Como você vai treinar do zero, o Augmentation é vital para evitar overfitting
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
# Espera-se que dentro de DATASET_PATH existam as pastas 'train' e 'test'
image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x), data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# ==========================================
# 3. DEFINIÇÃO DA DENSENET-121 DO ZERO
# ==========================================
def build_model(num_classes):
    # weights=None garante que não usaremos o modelo pré-treinado
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Ajustando a última camada (classifier)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model.to(DEVICE)

model = build_model(NUM_CLASSES)

# ==========================================
# 4. FUNÇÃO DE PERDA E OTIMIZADOR
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. LOOP DE TREINAMENTO (COM F1, PRECISION E RECALL)
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=25):
    # Dicionário expandido para guardar todas as métricas
    history = {
        'train_loss': [], 'test_loss': [], 
        'train_acc': [], 'test_acc': [],
        'train_f1': [], 'test_f1': [],
        'train_prec': [], 'test_prec': [],
        'train_rec': [], 'test_rec': []
    }

    for epoch in range(num_epochs):
        print(f'\nÉpoca {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Listas para guardar as previsões e gabaritos da época inteira
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
                
                # Salvando os resultados para o scikit-learn (movendo para CPU)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Cálculos das métricas
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            
            # Calculando F1, Precision e Recall (usando macro para balancear as classes)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f} | Prec: {epoch_prec:.4f} | Rec: {epoch_rec:.4f}')

            # Alimentando o histórico
            prefix = 'train_' if phase == 'train' else 'test_'
            history[f'{prefix}loss'].append(epoch_loss)
            history[f'{prefix}acc'].append(epoch_acc)
            history[f'{prefix}f1'].append(epoch_f1)
            history[f'{prefix}prec'].append(epoch_prec)
            history[f'{prefix}rec'].append(epoch_rec)

    return model, history

# ==========================================
# 6. PAINEL DE GRÁFICOS COMPLETO
# ==========================================
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Criando um painel 2x2 para acomodar 4 gráficos
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
    
    # Gráfico 4: Precisão e Recall (Apenas de Teste para não poluir)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['test_prec'], label='Precisão (Teste)', marker='s', color='purple')
    plt.plot(epochs, history['test_rec'], label='Recall (Teste)', marker='^', color='orange')
    plt.title('Precisão vs Recall (Validação)')
    plt.xlabel('Épocas')
    plt.ylabel('Métrica')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('grafico_treinamento_completo2.png', dpi=300)
    print("\nGráfico detalhado salvo como 'grafico_treinamento_completo2.png'!")
    plt.show()

# Iniciar o treino
if __name__ == '__main__':
    # Agora a função retorna o modelo e o histórico
    model_trained, training_history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
    
    # Salvar o modelo final
    torch.save(model_trained.state_dict(), 'densenet121_custom.pth')
    print("Modelo salvo com sucesso!")
    
    # Gerar e exibir os gráficos
    plot_history(training_history)