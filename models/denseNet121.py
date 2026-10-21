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
DATASET_PATH = os.path.join(BASE_DIR, '../dataset')

# ==========================================
# 1. CONFIGURAÇÕES E HIPERPARAMETROS
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30  # Com transfer learning, geralmente convergem mais rápido
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Executando em: {DEVICE}")

# ==========================================
# 2. PREPARAÇÃO DOS DADOS (DATA AUGMENTATION)
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x), data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# ==========================================
# 3. DEFINIÇÃO DA DENSENET-121 COM TRANSFER LEARNING
# ==========================================
def build_model(num_classes):
    # Carregando pesos oficiais pré-treinados do ImageNet
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # CONGELAR AS CAMADAS DE EXTRAÇÃO DE CARACTERÍSTICAS
    # Isso impede que os pesos pré-treinados sejam destruídos no início com poucos dados
    for param in model.parameters():
        param.requires_grad = False
        
    # Ajustando a última camada (classifier) permitindo gradiente apenas nela
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5), # Dropout adicionado para evitar overfitting
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model.to(DEVICE)

model = build_model(NUM_CLASSES)

# ==========================================
# 4. FUNÇÃO DE PERDA E OTIMIZADOR
# ==========================================
criterion = nn.CrossEntropyLoss()

# Otimizamos apenas os parâmetros da camada classificadora final
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. LOOP DE TREINAMENTO
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=25):
    history = {
        'train_loss': [], 'test_loss': [], 
        'train_acc': [], 'test_acc': [],
        'train_f1': [], 'test_f1': [],
        'train_prec': [], 'test_prec': [],
        'train_rec': [], 'test_rec': []
    }

    best_acc = 0.0

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
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f} | Prec: {epoch_prec:.4f} | Rec: {epoch_rec:.4f}')

            prefix = 'train_' if phase == 'train' else 'test_'
            history[f'{prefix}loss'].append(epoch_loss)
            history[f'{prefix}acc'].append(epoch_acc)
            history[f'{prefix}f1'].append(epoch_f1)
            history[f'{prefix}prec'].append(epoch_prec)
            history[f'{prefix}rec'].append(epoch_rec)

    return model, history

# ==========================================
# 6. EXECUÇÃO
# ==========================================
if __name__ == '__main__':
    model_trained, training_history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
    
    # Salvar o novo modelo treinado com transfer learning
    torch.save(model_trained.state_dict(), 'densenet121_custom.pth')
    print("Novo modelo com Transfer Learning salvo com sucesso como 'densenet121_custom.pth'!")