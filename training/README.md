# Training & Fine-tuning

Training and fine-tuning AI models is where the magic happens - transforming raw data and algorithms into intelligent systems that can solve real-world problems. This section covers everything from training your first model to advanced techniques for optimizing large-scale AI systems.

## 🎯 What is Model Training?

Model training is the process of teaching AI algorithms to make predictions or decisions by learning patterns from data. It involves:

- **Learning from Data**: Algorithms discover patterns in training datasets
- **Parameter Optimization**: Adjusting model weights to minimize errors
- **Validation**: Testing model performance on unseen data
- **Iteration**: Continuously improving through multiple training cycles
- **Generalization**: Ensuring models work well on new, unseen data

## 🧠 Types of Training

### From Scratch Training
- **Complete Model Development**: Building models from ground zero
- **Full Control**: Complete customization of architecture and training process
- **Resource Intensive**: Requires large datasets and computational resources
- **Use Cases**: Unique problems, research applications, proprietary data

### Transfer Learning
- **Pre-trained Foundation**: Start with models trained on large datasets
- **Adaptation**: Modify existing models for specific tasks
- **Efficiency**: Faster training with smaller datasets
- **Use Cases**: Common applications like image classification, NLP tasks

### Fine-tuning
- **Specialized Adaptation**: Refine pre-trained models for specific domains
- **Parameter Adjustment**: Modify a subset of model parameters
- **Domain Adaptation**: Adapt models to new domains or tasks
- **Use Cases**: Specialized applications, domain-specific improvements

## 📚 Learning Resources

### Practical Training Guides

* [Train a Reasoning-Capable LLM in One Weekend](https://www.youtube.com/watch?v=hMGikmMFLAU)
  - Practical guide to training language models
  - Weekend project approach for hands-on learning
  - Reasoning capabilities development
  - Cost-effective training strategies

## 🔧 Training Fundamentals

### Data Preparation
**Dataset Quality**
- **Clean Data**: Remove noise, outliers, and inconsistencies
- **Balanced Classes**: Ensure representative distribution
- **Sufficient Volume**: Adequate data for pattern learning
- **Diverse Examples**: Cover various scenarios and edge cases

**Data Splitting**
```python
from sklearn.model_selection import train_test_split

# Standard 70-20-10 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.222, random_state=42, stratify=y_temp
)  # 0.222 * 0.9 = 0.2 (20% of total)
```

### Model Architecture Design
**Neural Network Design Principles**
- **Layer Selection**: Choose appropriate layer types for your data
- **Depth vs. Width**: Balance network depth and width
- **Activation Functions**: Select suitable non-linearities
- **Regularization**: Prevent overfitting with dropout, batch norm

**Architecture Examples**
```python
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## ⚙️ Training Process

### Loss Functions
**Common Loss Functions**
- **Classification**: CrossEntropyLoss, BCELoss
- **Regression**: MSELoss, MAELoss, HuberLoss
- **Custom Losses**: Domain-specific objectives

```python
import torch.nn as nn

# Multi-class classification
criterion = nn.CrossEntropyLoss()

# Binary classification
criterion = nn.BCEWithLogitsLoss()

# Regression
criterion = nn.MSELoss()

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)
```

### Optimizers
**Popular Optimization Algorithms**
- **SGD**: Simple, reliable, requires tuning
- **Adam**: Adaptive learning rates, good default choice
- **AdamW**: Adam with weight decay correction
- **RMSprop**: Good for RNNs and unstable gradients

```python
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### Training Loop
```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step()
    
    return train_losses, val_losses
```

## 🔧 Advanced Training Techniques

### Regularization Methods
**Preventing Overfitting**
- **Dropout**: Randomly deactivate neurons during training
- **Batch Normalization**: Normalize layer inputs
- **Weight Decay**: L2 regularization on parameters
- **Early Stopping**: Stop training when validation performance plateaus

```python
# Dropout implementation
self.dropout = nn.Dropout(p=0.5)

# Batch normalization
self.batch_norm = nn.BatchNorm1d(hidden_size)

# Weight decay in optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
```

### Data Augmentation
**Increasing Dataset Diversity**
```python
from torchvision import transforms

# Image augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Text augmentation (example)
def augment_text(text):
    # Synonym replacement, random insertion, etc.
    return augmented_text
```

### Mixed Precision Training
**Faster Training with Reduced Memory**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 🚀 Distributed Training

### Multi-GPU Training
**Data Parallel Training**
```python
import torch.nn as nn

# Simple data parallelism
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)
```

**Distributed Data Parallel (DDP)**
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    model = YourModel()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Training code here
    
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

### Cloud Training Platforms
**Google Cloud Platform**
- Vertex AI Training: Managed training jobs
- TPU support for large models
- Automatic scaling and resource management

**AWS SageMaker**
- Distributed training jobs
- Spot instance support for cost reduction
- Automatic model tuning (hyperparameter optimization)

**Azure Machine Learning**
- Compute clusters for distributed training
- MLflow integration
- Automated ML capabilities

## 🎛️ Hyperparameter Optimization

### Manual Tuning
**Key Hyperparameters**
- **Learning Rate**: Most critical parameter to tune
- **Batch Size**: Affects training stability and speed
- **Architecture**: Number of layers, neurons, etc.
- **Regularization**: Dropout rates, weight decay

### Automated Methods
**Grid Search**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'hidden_size': [64, 128, 256]
}

# Custom implementation for deep learning
def grid_search_train():
    best_score = 0
    best_params = {}
    
    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [32, 64, 128]:
            # Train model with these parameters
            score = train_and_evaluate(lr, batch_size)
            if score > best_score:
                best_score = score
                best_params = {'lr': lr, 'batch_size': batch_size}
    
    return best_params, best_score
```

**Bayesian Optimization**
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    
    # Train model and return validation score
    model = create_model(hidden_size)
    score = train_model(model, lr, batch_size)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
```

## 🎯 Fine-tuning Pre-trained Models

### Transfer Learning Strategies
**Feature Extraction**
```python
import torchvision.models as models

# Load pre-trained model
resnet = models.resnet50(pretrained=True)

# Freeze all parameters
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# Only train the final layer
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
```

**Fine-tuning All Layers**
```python
# Load pre-trained model
model = models.resnet50(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Use different learning rates for different parts
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.features.parameters(), 'lr': 0.0001}
])
```

### Language Model Fine-tuning
**Hugging Face Transformers**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_classes
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()
```

## 📊 Training Monitoring and Visualization

### Real-time Monitoring
**TensorBoard Integration**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# Log training metrics
writer.add_scalar('Loss/Train', train_loss, epoch)
writer.add_scalar('Loss/Validation', val_loss, epoch)
writer.add_scalar('Accuracy/Train', train_acc, epoch)
writer.add_scalar('Accuracy/Validation', val_acc, epoch)

# Log learning rate
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

# Log model graph
writer.add_graph(model, sample_input)

# Log histograms of model parameters
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)

writer.close()
```

**Weights & Biases Integration**
```python
import wandb

# Initialize wandb
wandb.init(project="model-training", config={
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
})

# Log metrics during training
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "train_acc": train_acc,
    "val_acc": val_acc
})

# Log model
wandb.watch(model, log="all")
```

### Performance Visualization
```python
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## 🔧 Training Infrastructure

### Hardware Considerations
**GPU Requirements**
- **Memory**: Sufficient VRAM for model and batch size
- **Compute Capability**: Modern GPUs for tensor operations
- **Multi-GPU**: Data parallel training for large models
- **TPUs**: Google's tensor processing units for specific workloads

**Memory Optimization**
```python
# Gradient accumulation for large effective batch sizes
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Cloud Training Services
**Benefits of Cloud Training**
- **Scalability**: Scale resources up or down as needed
- **Cost Efficiency**: Pay only for compute time used
- **Specialized Hardware**: Access to GPUs, TPUs, and high-memory instances
- **Managed Services**: Reduced infrastructure management overhead

**Platform Comparison**
- **Google Cloud**: Strong TPU support, Vertex AI platform
- **AWS**: Comprehensive ML services, SageMaker ecosystem
- **Azure**: Enterprise integration, Azure ML studio
- **Paperspace**: Developer-friendly, gradient platform

## 🔗 Integration with Other Topics

### Cross-References
- **[Deep Learning](../deep-learning/README.md)**: Advanced architectures and techniques
- **[Python](../python/README.md)**: Implementation skills and libraries
- **[Mathematics](../math/README.md)**: Understanding optimization and gradients
- **[Tools](../tools/README.md)**: Development and monitoring tools

## 💡 Best Practices

### Training Workflow
1. **Start Simple**: Begin with basic models and gradually increase complexity
2. **Baseline First**: Establish a simple baseline before optimization
3. **Iterative Improvement**: Make incremental changes and measure impact
4. **Reproducibility**: Set random seeds and document configurations
5. **Regular Checkpointing**: Save model states frequently

### Common Pitfalls
- **Overfitting**: Monitor validation metrics carefully
- **Underfitting**: Ensure model has sufficient capacity
- **Data Leakage**: Maintain strict train/validation/test separation
- **Inappropriate Metrics**: Choose metrics aligned with business objectives
- **Hyperparameter Tunnel Vision**: Don't over-optimize on validation set

### Performance Optimization
- **Profiling**: Identify bottlenecks in training pipeline
- **Data Loading**: Optimize data preprocessing and loading
- **Mixed Precision**: Use half-precision for faster training
- **Gradient Clipping**: Prevent exploding gradients
- **Learning Rate Scheduling**: Adapt learning rate during training

Training effective AI models is both an art and a science. Focus on understanding the fundamentals while staying current with the latest techniques and best practices. Remember that good data and careful validation are often more important than complex algorithms!
