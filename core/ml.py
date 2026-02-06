import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Tuple, Dict
import os

# 设置 MLflow 追踪地址
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# --- 1. 定义 Transformer 模型 ---
class TabularTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64, nhead=4, num_layers=2, task_type="classification"):
        super(TabularTransformer, self).__init__()
        self.task_type = task_type
        
        # 特征投影层：把输入特征映射到高维空间
        self.embedding = nn.Linear(num_features, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头
        output_dim = num_classes if task_type == "classification" else 1
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, num_features]
        x = self.embedding(x)   # [batch, d_model]
        x = x.unsqueeze(1)      # [batch, 1, d_model] -> 伪造序列长度为1
        
        x = self.transformer_encoder(x)
        x = x.squeeze(1)        # [batch, d_model]
        x = self.fc_out(x)      # [batch, output_dim]
        return x

# --- 2. 核心训练函数 ---
def train_transformer_model(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 15,    # 训练轮数
    batch_size: int = 32 # 批次大小
) -> Tuple[str, Dict[str, float]]:
    
    # === A. 数据预处理 ===
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. 填充缺失值 (Transformer 不能吃 NaN)
    # 简单策略：数值填 0，字符串填 "Unknown"
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna("Unknown")
        else:
            X[col] = X[col].fillna(0)
            
    # 2. 编码类别特征 (Label Encoding)
    for col in X.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # 3. 处理目标列
    num_classes = 1
    if task_type == "classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        num_classes = len(np.unique(y))
        print(f"📊 Detected {num_classes} classes.")
    
    # 4. 数据切分
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 5. 标准化 (StandardScaling) - 对深度学习至关重要！
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 6. 转为 PyTorch Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    if task_type == "classification":
        y_train_t = torch.tensor(y_train, dtype=torch.long) # 分类用 Long (整数索引)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # 回归要保持维度一致
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # 7. 创建 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # === B. 模型初始化 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Training on: {device}")
    
    model = TabularTransformer(
        num_features=X_train.shape[1], 
        num_classes=num_classes, 
        task_type=task_type
    ).to(device)
    
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === C. MLflow 实验记录 ===
    mlflow.set_experiment("transformer_training")
    
    with mlflow.start_run() as run:
        # 记录参数
        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "model_type": "Transformer",
            "optimizer": "Adam",
            "lr": 0.001
        }
        mlflow.log_params(params)
        
        # === D. 训练循环 ===
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            
            # 🔥 2. 验证模式 (移到循环内部了！)
            model.eval()
            val_loss = 0.0
            metrics = {}
            
            with torch.no_grad():
                val_inputs = X_val_t.to(device)
                val_labels = y_val_t.to(device)
                
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                val_loss = v_loss.item()
                
                # 计算准确率 (Accuracy)
                if task_type == "classification":
                    _, preds = torch.max(val_outputs, 1)
                    acc = accuracy_score(y_val_t.numpy(), preds.cpu().numpy())
                    metrics["accuracy"] = acc
                else:
                    mse = mean_squared_error(y_val_t.numpy(), val_outputs.cpu().numpy())
                    metrics["mse"] = mse

            # 🔥 3. 实时记录日志
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {metrics.get('accuracy', 0):.4f}")
            
            # 记录到 MLflow (这样你在网页上就能看到两条曲线了)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            if "accuracy" in metrics:
                mlflow.log_metric("accuracy", metrics["accuracy"], step=epoch)
            elif "mse" in metrics:
                mlflow.log_metric("mse", metrics["mse"], step=epoch)

        # 保存模型 (保持不变)
        mlflow.pytorch.log_model(model, "model")
        
        return run.info.run_id, metrics