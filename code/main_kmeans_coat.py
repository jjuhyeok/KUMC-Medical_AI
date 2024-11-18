import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base_path = ''

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 8,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 32,
    'SEED': 42,
    'alpha': 0.6,
    'beta': 0.3
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
df = pd.read_csv(os.path.join(base_path, 'train.csv'))

df['path'] = df['path'].map(lambda x: os.path.join(base_path, x))

CFG['label_size'] = df.iloc[:,2:].values.shape[1]

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

train_transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, interpolation=0, border_mode=0, p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

class BaseModel(nn.Module):
    def __init__(self, gene_size=CFG['label_size']):
        super(BaseModel, self).__init__()
        
        self.backbone = timm.create_model(
            model_name='coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k', # coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k  xcit_large_24_p16_224.fb_in1k  convnextv2_large.fcmae_ft_in22k_in1k
            pretrained=True,
        )
        
        self.regressor = nn.Linear(1000, gene_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x



def correlation_loss(preds, targets):
    preds_mean = preds - preds.mean(dim=0)
    targets_mean = targets - targets.mean(dim=0)

    numerator = (preds_mean * targets_mean).sum(dim=0)
    denominator = torch.sqrt((preds_mean ** 2).sum(dim=0) * (targets_mean ** 2).sum(dim=0))

    correlation = numerator / (denominator + 1e-8)
    loss = 1 - correlation.mean()
    return loss

def covariance_loss(preds, targets):
    preds_mean = preds - preds.mean(dim=0)
    targets_mean = targets - targets.mean(dim=0)

    preds_cov = preds_mean.t() @ preds_mean / (preds.size(0) - 1)
    targets_cov = targets_mean.t() @ targets_mean / (targets.size(0) - 1)

    loss = nn.MSELoss()(preds_cov, targets_cov)
    return loss

def total_loss(preds, targets, alpha=CFG['alpha'], beta=CFG['beta']):
    mse = nn.MSELoss()(preds, targets)
    corr_loss = correlation_loss(preds, targets)
    cov_loss = covariance_loss(preds, targets)
    return alpha * mse + beta * corr_loss + (1 - alpha - beta) * cov_loss

def pearson_correlation(y_true, y_pred):
    # y_true와 y_pred는 (n_samples, n_targets) 형태의 numpy 배열
    correlations = []
    
    for i in range(len(y_true)):
        # 각 샘플 i에 대해 실제값과 예측값을 추출
        y_i_true = y_true[i]
        y_i_pred = y_pred[i]
        
        # 실제값과 예측값의 평균
        y_true_mean = np.mean(y_i_true)
        y_pred_mean = np.mean(y_i_pred)
        
        # 분자: (y - y_mean) * (y_hat - y_hat_mean)
        numerator = np.sum((y_i_true - y_true_mean) * (y_i_pred - y_pred_mean))
        
        # 분모: sqrt(sum((y - y_mean)^2) * sum((y_hat - y_hat_mean)^2))
        denominator = np.sqrt(np.sum((y_i_true - y_true_mean) ** 2) * np.sum((y_i_pred - y_pred_mean) ** 2))
        
        # 상관계수 계산
        if denominator != 0:
            correlation = numerator / denominator
        else:
            correlation = 0  # 분모가 0인 경우 상관계수를 0으로 처리
        
        correlations.append(correlation)
    
    # 상관계수들의 평균 반환
    return np.mean(correlations)

def max_target_correlation(y_true, y_pred):
    """
    각 타겟 j에 대해 예측값과 실제값 사이의 상관계수를 계산하고,
    가장 큰 상관계수를 반환합니다.

    Parameters:
    y_true (np.ndarray): 실제값, (n_samples, n_targets) 형태
    y_pred (np.ndarray): 예측값, (n_samples, n_targets) 형태

    Returns:
    float: 각 타겟에 대한 상관계수 중 가장 큰 값
    """
    # 각 타겟에 대한 상관계수를 저장할 리스트
    correlations = []

    # 타겟별로 상관계수 계산
    for j in range(y_true.shape[1]):
        # 타겟 j에 대한 실제값과 예측값
        y_j_true = y_true[:, j]
        y_j_pred = y_pred[:, j]

        # 타겟 j의 상관계수 계산
        y_j_true_mean = np.mean(y_j_true)
        y_j_pred_mean = np.mean(y_j_pred)

        # 분자와 분모 계산
        numerator = np.sum((y_j_true - y_j_true_mean) * (y_j_pred - y_j_pred_mean))
        denominator = np.sqrt(np.sum((y_j_true - y_j_true_mean) ** 2) * np.sum((y_j_pred - y_j_pred_mean) ** 2))

        # 상관계수 계산 (안정성을 위해 작은 값 추가)
        correlation = numerator / (denominator + 1e-8)
        correlations.append(correlation)

    # 타겟별 상관계수 중 최댓값 반환
    max_correlation = np.max(correlations)

    return max_correlation


def train(model, optimizer, train_loader, val_loader, scheduler, criterion, device):
    model.to(device)
    
    best_score = 0
    best_model = None
    best_corr = []
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            # loss = criterion(output, labels)
            loss = total_loss(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_loss2, corr, max_corr = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Loss2 : [{_val_loss2:.5f}] Val_Corr: [{corr}] Val_MaxCorr: [{max_corr}]')
        score = (corr + max_corr) / 2

        if scheduler is not None:
            scheduler.step(_val_loss2)
            
        if best_score < score:
            best_score = score
            best_model = model
            best_corr = [corr, max_corr]
    
    return best_model, best_score, best_corr

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_loss2 = []

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            other_loss = total_loss(pred, labels)
            
            val_loss.append(loss.item())
            val_loss2.append(other_loss.item())

            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())
        
        _val_loss = np.mean(val_loss)
        _val_loss2 = np.mean(val_loss2)
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    corr = pearson_correlation(y_true, y_pred)
    max_corr = max_target_correlation(y_true, y_pred)
    
    return _val_loss, _val_loss2, corr, max_corr

test = pd.read_csv(os.path.join(base_path, 'test.csv'))

test['path'] = test['path'].map(lambda x: os.path.join(base_path, x))

test_dataset = CustomDataset(test['path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device).float()
            pred = model(imgs)
            
            preds.append(pred.detach().cpu())
    
    preds = torch.cat(preds).numpy()

    return preds

from sklearn.cluster import KMeans

y_cols = df.iloc[:,2:].T

num_clusters = 5  # 원하는 클러스터 수 설정
kmeans = KMeans(n_clusters=num_clusters, random_state=CFG['SEED'])
y_cols['clusters'] = kmeans.fit_predict(y_cols)

from sklearn.model_selection import KFold

kf = KFold(n_splits=7, shuffle=True, random_state=CFG['SEED'])

submits = []

scores = []
for y_idx in range(num_clusters):

    labels = y_cols[y_cols['clusters'] == y_idx].index
    tmp_df = df[labels]
    tmp_df['path'] = df['path']
    
    results = []
    val_scores = []
    criterion = nn.MSELoss().to(device)
    
    for f, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f'----Cluster: {y_idx} Fold: {f}----')
        train_df = tmp_df.iloc[train_index].reset_index(drop=True)
        val_df = tmp_df.iloc[valid_index].reset_index(drop=True)

        train_label_vec = train_df.drop(columns=['path']).values.astype(np.float32)
        val_label_vec = val_df.drop(columns=['path']).values.astype(np.float32)

        train_dataset = CustomDataset(train_df['path'].values, train_label_vec, train_transform)
        train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

        val_dataset = CustomDataset(val_df['path'].values, val_label_vec, test_transform)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

        model = BaseModel(gene_size=len(labels))
        model.eval()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

        infer_model, best_loss, best_corr = train(model, optimizer, train_loader, val_loader, scheduler, criterion, device)

        preds = inference(infer_model, test_loader, device)

        results.append(preds)
        val_scores.append([best_loss, best_corr[0], best_corr[1]])

        del model, infer_model
        torch.cuda.empty_cache()
        
    preds = np.mean(results, axis=0)
    
    submit = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
    submit[labels] = np.array(preds).astype(np.float32)
    submits.append(submit)

    scores.append(np.mean(val_scores, axis=0))

submit = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

for y_idx in range(num_clusters):
    labels = y_cols[y_cols['clusters'] == y_idx].index
    submit[labels] = submits[y_idx][labels]
    
submit.to_csv('submit_corrcovloss631_coatnet.csv', index=False)
