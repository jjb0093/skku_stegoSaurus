import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os, pickle
from PIL import Image
from tqdm import tqdm

from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from dataComprimer_structure_new import DataComprimer

class StegoDataset(Dataset):
    def __init__(self, root, stegoType, dataType = 'train'):
        if(dataType == 'train'):
            self.cover = root + "train/cover/"
            self.stego = root + "train/stego/"
        elif(dataType == 'val'):
            self.cover = root  + "val/cover/"
            self.stego = root + "val/stego/"
        self.stegoType = stegoType

        self.indices = []
        for i in range(1, 10000):
            coverPath = self.cover + str(i) + ".pgm"
            if(os.path.exists(coverPath)):
                for k in range(len(stegoType)):
                    print(f"Processing image {i} for stego type {stegoType[k]}")
                    stegoPath = self.stego + self.stegoType[k] + str(i) + ".pgm"
                    if(not os.path.exists(stegoPath)): break

                    if(k == 0): self.indices.append((coverPath, 0))
                    self.indices.append((stegoPath, 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path, label = self.indices[index]
        img = Image.open(path)
        img_np = np.array(img, dtype = np.float32) / 255.0
        img_tensor = torch.tensor(img_np).unsqueeze(0)

        label_tensor = torch.tensor(label, dtype = torch.long)
        return img_tensor, label_tensor

stegoType = ['HILL/0.2bpp/', 'HILL/0.4bpp/', 'HUGO/0.2bpp/', 'HUGO/0.4bpp/', 'MiPOD/0.2bpp/', 'MiPOD/0.4bpp/', 'S-UNIWARD/0.2bpp/', 'S-UNIWARD/0.4bpp/', 'WOW/0.2bpp/', 'WOW/0.4bpp/']

def extractFeature(model, dataLoader):
    x, y = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataLoader, total = len(dataLoader)):
            images = images.to(device)

            result = model(images, True)
            x.append(result.cpu().numpy())
            y.append(labels.numpy())

        x = np.concatenate(x, axis = 0)
        y = np.concatenate(y, axis = 0)

        return x, y

device = "cuda"
model = DataComprimer(outputDim = 128).to(device)

ckpt = torch.load("F:/csdfLab/2. StegoSaurus/model/dataComprimer2.pth", map_location = device)
model.layers.load_state_dict(ckpt["CNN"])
model.fc.load_state_dict(ckpt["Fc"])

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="gpu_hist",
    n_jobs=-1
)

lgbm = lgb.LGBMClassifier(
    n_estimators=400,
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary",
    n_jobs=-1
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced"
)

trainDataset = StegoDataset("F:/BOSSbase-1.01/", stegoType)
trainLoader = DataLoader(trainDataset, batch_size = 32, shuffle = False)

valDataSet = StegoDataset("F:/BOSSbase-1.01/", stegoType, dataType = 'val')
valLoader = DataLoader(valDataSet, batch_size = 32, shuffle = False)

x_train, y_train = extractFeature(model, trainLoader)
x_val, y_val = extractFeature(model, valLoader)

print("XGB 훈련 시작")
xgb.fit(x_train, y_train, verbose = 10)
with open("F:/csdfLab/2. StegoSaurus/model/xgb.pkl" ,'wb') as f:
    pickle.dump(xgb, f)

print("LGBM 훈련 시작")
lgbm.fit(x_train, y_train)
with open("F:/csdfLab/2. StegoSaurus/model/lgbm.pkl", 'wb') as f:
    pickle.dump(lgbm, f)

print("RF 훈련 시작")
rf.fit(x_train, y_train)
with open("F:/csdfLab/2. StegoSaurus/model/rf.pkl", 'wb') as f:
    pickle.dump(rf, f)

for model, name in [(xgb, "XGB"), (lgbm, "LGBM"), (rf, "RF")]:
    pred = model.predict(x_val)
    prob = model.predict_proba(x_val)[:, 1]

    print(f"모델 : {model} -> Acc Score : {accuracy_score(y_val, pred)} / Roc Score : {roc_auc_score(y_val, prob)}")

    with open("training_log.txt", 'a') as f:
        f.write(f"모델 : {model} -> ")
        f.write(f"Acc Score: {accuracy_score(y_val, pred)}, Roc Score: {roc_auc_score(y_val, prob)}\n")
        f.write("-" * 30 + "\n")