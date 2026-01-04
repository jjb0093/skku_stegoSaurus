import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, datetime, time

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
                    self.indices.append((stegoPath, k+1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path, label = self.indices[index]
        img = Image.open(path)
        img_np = np.array(img, dtype = np.float32) / 255.0
        img_tensor = torch.tensor(img_np).unsqueeze(0)

        label_tensor = torch.tensor(label, dtype = torch.long)
        return img_tensor, label_tensor

if __name__ == "__main__" :
    stegoType = ['HILL/0.2bpp/', 'HILL/0.4bpp/', 'HUGO/0.2bpp/', 'HUGO/0.4bpp/', 'MiPOD/0.2bpp/', 'MiPOD/0.4bpp/', 'S-UNIWARD/0.2bpp/', 'S-UNIWARD/0.4bpp/', 'WOW/0.2bpp/', 'WOW/0.4bpp/']

    trainDataset = StegoDataset("F:/BOSSbase-1.01/", stegoType)
    trainLoader = DataLoader(trainDataset, batch_size = 32, shuffle = True)

    valDataSet = StegoDataset("F:/BOSSbase-1.01/", stegoType, dataType = 'val')
    valLoader = DataLoader(valDataSet, batch_size = 32, shuffle = False)

    print(len(trainLoader), len(valLoader))
    time.sleep(5)

    device = "cuda"

    model = DataComprimer(outputDim = 128)
    model = model.to(device)

    weights = torch.tensor([10] + [1] * 10, dtype = torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scaler = torch.amp.GradScaler()

    epochs = 10

    train_loss_arr, val_loss_arr = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        for batchIdx, (images, labels) in enumerate(trainLoader):
            #if((batchIdx+1) % 50 == 0):
            print(f"{epoch+1}/{epochs} 에포크 중 {batchIdx+1}/{len(trainLoader)} 번째 배치 처리 중..")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()    

            train_loss += loss.item()

        train_loss /= len(trainLoader)
        train_loss_arr.append(train_loss)
        
        val_loss = 0
        model.eval()
        
        with torch.no_grad():
            for images, labels in valLoader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            val_loss /= len(valLoader)
            val_loss_arr.append(val_loss)

            '''
            if(val_loss < best_val_loss):
                best_val_loss = val_loss
            '''

            torch.save({
                "CNN": model.layers.state_dict(),
                "Fc": model.fc.state_dict()
            }, f"model/dataComprimer3/{epoch + 1}.pth")

        print(f"Epoch [{epoch+1}/10]")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("-" * 30)

        now = datetime.datetime.now()
        with open("training_log.txt", 'a') as f:
            f.write(f"Epoch [{epoch+1}/10] - {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
            f.write("-" * 30 + "\n")

    plt.figure()
    plt.plot(train_loss_arr, label = "Train Loss")
    plt.plot(val_loss_arr, label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("lossCurve.png")
    plt.close()