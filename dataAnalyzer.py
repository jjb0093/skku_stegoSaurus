from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

root = "D:/BOSSbase-1.01/"

stegoType = ['HILL/0.2bpp/', 'HILL/0.4bpp/', 'HUGO/0.2bpp/', 'HUGO/0.4bpp/', 'MiPOD/0.2bpp/', 'MiPOD/0.4bpp/', 'S-UNIWARD/0.2bpp/', 'S-UNIWARD/0.4bpp/', 'WOW/0.2bpp/', 'WOW/0.4bpp/']
trainRoot = root + "train/"
testRoot = root + "test/"
valRoot = root + "val/"

diffMean = []
threshold = [0.0001, 0.0003, 0.0002, 0.0003, 0.0001, 0.0003, 0.0001, 0.0003, 0.0001, 0.0003]

for i in range(len(stegoType)):
    heatMap = np.zeros((256, 256), dtype = np.int32)
    diffArr = []

    for j in range(1, 10000):
        path = trainRoot + "cover/" + str(j) + ".pgm"
        print(path)

        if(os.path.exists(path)):
            cover = Image.open(path)
            cover_np = np.array(cover, dtype = np.float32) / 255.0

            stegoPath = trainRoot + "stego/" + stegoType[i] + str(j) + ".pgm"

            if(not os.path.exists(stegoPath)):
                break
            
            stego = Image.open(stegoPath)
            stego_np = np.array(stego, dtype = np.float32) / 255.0

            diff = np.abs(stego_np - cover_np)
            diffArr.append(diff)
            heatMap += (diff >= 0.0001).astype(np.int32)

    print(f"Stego Type : {stegoType[i][:-1]}, 평균 차이 : {np.mean(diffArr)}")
    diffMean.append(np.mean(diffArr))

    plt.figure()
    plt.imshow(heatMap, cmap = 'hot')
    plt.title(stegoType[i][:-1])
    plt.colorbar()
    plt.savefig(f"heatmap/{i+1}_{stegoType[i][:-1].replace("/", "_")}.png")

for i in range(len(stegoType)):
    print(f"Stego Type : {stegoType[i][:-1]}, 평균 차이 : {diffMean[i]}")

"""
Stego Type : MiPOD/0.4bpp, 평균 차이 : 0.00037042732583358884
Stego Type : S-UNIWARD/0.2bpp, 평균 차이 : 0.00013893791765440255
Stego Type : S-UNIWARD/0.4bpp, 평균 차이 : 0.00030913049704395235
Stego Type : WOW/0.2bpp, 평균 차이 : 0.00017235353880096227
Stego Type : WOW/0.4bpp, 평균 차이 : 0.0003789231996051967
"""