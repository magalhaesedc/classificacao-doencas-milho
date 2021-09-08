import cv2
import numpy as np
import glob
import imageio
import os

def getImgByClass(path, category):
    files = glob.glob(path)
    X = []
    y = []
    for f in files:
        img = imageio.imread(f)
        resized = cv2.resize(img, (150, 150))
        X.append(resized)
        y.append([category])
    return X, y

def loadData():
    print("Loading data...")
    (X1, y1) = getImgByClass(os.getcwd()+"/banco_imagens/treino/saudavel/*", 0)
    (X2, y2) = getImgByClass(os.getcwd()+"/banco_imagens/treino/mancha/*", 1)
    (X3, y3) = getImgByClass(os.getcwd()+"/banco_imagens/treino/ferrugem/*", 2)
    (X4, y4) = getImgByClass(os.getcwd()+"/banco_imagens/treino/cercosporiose/*", 3)

    X = np.concatenate([X1,X2,X3,X4], axis=0)
    y = np.concatenate([y1,y2,y3,y4], axis=0)
    print(X.shape)
    return X, y

def loadDataUnit():
    print("Loading data...")
    (X4, y4) = getImgByClass(os.getcwd()+"/banco_imagens/teste/cercosporiose/1afa65e8-15a9-46d4-b612-e7a7b23b10b5___RS_GLSp 7326.JPG", 3)
    X = np.concatenate([X4], axis=0)
    y = np.concatenate([y4], axis=0)
    print(X.shape)
    return X, y

def loadDataTest():
    print("Loading data test...")
    (X1, y1) = getImgByClass(os.getcwd()+"/banco_imagens/teste/saudavel/*", 0)
    (X2, y2) = getImgByClass(os.getcwd()+"/banco_imagens/teste/mancha/*", 1)
    (X3, y3) = getImgByClass(os.getcwd()+"/banco_imagens/teste/ferrugem/*", 2)
    (X4, y4) = getImgByClass(os.getcwd()+"/banco_imagens/teste/cercosporiose/*", 3)

    X = np.concatenate([X1,X2,X3,X4], axis=0)
    y = np.concatenate([y1,y2,y3,y4], axis=0)
    print(X.shape)
    return X, y