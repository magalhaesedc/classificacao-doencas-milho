import glob
import shutil
import numpy as np
import os

path_data = os.getcwd()+'/banco_imagens/treino/'
path_teste =  os.getcwd()+'/banco_imagens/teste/'

def splitData(folders, percent):
    resetData(folders)
    for folder in folders:
        files = glob.glob(path_data+folder+'/*.*')
        n = len(files) / 100 * percent
        # print(len(files))
        imgs = np.random.choice(files, int(n), False)
        for img in imgs:
            shutil.move(img, path_teste+folder)

def resetData(folders):
    for folder in folders:
        files = glob.glob(path_teste+folder+'/*.*')
        for img in files:
            shutil.move(img, path_data+folder)

# splitData(["cercosporiose","ferrugem", "mancha", "saudavel"], 15)