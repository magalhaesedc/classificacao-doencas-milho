from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import numpy as np
import imageio

def getImgByClass(path, category):
    files = glob.glob(path)
    X = []
    y = []
    i = 0
    for f in files:
        i+=1
        if i % 3 != 0:
            continue
        img = imageio.imread(f)
        #resized = cv2.resize(img, (150, 150))
        X.append(img)
        #X.append(img)
        y.append([category])
    return X, y

def loadData():
    print("Loading data...")
    X1, y1 = getImgByClass(os.getcwd()+"/cercosporiose/*", 0)
    
    X = np.array(X1)
    y = np.array(y1)
    print(X.shape)
    return X, y

X, y = loadData()

datagen = ImageDataGenerator(
    rotation_range = 7,
    horizontal_flip = True,
    shear_range = 0.2,
    height_shift_range = 0.07,
    zoom_range = 0.2)

count = 0
for batch in datagen.flow(X, batch_size=1,save_to_dir="cercosporiose_generator", save_prefix='cercosporiose', save_format='jpg'):
    count += 1
    if count > 513:
        break