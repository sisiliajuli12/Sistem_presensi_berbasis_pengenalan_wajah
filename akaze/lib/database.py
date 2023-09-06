import pandas as pd
import pickle
import os
import cv2
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
pathDB = f'{dir_path}\\..\\data\\faceDB.csv'


def get_data():
    df = pd.read_csv(pathDB)
    return df.to_list()


def get_attendances():
    df = pd.read_csv(pathDB)
    return list(dict.fromkeys(df['name'].values.tolist()))


def save_data(name, desc=None, img=None):
    df = pd.read_csv(pathDB)
    filename = None
    if desc is not None:
        i = 1
        while True:
            if os.path.isfile(f'{dir_path}\\..\\data\\{name}{i}.pkl') :
                i += 1
            else :
                break
        
        filename = f'{name}{i}.pkl'
        with open(f'{dir_path}\\..\\data\\{filename}', 'wb') as f:
            pickle.dump(desc, f)
        if img is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{dir_path}\\..\\data\\image\\{name}{i}.png', img)
            print("name", name)
    df.loc[len(df.index)] = [name, filename]

    df.to_csv(pathDB, index=False)
