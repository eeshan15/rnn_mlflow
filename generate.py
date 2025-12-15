import torch
import numpy as np
import pandas as pd
import os 

T = 20 ; L = 1000 ; N = 100 

os.makedirs("data" , exist_ok = True)

x = np.empty((N , L) ,'int64')
x[:] = np.array(range(L)) + np.random.randint(-4*T,4*T,N).reshape(N,1)
data = np.sin(x / 1.0 / T).astype("float32")

df = pd.DataFrame(data)
df.to_csv("data/sine_wave.csv" , index = False)

print("Data Generated !! ")