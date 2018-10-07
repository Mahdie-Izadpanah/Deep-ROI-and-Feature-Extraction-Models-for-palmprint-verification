import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.image as mp

data = pd.read_csv('final_left_hand.csv')
names = [name for name in os.listdir('./original_data/') if name[:name.rfind('.')]+'.jpg' in data['Name'].values]
print(names)
data = data.set_index('Name')
print(data)
for image in names:
    im = mp.imread('./original_data/' + image)[:,66:350]
    print 'shape:', np.shape(im)
    record = data.loc[name[:name.rfind('.')]+'.jpg']

    x1 = record['X1']
    x2 = record['X2']
    y1 = record['Y1']+66
    y2 = record['Y2']+66
    print(im.shape)
    print(int(x1),int(x2),int(y1),int(y2))
    plt.imshow(im[int(x1):int(x2),int(y1):int(y2)])
    plt.show()