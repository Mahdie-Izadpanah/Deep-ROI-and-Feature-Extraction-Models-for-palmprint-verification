import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.image as mp


def draw_box(base_dir, name, points):
    image = mp.imread(base_dir + name)
    plt.imshow(image, cmap='gray')
    plt.plot([points[0], points[2]], [points[1], points[1]], c='g')
    plt.plot([points[2], points[2]], [points[1], points[3]], c='g')
    plt.plot([points[2], points[0]], [points[3], points[3]], c='g')
    plt.plot([points[0], points[0]], [points[3], points[1]], c='g')
    plt.show()


data = pd.read_csv('left_hand_box.csv')

names = [name for name in os.listdir('./left_hand/') if name in data['Name'].values]

print(names)

data = data.set_index(['Name'])
for image in names:
    plt.imshow(mp.imread('./left_hand/' + image))

    draw_box('./left_hand/', image, data.loc[image][['X1', 'Y1', 'X2', 'Y2']].values)
#
# data['X1'] *= 284/224
# data['X2'] *= 284/224
# data['Y1'] *= 284/224
# data['Y2'] *= 284/224
#
# data.to_csv('final_left_hand.csv',index=False)