import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.image as mp
import numpy as np


def draw_box(base_dir, name, points):
    image = mp.imread(base_dir + name)
    plt.imshow(image[:, 66:350])
    plt.plot([points[0], points[2]], [points[1], points[1]], c='g')
    plt.plot([points[2], points[2]], [points[1], points[3]], c='g')
    plt.plot([points[2], points[0]], [points[3], points[3]], c='g')
    plt.plot([points[0], points[0]], [points[3], points[1]], c='g')
    plt.show()


data = pd.read_csv('left_hand_box.csv')
names = [name for name in os.listdir('./original_data/') if name[:name.rfind('.')] + '.jpg' in data['Name'].values]
print(names)
data = data.set_index('Name')
print(data)
for image in names:
    im = mp.imread('./left_hand_full/' + image[:image.rfind('.')] + '.bmp')[:, 66:350]
    print 'shape:', np.shape(im)
    record = data.loc[image[:image.rfind('.')] + '.jpg'] * 284 / 224

    x1 = record['X1']
    x2 = record['X2']
    y1 = record['Y1']
    y2 = record['Y2']
    print(im.shape)

    points = [x1, y1, x2, y2]
    plt.imshow(im)
    plt.plot([points[0], points[2]], [points[1], points[1]], c='k', linewidth=4)
    plt.plot([points[2], points[2]], [points[1], points[3]], c='k', linewidth=4)
    plt.plot([points[2], points[0]], [points[3], points[3]], c='k', linewidth=4)
    plt.plot([points[0], points[0]], [points[3], points[1]], c='k', linewidth=4)
    plt.show()
