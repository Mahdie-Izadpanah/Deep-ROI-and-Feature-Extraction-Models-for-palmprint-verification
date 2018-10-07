import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib.image as mp

from PIL import Image
import pandas as pd


def flip_image(image_path, save):

    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image = rotated_image.transpose(Image.FLIP_TOP_BOTTOM)
    rotated_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(save)
    # rotated_image.show()


names = [name for name in os.listdir('./data/') if int(re.findall('[0-9]+', name)[0]) % 2 == 0]
# for name in names:
    # plt.imshow(mp.imread('./org_data/' + name))
    # plt.show()
    # flip_image('./data/' + name, './left_hand_full/' + name)


data = pd.read_csv('final_left_hand.csv')
names = [name for name in os.listdir('./data/') if name[:name.rfind('.')]+'.jpg' in data['Name'].values]
print(names)
data = data.set_index('Name')


for image in names:
    im = mp.imread('./left_hand_full/' + image)[:,66:350]
    print 'shape:', np.shape(im)
    record = data.loc[image[:image.rfind('.')]+'.jpg']

    x1 = record['X1']
    x2 = record['X2']
    y1 = record['Y1']+66
    y2 = record['Y2']+66
    print(im.shape)
    print(int(x1),int(x2),int(y1),int(y2))
    im = im[int(x1):int(x2),int(y1):int(y2)]
    mp.imsave('./final_left_hand_data/' + image[:image.rfind('.')]+'.jpg' , im, format='jpg')