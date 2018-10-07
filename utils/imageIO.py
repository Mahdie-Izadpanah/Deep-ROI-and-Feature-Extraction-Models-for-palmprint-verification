import os
import matplotlib.image as mp
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

base_dir = './org_data/'
d = pd.read_csv('Data.csv')
d = d.set_index('Name')


def pure_list(l):
    l_ = [name[:name.rfind('.')] for name in l]
    return l_


def in_list(name, l):
    if name[:name.rfind('.')] in pure_list(l):
        return True
    else:
        return False


def draw_box(name):
    image = mp.imread(base_dir + name)
    points = d.loc[name].values
    points[0] -= 66
    points[2] -= 66
    points = d.loc[name].values * 224 / 284
    plt.imshow(image, cmap='gray')
    plt.plot([points[0], points[2]], [points[1], points[1]], c='g')
    plt.plot([points[2], points[2]], [points[1], points[3]], c='g')
    plt.plot([points[2], points[0]], [points[3], points[3]], c='g')
    plt.plot([points[0], points[0]], [points[3], points[1]], c='g')
    plt.show()


# g = pd.read_csv('Data.csv')
# names = [name for name in os.listdir('./final/') if in_list(name, g['Name'].values)]
# print(names)
# print(names[1])
# draw_box(names[3])
# g['X1'] -= 66
# g['X2'] -= 66
# g['X1'] *= 224 / 284
# g['X2'] *= 224 / 284
# g['Y1'] *= 224 / 284
# g['Y2'] *= 224 / 284
# g.to_csv('NData.csv', index=False)

# plt.imshow(mp.imread('./org_data0/' + names[0]))
# print(np.shape(mp.imread('./org_data0/' + names[0])))
# plt.show()
for name in os.listdir('./final_left_hand_data/'):
    # im = mp.imread('./final/' + name)[:, 66:350]
    # n = name[:name.rfind('.')]
    # n = n[:n.rfind('.')]
    #
    # mp.imsave('./final2/' + n, im, format='jpg')
    image = Image.open('./final_left_hand_data/' + name)
    im = image.resize((224, 224), Image.NEAREST)
    im.save('./resized_final_left_hand_data/' + name + '.jpg')
