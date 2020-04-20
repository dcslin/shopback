# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


def random_index_no_inplace(length):
    l = list(range(length))
    random.shuffle(l)
    ret = []
    idx = 0
    while l:
        val = l.pop(0)
        if idx != val:
            idx +=1
            ret.append(val)
        else:
            l.append(val)
    return ret


with open('top_bottom_imgs.pickle', 'rb') as handle:
    (top_imgs, bottom_imgs) = pickle.load(handle)

batch=top_imgs.shape[0]
shuffle_bottom_imgs = bottom_imgs[random_index_no_inplace(batch)]

shuffle_idx = random_index_no_inplace(batch*2)

train_top_img = np.concatenate((top_imgs, np.copy(top_imgs)))[shuffle_idx]
print("concated top imgs", train_top_img.shape)

train_bottom_img = np.concatenate((bottom_imgs, shuffle_bottom_imgs))[shuffle_idx]
print("concated bottom imgs", train_bottom_img.shape)

train_label = [1] * batch + [0] * batch
train_label = np.asarray(train_label)[shuffle_idx]
print("label", train_label.shape)


with open('training_data.pickle', 'wb') as handle:
    pickle.dump((train_top_img, train_bottom_img, train_label), handle, protocol=pickle.HIGHEST_PROTOCOL)
