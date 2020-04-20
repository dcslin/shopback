import json
import os.path
import skimage.io
import skimage.transform
import numpy as np
import pickle

##
filename = "outfits_with_products.json"

f = open(filename, 'r')
outs = json.load(f)
f.close()

count = 0
# cg = set()
# {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None}
# 1 glasses
# 2 bag
# 3 bottom
# 4 top
# 5 top bottom one piece
# 6 coat
# 7 like bra
# 8 accessories
# 9 high hills
# 10 makeup
prefix="product_images/"
top_resize_size=(100,100,3) # h*w
bottom_resize_size=(100,100,3) # h*w
top_imgs = []
bottom_imgs = []
for o in outs:
    # print(o['outfit_id'])
    top = None
    bottom = None
    for p in o['products']:
        # cg.add(p['category_group'])
        if p['category_group'] == 4:
            top = p
        if p['category_group'] == 3:
            bottom = p
            # print(p['title'], p['image'])

    if top and bottom:
        #skimage.io.imread(top['image'])

        try:
            assert os.path.isfile(prefix+str(top['id'])+".jpg")
            print(prefix+str(top['id'])+".jpg")
            print(top['image'])
            top_img = skimage.io.imread(prefix+str(top['id'])+".jpg")
            top_img = skimage.transform.resize(top_img, top_resize_size)

            assert os.path.isfile(prefix+str(bottom['id'])+".jpg")
            print(prefix+str(bottom['id'])+".jpg")
            print(bottom['image'])
            bottom_img = skimage.io.imread(prefix+str(bottom['id'])+".jpg")
            bottom_img = skimage.transform.resize(bottom_img, bottom_resize_size)

            # print(top['id'], bottom['id'])
        except:
            print("malformat image")
            continue

        # skimage.io.imsave(str(bottom['id'])+"_s.jpg", bottom_img)
        # skimage.io.imsave(str(top['id'])+"_s.jpg", top_img)
        # print(top_img.shape)
        # print(bottom_img.shape)
        top_imgs.append(top_img)
        bottom_imgs.append(bottom_img)
        count+=1

    if count > 10000:
        top_imgs = np.asarray(top_imgs)
        bottom_imgs = np.asarray(bottom_imgs)
        print(top_imgs.shape)
        print(bottom_imgs.shape)
        with open('top_bottom_imgs.pickle', 'wb') as handle:
            pickle.dump((top_imgs,bottom_imgs), handle, protocol=pickle.HIGHEST_PROTOCOL)
        exit()

# print(cg)
