#
#   Fully Convolutional Networks: Data Augmentation
#   Written by Qhan
#

import numpy as np
#from PIL import Image
import cv2

s = lambda x: max(0, x)
e = lambda x: x if x < 0 else None

def augment(np_image, flip_prob, aug_type, randoms):
    # annotation 3rd dim is 1 -> need to shrink shape to 2 dims for PIL
    if np_image.shape[2] == 1: # annotation
        im = np.array(np_image[:, :, 0])
    else: # image
        im = np.array(np_image)

    if flip_prob >= 0.5: 
        im = im[:, ::-1]

    h, w = im.shape[:2]
    #pil_im = Image.fromarray(im.astype(np.uint8))

    # zoom: 1 ± 0.5
    if aug_type == 0:
        max_scale = 0.5
        zoom = (randoms[0] * 2 - 1) * max_scale + 1
        new_h, new_w = int(h * zoom), int(w * zoom)
        pad_x, pad_y = int((new_w - w) / 2), int((new_h - h) / 2)
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        im = im[pad_y:pad_y+h, pad_x:pad_x+w]
        #pil_im = pil_im.resize((new_w, new_h), Image.BICUBIC).crop((pad_x, pad_y, pad_x + w, pad_y + h))
        #im = np.array(pil_im)

    # rotation: ± 90
    elif aug_type == 1:
        max_angle = 90
        angle = ( randoms[0] * 2 - 1 ) * max_angle
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        im = cv2.warpAffine(im, M, (w, h))
        #pil_im = pil_im.rotate(angle, resample=Image.BICUBIC)
        #im = np.array(pil_im)

    # horizontal and vertical shift: ± 50%
    elif aug_type == 2:
        max_sft = 0.5
        max_dx, max_dy = int(w * max_sft), int(h * max_sft)
        dx = int( randoms[0] * 2 - 1 ) * max_dx
        dy = int( randoms[1] * 2 - 1 ) * max_dy
        sft_im = np.zeros_like(im)
        sft_im[s(dy):e(dy), s(dx):e(dx)] = im[s(-dy):e(-dy), s(-dx):e(-dx)] # crop
        im = sft_im

    else:
        pass
    
    if np_image.shape[2] == 1: # annotation
        result_im = np.array(np_image)
        result_im[:, :, 0] = im
        return result_im
    else: # image
        return im
