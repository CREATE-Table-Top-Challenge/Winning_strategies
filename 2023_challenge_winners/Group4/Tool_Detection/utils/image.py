import cv2
import numpy as np
import copy
import random

def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1./scale;

def _constrain(min_v, max_v, value):
    if value < min_v: return min_v
    if value > max_v: return max_v
    return value 

def random_flip(image, flip,axis):
    if flip == 1: return cv2.flip(image, axis)
    return image

def random_rotate90(image,angle):
    if angle>0:
        for i in range(angle):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, axis,angle,image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1 and axis==1:
            swap = boxes[i]['xmin'];
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap
        elif flip ==1 and axis==0:
            swap = boxes[i]['ymin'];
            boxes[i]['ymin'] = net_w - boxes[i]['ymax']
            boxes[i]['ymax'] = net_w - swap

        old_xmin = boxes[i]["xmin"]
        old_xmax = boxes[i]["xmax"]
        old_ymin = boxes[i]["ymin"]
        old_ymax = boxes[i]["ymax"]
        if angle == 1:
            # 90 degree rotation
            boxes[i]["ymin"] = old_xmin
            boxes[i]["ymax"] = old_xmax
            boxes[i]["xmin"] = net_w - old_ymax
            boxes[i]["xmax"] = net_w - old_ymin
        elif angle == 2:
            # 180 degree rotation
            boxes[i]["ymin"] = net_h - old_ymax
            boxes[i]["ymax"] = net_h - old_ymin
            boxes[i]["xmin"] = net_w - old_xmax
            boxes[i]["xmax"] = net_w - old_xmin
        elif angle == 3:
            # 270 degree rotation
            boxes[i]["ymin"] = net_w - old_xmax
            boxes[i]["ymax"] = net_w - old_xmin
            boxes[i]["xmin"] = old_ymin
            boxes[i]["xmax"] = old_ymax

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation);
    dexp = _rand_scale(exposure);     

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    
    # change satuation and exposure
    image[:,:,1] *= dsat
    image[:,:,2] *= dexp
    
    # change hue
    image[:,:,0] += dhue
    image[:,:,0] -= (image[:,:,0] > 180)*180
    image[:,:,0] += (image[:,:,0] < 0)  *180
    
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))
    
    if dx > 0: 
        im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:,-dx:,:]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)
               
    if dy > 0: 
        im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:,:,:]
        
    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)
        
    return im_sized[:net_h, :net_w,:]     