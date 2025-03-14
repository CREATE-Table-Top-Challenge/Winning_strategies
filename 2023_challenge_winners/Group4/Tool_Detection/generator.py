import gc
import cv2
import copy
import time
import numpy as np
from tensorflow.keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, random_rotate90,correct_bounding_boxes

class BatchGenerator(Sequence):
    def __init__(self, 
        instances, 
        anchors,   
        labels,        
        downsample=32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=30,
        batch_size=1,
        min_net_size=320,
        max_net_size=608,    
        shuffle=True, 
        jitter=True, 
        norm=None,
        modeName="Test",
        maxCount=-1
    ):
        self.modeName           = modeName
        self.instances          = instances
        self.batch_size         = batch_size
        self.labels             = labels
        self.downsample         = downsample
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//self.downsample)*self.downsample
        self.max_net_size       = (max_net_size//self.downsample)*self.downsample
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.norm               = norm
        self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]
        self.net_h              = 416  
        self.net_w              = 416
        self.batch_num          = 0

        if shuffle: np.random.shuffle(self.instances)
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    def __getitem__(self, idx):
        self.endTime = time.time()
        totalNumBatches = len(self.instances)//self.batch_size
        if self.batch_num == 0:
            numEqualSigns = 0
            numArrows = 0
            self.startTime = time.time()
            self.totalBatchTime = 0
        else:
            numEqualSigns = int((round(self.batch_num/totalNumBatches*50)) -1)
            numArrows=1
        self.totalBatchTime += self.endTime - self.startTime
        if not self.batch_num == 0:
            timeRemainingSec = round((self.totalBatchTime/self.batch_num)*(totalNumBatches-self.batch_num))
        else:
            timeRemainingSec = 5000
        numMinutes = int(timeRemainingSec //60)
        numSeconds = int(timeRemainingSec % 60)
        numDashes = 50 - numEqualSigns-numArrows
        self.startTime = time.time()
        print("{}/{} [{}{}{}] Est:{}:{}".format(str(self.batch_num).rjust(len(str(totalNumBatches))),totalNumBatches,"="*numEqualSigns,">"*numArrows,"-"*numDashes,str(numMinutes).rjust(4),str(numSeconds).zfill(2)))
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))             # input images
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1,  self.max_box_per_image, 4))   # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))
        
        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
                # augment input image and fix object's position and size
                img, all_objs = self._aug_image(train_instance, net_h, net_w)

                for obj in all_objs:
                    # find the best anchor box for this object
                    max_anchor = None
                    max_index  = -1
                    max_iou    = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           obj['xmax']-obj['xmin'],
                                           obj['ymax']-obj['ymin'])

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou    = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            max_anchor = anchor
                            max_index  = i
                            max_iou    = iou

                    # determine the yolo to be responsible for this bounding box
                    yolo = yolos[max_index//3]
                    grid_h, grid_w = yolo.shape[1:3]

                    # determine the position of the bounding box on the grid
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y

                    # determine the sizes of the bounding box
                    w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax)) # t_w
                    h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax)) # t_h

                    box = [center_x, center_y, w, h]

                    # determine the index of the label
                    obj_indx = self.labels.index(obj['name'])

                    # determine the location of the cell responsible for this object
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    yolo[instance_count, grid_y, grid_x, max_index%3]      = 0
                    yolo[instance_count, grid_y, grid_x, max_index%3, 0:4] = box
                    yolo[instance_count, grid_y, grid_x, max_index%3, 4  ] = 1.
                    yolo[instance_count, grid_y, grid_x, max_index%3, 5+obj_indx] = 1

                    # assign the true box to t_batch
                    true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                    t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                    true_box_index += 1
                    true_box_index  = true_box_index % self.max_box_per_image
                    del w
                    del h
                    del grid_x
                    del grid_y
                # assign input image to x_batch
                if self.norm != None:
                    x_batch[instance_count] = self.norm(img)
                else:
                    # plot image and bounding boxes for sanity check
                    for obj in all_objs:
                        cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img, obj['name'],
                                    (obj['xmin']+2, obj['ymin']+12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0,255,0), 2)

                    x_batch[instance_count] = img

                # increase instance counter in the current batch
                instance_count += 1
                del img
        self.batch_num +=1
        gc.collect()
        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w
    
    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image_name = image_name.replace("b", "", 1)
        image_name = image_name.replace("ch", "", 1)
        image = cv2.imread(image_name)  # RGB image

        if image is None: print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image

        image_h, image_w, _ = image.shape

        #image = cv2.resize(image,(int(round(480 * (520.0/640.0))),520))
        #image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        if self.modeName == "Train":
            dw = self.jitter * image_w;
            dh = self.jitter * image_h;
            scale = np.random.uniform(0.25, 2);
        else:
            dw = 0
            dh = 0
            scale = 1

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));

        if (new_ar < 1):
            new_h = int(scale * net_h);
            new_w = int(net_h * new_ar);
        else:
            new_w = int(scale * net_w);
            new_h = int(net_w / new_ar);

        dx = int(np.random.uniform(0, net_w - new_w));
        dy = int(np.random.uniform(0, net_h - new_h));
        flip = np.random.randint(2)
        axis = np.random.randint(2)
        angle = np.random.randint(4)

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        # randomly distort hsv space
        #im_sized = random_distort_image(im_sized)

        # randomly flip

        im_sized = random_flip(im_sized, flip ,axis)

        #im_sized= random_rotate90(im_sized,angle)
        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, axis, angle, image_w, image_h)
        #self.showAugmentedImg(im_sized.copy(),all_objs)
        del flip
        del dx
        del dy
        del new_ar
        del scale
        del image
        del axis
        del angle
        return im_sized, all_objs

    def showAugmentedImg(self,image,boundingBoxes):
        for bbox in boundingBoxes:
            image = cv2.rectangle(image, (bbox["xmin"], bbox["ymin"]), (bbox["xmax"], bbox["ymax"]), (255, 0, 0), 2)
            try:
                cv2.putText(image, bbox["name"], (bbox["xmin"], bbox["ymin"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 0, 0), 2)
            except KeyError:
                print(bbox)
        cv2.imshow("augmented image", image)
        cv2.waitKey(0)

    def on_epoch_end(self):
        self.batch_num = 0
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []
        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        try:
            int(i)
            instance = self.instances[i]
        except:
            instance = i
        image_name = instance['filename']
        image_name = image_name.replace("b", "", 1)
        image = cv2.imread(image_name)
        #image = image[:, :, ::-1]
        return image