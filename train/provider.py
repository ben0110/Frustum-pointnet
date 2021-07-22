''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

#import cPickle as pickle
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'models'))

from box_util import box3d_iou
from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

from dataset import KittiDataset
from collections import Counter
import kitti_utils

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - \
        (class_id * angle_per_class + angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle
        
def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.
 
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box2d(pc,pixels, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    if box2d[0] > box2d[2]:
        a = box2d[0]
        box2d[0] = box2d[2]
        box2d[2] = a
    assert box2d[1]< box2d[3]
    img_width = 1280.0
    img_height = 720.0
    # assert x1<1280.0 and x2 < 1280.0 and y1 < 720.0 and y2 < 720.0
    if box2d[0] > img_width:
        box2d[0] = img_width
    elif box2d[0] < 0.0:
        box2d[0] = 0.0

    if box2d[2] > img_width:
        box2d[2] = img_width
    elif box2d[2]< 0.0:
        box2d[2] = 0.0
    if box2d[1] > img_height:
        box2d[1] = img_height
    elif box2d[1] < 0.0:
        box2d[1] = 0.0
    if box2d[3] > img_height:
        box2d[3] = img_height
    elif box2d[3] < 0.0:
        box2d[3] = 0.0

    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]]
    box2d_corners[1,:] = [box2d[2],box2d[1]]
    box2d_corners[2,:] = [box2d[2],box2d[3]]
    box2d_corners[3,:] = [box2d[0],box2d[3]]
    print("box2d_corners",box2d_corners)
    box2d_roi_inds = in_hull(pixels, box2d_corners)
    #box2D_mask= np.zeros((pc.shape[0]),dtype=np.float32)
    #box2D_mask[box2d_roi_inds]=1
    return pc[box2d_roi_inds], box2d_roi_inds
def get_pixels(index,split):
    if split=="val" or split=="train":
        pixel_dir = "/root/frustum-pointnets_RSC/dataset/KITTI/object/training/pc_to_pixels/"
    else:
        pixel_dir = "/root/frustum-pointnets_RSC/dataset/KITTI_2/object/testing/pc_to_pixels/"
    pixel_file = os.path.join(pixel_dir, '%06d.txt' % index)
    print(pixel_file)
    assert os.path.exists(pixel_file)
    pixels = np.loadtxt(pixel_file,delimiter=",")
    return pixels
def random_shift_box2d(box2d, shift_ratio=0.2):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
def get_closest_pc_to_center(pc,pixels,center_box2d):
    idx = np.argmin(np.linalg.norm(pixels-center_box2d,axis=1))
    center3d = pc[idx,:]
    return center3d
def get_2Dboxes_detected(idx,res,split):
    if split=="val":
        det_2dboxes_path = "/root/frustum-pointnets_RSC_2D/dataset/RSC/labelsVal2D/"+res+"/"
    else:
        det_2dboxes_path = "/root/frustum-pointnets_RSC_2D/dataset/RSC/labelsTest2D/" + res + "/"
    det_2dboxes_file = det_2dboxes_path + "%06d.txt" %idx
    if not os.path.exists(det_2dboxes_file):
        return None
    else:
        with open(det_2dboxes_file, 'r') as f:
            labels = []
            for line in f:
                label = line.strip().split(' ')
                label_=[]
                for k in range(len(label)):
                    print(label[k])
                    label_.append(int(label[k]))
                labels.append(label_)
    return labels

def load_GT_eval(indice,database,split):
    ''' Load GT value for the respective "split" dataset for evaluation purposes
    '''
    data_val=KittiDataset( root_dir='/root/frustum-pointnets_RSC/dataset/', dataset=database, mode='TRAIN', split=split)
    id_list = data_val.sample_id_list
    obj_frame=[]
    corners_frame=[]
    size_class_frame=[]
    size_residual_frame=[]
    angle_class_frame=[]
    angle_residual_frame=[]
    center_frame=[]
    id_list_new=[]
    for i in range(len(id_list)):
        if(id_list[i]<indice+1):
            gt_obj_list = data_val.filtrate_objects(
                data_val.get_label(id_list[i]))
            #print("GT objs per frame", id_list[i],len(gt_obj_list))
            gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
            obj_frame.append(gt_obj_list)
            corners_frame.append(gt_corners)
            angle_class_list=[]
            angle_residual_list=[]
            size_class_list=[]
            size_residual_list=[]
            center_list=[]
            for j in range(len(gt_obj_list)):

                angle_class, angle_residual = angle2class(gt_boxes3d[j][6],
                                                      NUM_HEADING_BIN)
                angle_class_list.append(angle_class)
                angle_residual_list.append(angle_residual)

                size_class, size_residual = size2class(np.array([gt_boxes3d[j][3], gt_boxes3d[j][4], gt_boxes3d[j][5]]),
                                                   "Pedestrian")
                size_class_list.append(size_class)
                size_residual_list.append(size_residual)

                center_list.append( (gt_corners[j][0, :] + gt_corners[j][6, :]) / 2.0)
            size_class_frame.append(size_class_list)
            size_residual_frame.append(size_residual_list)
            angle_class_frame.append(angle_class_list)
            angle_residual_frame.append(angle_residual_list)
            center_frame.append(center_list)
            id_list_new.append(id_list[i])

    return corners_frame,id_list_new

class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    def __init__(self, npoints, database,  split, res,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.dataset_kitti = KittiDataset(root_dir='/root/frustum-pointnets_RSC/dataset/',dataset=database, mode='TRAIN', split=split)
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.res_det = res
        self.one_hot = one_hot

        if overwritten_data_path is None:
            overwritten_data_path = os.path.join(ROOT_DIR,
                'kitti/frustum_carpedcyc_%s.pickle'%(split))

        self.from_rgb_detection = from_rgb_detection
        if from_rgb_detection:
            with open(overwritten_data_path,'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp) 
                self.prob_list = pickle.load(fp)
        elif(split=='train'):

            self.id_list = self.dataset_kitti.sample_id_list
            self.idx_batch = self.id_list
            batch_list = []
            self.frustum_angle_list=[]
            self.input_list=[]
            self.label_list=[]
            self.box3d_list = []
            self.box2d_list = []
            self.type_list = []
            self.heading_list=[]
            self.size_list = []

            perturb_box2d=True
            augmentX = 5
            for i in range(len(self.id_list)):
                #load pc
                print(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])
                #load_labels
                gt_obj_list_2D = self.dataset_kitti.get_label_2D(self.id_list[i])
                ps = pc_lidar
                #load pixels
                pixels = get_pixels(self.id_list[i],split)
                for j in range(len(gt_obj_list_2D)):
                    for _ in range(augmentX):
                        # Augment data by box2d perturbation
                        if perturb_box2d:
                            box2d = random_shift_box2d(gt_obj_list_2D[j].box2d)
                        frus_pc, frus_pc_ind = extract_pc_in_box2d(pc_lidar,pixels,box2d)
                        #get frus angle
                        center_box2d = np.array([(box2d[0]+box2d[2])/2.0, (box2d[1]+box2d[2])/2.0])
                        pc_center_frus = get_closest_pc_to_center(pc_lidar,pixels,center_box2d)
                        frustum_angle =  - np.arctan2(pc_center_frus[2],pc_center_frus[0])



                        #get label list
                        gt_obj_list=self.dataset_kitti.get_label(self.id_list[i])

                        cls_label = np.zeros((frus_pc.shape[0]), dtype=np.int32 )
                        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                        for k in range(gt_boxes3d.shape[0]):
                            box_corners = gt_corners[k]
                            fg_pt_flag = kitti_utils.in_hull(frus_pc[:, 0:3], box_corners)
                            cls_label[fg_pt_flag] = k+1
                        max = 0
                        corners_max = 0
                        for k in range(gt_boxes3d.shape[0]):
                            count = np.count_nonzero(cls_label == k + 1)
                            if count > max:
                                max = count
                                corners_max = k
                        seg = np.where(cls_label == corners_max + 1, 1.0, 0.0)

                        cls_label=seg
                        print("train", np.count_nonzero(cls_label==1))
                        if box2d[3] - box2d[1] < 25 or np.sum(cls_label) == 0:
                            continue
                        self.input_list.append(frus_pc)
                        self.frustum_angle_list.append(frustum_angle)
                        self.label_list.append(cls_label)
                        self.box3d_list.append(gt_corners[corners_max])
                        self.box2d_list.append(box2d)
                        self.type_list.append("Pedestrian")
                        self.heading_list.append(gt_obj_list[corners_max].ry)
                        self.size_list.append(np.array([gt_obj_list[corners_max].h, gt_obj_list[corners_max].w, gt_obj_list[corners_max].l]))
                        batch_list.append(self.id_list[i])
            self.id_list = batch_list
        elif(split=='val' or split=='test'):

            self.indice_box = []
            self.dataset_kitti.sample_id_list = self.dataset_kitti.sample_id_list
            self.id_list = self.dataset_kitti.sample_id_list
            self.idx_batch = self.id_list
            batch_list = []
            self.frustum_angle_list = []
            self.input_list = []
            self.label_list = []
            self.box3d_list = []
            self.box2d_list = []
            self.type_list = []
            self.heading_list = []
            self.size_list = []
            for i in range(len(self.id_list)):
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])
                gt_obj_list = self.dataset_kitti.get_label(self.id_list[i])
                print(self.id_list[i])
                #get val 2D boxes:
                box2ds = get_2Dboxes_detected(self.id_list[i],self.res_det,split)
                if box2ds == None:
                    continue
                print("number detection", len(box2ds))
                pixels = get_pixels(self.id_list[i],split)
                for j in range(len(box2ds)):
                    box2d = box2ds[j]

                    if (box2d[3] - box2d[1]) < 25 or ((box2d[3]>720 and box2d[1]>720)) or ((box2d[0]>1280 and box2d[2]>1280)) or ((box2d[3]<=0 and box2d[1]<=0)) or (box2d[0]<=0 and box2d[2]<=0) :
                        continue
                    print(box2d)
                    print("box_height", box2d[3] - box2d[1])
                    frus_pc, frus_pc_ind = extract_pc_in_box2d(pc_lidar, pixels, box2d)
                    # get frus angle
                    center_box2d = np.array([(box2d[0] + box2d[2]) / 2.0, (box2d[1] + box2d[2]) / 2.0])
                    pc_center_frus = get_closest_pc_to_center(pc_lidar, pixels, center_box2d)
                    frustum_angle = -1 * np.arctan2(pc_center_frus[2], pc_center_frus[0])

                    if len(frus_pc) < 20:
                        continue

                    # get_labels
                    gt_obj_list = self.dataset_kitti.filtrate_objects(self.dataset_kitti.get_label(self.id_list[i]))
                    gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                    # gt_boxes3d = gt_boxes3d[self.box_present[index] - 1].reshape(-1, 7)

                    cls_label = np.zeros((frus_pc.shape[0]), dtype=np.int32)
                    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                    for k in range(gt_boxes3d.shape[0]):
                        box_corners = gt_corners[k]
                        fg_pt_flag = kitti_utils.in_hull(frus_pc[:, 0:3], box_corners)
                        cls_label[fg_pt_flag] = k + 1
                    if (np.count_nonzero(cls_label > 0) < 20):
                        center = np.ones((3))*(-10.0)
                        heading = 0.0
                        size = np.ones((3))
                        cls_label[cls_label > 0] = 0
                        seg=cls_label
                        rot_angle = 0.0
                        box3d_center = np.ones((3))*(-1.0)
                        box3d = np.array([[box3d_center[0],box3d_center[1],box3d_center[2],size[0],size[1],size[2],rot_angle]])
                        corners_empty =  kitti_utils.boxes3d_to_corners3d(box3d, transform=False)
                        bb_corners = corners_empty[0]
                        self.indice_box.append(0)
                    else :
                        max = 0
                        corners_max = 0
                        for k in range(gt_boxes3d.shape[0]):
                            count = np.count_nonzero(cls_label == k + 1)
                            if count > max:
                                max = count
                                corners_max = k
                        seg = np.where(cls_label==corners_max+1,1,0)
                        self.indice_box.append(corners_max+1)
                        bb_corners = gt_corners[corners_max]
                        obj = gt_boxes3d[corners_max]
                        center = np.array([obj[0],obj[1],obj[2]])
                        size = np.array([obj[3],obj[4],obj[5]])
                        rot_angle = obj[6]
                    self.input_list.append(frus_pc)
                    self.frustum_angle_list.append(frustum_angle)
                    self.label_list.append(seg)
                    self.box3d_list.append(bb_corners)
                    self.box2d_list.append(box2d)
                    self.type_list.append("Pedestrian")
                    self.heading_list.append(rot_angle)
                    self.size_list.append(size)
                    batch_list.append(self.id_list[i])
            self.id_list = batch_list
            print("batch_list",batch_list)












    def __len__(self):
            return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]
        if self.from_rgb_detection:
            if self.one_hot:
                return point_set, rot_angle, self.prob_list[index], one_hot_vec
            else:
                return point_set, rot_angle, self.prob_list[index]
        
        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index] 
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)
        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
            self.type_list[index])
        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random()>0.5: # 50% chance flipping
                point_set[:,0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            point_set[:,2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
            NUM_HEADING_BIN)
        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual,\
                size_class, size_residual, rot_angle, one_hot_vec
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual,\
                size_class, size_residual, rot_angle

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), \
            self.get_center_view_rot_angle(index)).squeeze()
        
    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
            self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
            self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    h,w,l = box_size
    x_corners = [w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou_batch(logits,center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    pred_val = np.argmax(logits, 2)
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residuals[i,heading_class[i]] \
        for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residuals[i,size_class[i],:] \
        for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    box_pred_nbr=0
    for i in range(batch_size):
        # if object has low seg mask break
        if (np.sum(pred_val[i]) < 50):

            continue
        else:
            heading_angle = class2angle(heading_class[i],
                                        heading_residual[i], NUM_HEADING_BIN)
            box_size = class2size(size_class[i], size_residual[i])
            corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

            heading_angle_label = class2angle(heading_class_label[i],
                                              heading_residual_label[i], NUM_HEADING_BIN)
            box_size_label = class2size(size_class_label[i], size_residual_label[i])
            if (center_label[i][2] < 0.0):
                iou3d_list.append(0.0)
                iou2d_list.append(0.0)
            else:
                corners_3d_label = get_3d_box(box_size_label,
                                              heading_angle_label, center_label[i])

                iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
                iou3d_list.append(iou_3d)
                iou2d_list.append(iou_2d)
            box_pred_nbr = box_pred_nbr + 1.0

    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32), np.array(box_pred_nbr, dtype=np.float32)




def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residuals[i,heading_class[i]] \
        for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residuals[i,size_class[i],:] \
        for i in range(batch_size)])

    iou2d_list = [] 
    iou3d_list = [] 
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
            heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
            heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
            heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label) 
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res,\
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    #ty += h/2.0
    return h,w,l,tx,ty,tz,ry

if __name__=='__main__':
    import mayavi.mlab as mlab 
    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d
    median_list = []
    dataset = FrustumDataset(1024,database='KITTI_2', split='test',res="224",
        rotate_to_center=False, random_flip=False, random_shift=False)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
            'angle_class: ', data[3], 'angle_res:', data[4], \
            'size_class: ', data[5], 'size_residual:', data[6], \
            'real_size:', g_type_mean_size[g_class2type[data[5]]]+data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:,0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5],data[6]), class2angle(data[3], data[4],12), data[2])

        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:,0], ps[:,1], ps[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1,0,0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))
