import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
import pydiffvg
import matplotlib.pyplot as plt
import time
from torchvision.transforms.functional import to_tensor
import matplotlib.cm as cm
import random

from dr import calculate_poses_batch

from io import BytesIO

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.cm as cm
# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                # if len(id_list) == 10: #TODO change this
                #     break
        id_list = np.array(id_list) 

        new_name_list = []
        length_list = []

        fullpath = opt.motion_dir.rsplit('/', 1)
        joints_dir = pjoin(fullpath[0],'new_joints')
        # conditions_dir = pjoin(fullpath[0],'conditions')
        print("id_list", id_list.shape[0])
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                joints = np.load(pjoin(joints_dir, name + '.npy'))

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                    'joints': joints,
                                                    # 'condition': img_data,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'joints': joints,
                                    #    'condition': img_data,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
            
        print('len(new_name_list):', len(new_name_list))
        print('len(length_list):', len(length_list))

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, joints = data['motion'], data['length'], data['text'], data['joints']
        # Randomly select a caption
        text_data = random.choice(text_list)
        # og_caption = text_data['caption']

        caption, tokens = text_data['caption'], text_data['tokens']
        
        # if random.random() > 0.5:
        #     caption = ''
        #     tokens = []
            
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        joints = joints[idx:idx+m_length]
        # motion_length = len(motion)
        # loaded_img_condition = torch.from_numpy(plot_3d_motion(joints))
        loaded_img_condition,indicies = Get_frames(joints)
        # inx = [fi,si,ti]
        # loaded_img_condition = torch.stack(inx)
        
        

        
        # img_condition = torch.rand(3,480,480)
        "Z Normalization"
        motion = (motion - self.mean) / self.std



        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        # go to train_loop and extract img_cond
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), loaded_img_condition, indicies



def generate_pose_img(jointsl):
    
    canvas_width = 480
    canvas_height = 480
    shapes = []
    path_groups = []
    ids = []
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    # Create the plot
    # fig, ax = plt.subplots(figsize=(4.8, 4.8))
    i=0
    
    # joints[:,0] = -(joints[:,0] - joints[:,0].mean())
    # joints[:,1] = -(joints[:,1] - 2.2)
    
    # maxX = np.max(np.abs(joints[:,0]))
    # minX = -np.max(np.abs(joints[:,0]))
    
    # maxAxis = 1.2
    
    # xFactor = (maxX / maxAxis) * 480 
    
    # xAdditive = -(xFactor - 480) /2
    
    # joints[:,0] = ((joints[:,0] - minX) / (maxX - minX)* xFactor + xAdditive)
    # joints[:,1] = ((joints[:,1]) / (2.2) * 440 + 20)
    
    
    
    joints = torch.randn(jointsl.shape)
    joints[:, 0] = -(joints[:, 0] - joints[:, 0].mean())
    joints[:, 1] = -(joints[:, 1] - 2.2)
    maxX = torch.max(torch.abs(joints[:, 0]))
    minX = -maxX
    maxAxis = 1.2
    xFactor = (maxX / maxAxis) * 480
    xAdditive = -(xFactor - 480) / 2
    joints[:, 0] = ((joints[:, 0] - minX) / (maxX - minX) * xFactor + xAdditive)
    joints[:, 1] = (joints[:, 1] / 2.2 * 440 + 20)
    
    
    

    for parent_idx in range(len(kinematic_tree)):
        # shapes = []
        ids=[]
        target_color = torch.tensor([0.1, 0.1, 0.8, 0.5])
        if parent_idx == 1 or parent_idx == 4:
            target_color = torch.tensor([0.8, 0.1, 0.1, 0.4])
        for window_start in range(len(kinematic_tree[parent_idx]) - 1):
            window_end = window_start + 1
            start_joint = joints[kinematic_tree[parent_idx][window_start]]
            end_joint = joints[kinematic_tree[parent_idx][window_end]]
                
            points = torch.tensor([[start_joint[0] ,  start_joint[1] ], # base
                #    [150.0,  60.0], # control point
                #    [ 90.0, 198.0], # control point
                    [ end_joint[0] , end_joint[1] ]])
            
            num_control_points = torch.tensor([0])
            path = pydiffvg.Path(num_control_points = num_control_points,
                    points = points,
                    is_closed = False,
                    stroke_width = torch.tensor(4.0))
            shapes.append(path)
            ids.append(i)
            i = i + 1
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor(ids),
                                fill_color = None,
                                stroke_color = target_color)
        path_groups.append(path_group)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, path_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(480, # width
                480, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None,
                *scene_args)
    
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    img = img[:, :, :3] / 255.0  # Remove the alpha channel and normalize
    # pydiffvg.imwrite(img.cpu(), 'results/single_circle/target2.png', gamma=2.2)
    
    # print(type(img))
    # print(img)
    # target = img.clone()
    return img



def visualize_pose(joints, filename='./conditions/pose.jpg'):
    """
    This function visualizes a 22-joint pose with labeled lines between connected joints.

    Args:
        joints: A numpy array of shape (22, 3) representing joint positions (x, y, z).
        kinematic_tree: A list of lists representing the parent-child relationships between joints.
        filename: The filename to save the image (default: './conditions/pose.jpg').
    """
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    # Create the plot
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    for parent_idx in range(len(kinematic_tree)):
        for window_start in range(len(kinematic_tree[parent_idx]) - 1):
            window_end = window_start + 1
            start_joint = joints[kinematic_tree[parent_idx][window_start]]
            end_joint = joints[kinematic_tree[parent_idx][window_end]]
            color = 'k-'
            if parent_idx == 1 or parent_idx == 4:
                color = 'r-'
            ax.plot([start_joint[0], end_joint[0]], [start_joint[1], end_joint[1]], color, alpha=0.7)

    # Set axis limits slightly bigger than joint range for better visualization
    plt.xlim([joints[:, 0].min()-0.2, joints[:, 0].max()+0.2])
    plt.ylim([joints[:, 1].min()-0.05, joints[:, 1].max()+0.05])
    
    ax.set_axis_off()
    ax.set_facecolor('white')

    # Convert plot to tensor
    fig.canvas.draw()
    image_np = np.array(fig.canvas.renderer.buffer_rgba())
    image_np = image_np[:, :, :3] / 255.0
    # image_np = image_np / 255.0  # Scale to range [0, 1]
    image_tensor = to_tensor(image_np).float() 
    # Convert to tensor (3, 480, 480)

    # # Save the image tensor
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=90)  # Save the image as PNG

    plt.close(fig)  # Close the figure to free memory

    return image_tensor


def visualize_pose(joints, kinematic_tree, filename='./conditions/pose.jpg'):
    """
    This function visualizes a 22-joint pose with labeled lines between connected joints.

    Args:
        joints: A numpy array of shape (22, 3) representing joint positions (x, y, z).
        kinematic_tree: A list of lists representing the parent-child relationships between joints.
        filename: The filename to save the image (default: './conditions/pose.jpg').
    """
    # Convert joints to a PyTorch tensor
    joints = torch.tensor(joints, dtype=torch.float32)

    # Create the plot
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    for parent_idx in range(len(kinematic_tree)):
        for window_start in range(len(kinematic_tree[parent_idx]) - 1):
            window_end = window_start + 1
            start_joint = joints[kinematic_tree[parent_idx][window_start]]
            end_joint = joints[kinematic_tree[parent_idx][window_end]]
            color = 'k-'
            if parent_idx == 1 or parent_idx == 4:
                color = 'r-'
            ax.plot([start_joint[0].item(), end_joint[0].item()], [start_joint[1].item(), end_joint[1].item()], color, alpha=0.7)

    # Set axis limits slightly bigger than joint range for better visualization
    plt.xlim([joints[:, 0].min().item()-0.2, joints[:, 0].max().item()+0.2])
    plt.ylim([joints[:, 1].min().item()-0.05, joints[:, 1].max().item()+0.05])
    
    ax.set_axis_off()
    ax.set_facecolor('white')

    # Convert plot to tensor
    fig.canvas.draw()
    image_np = np.array(fig.canvas.renderer.buffer_rgba())
    image_np = image_np[:, :, :3] / 255.0
    image_tensor = to_tensor(image_np).float()
    
    plt.close(fig)  # Close the figure to free memory

    return image_tensor





def Get_frames(data):
    
    number_of_frames = random.choice([2,3,4,5])
    number_of_frames = 3
    
    conditions = []
    indicies = []
    for i in range(0,number_of_frames):
        KeyFrame = (1/number_of_frames) * i  + (1/(2*number_of_frames)) 
        # + (random.random() - 0.5) * 0.18 # TODO: account for this in forward method with some encoding, removed perturbations for now
        frame = int(len(data)*KeyFrame)
        indicies.append(frame)
    
    ten = data[indicies]
    ten = ten[..., :2]
    ten = torch.tensor(ten)
    
    # print(ten.shape)
    
    img = calculate_poses_batch(ten)
    img = img.permute(0,2,3,1)
    # conditions.append(img)
    # indicies.append(frame)
        
    return img, indicies
    return torch.stack(conditions), indicies
    # firstKeyFrame = 0.25 + (random.random() - 0.5) * 0.2
    # seconKeyFrame = 0.5 + (random.random() - 0.5) * 0.2
    # thirdKeyFrame = 0.75 + (random.random() - 0.5) * 0.2
    
    # first = int(len(data)*firstKeyFrame)
    # second = int(len(data)*seconKeyFrame)
    # third = int(len(data)*thirdKeyFrame)
    
    # # fi = visualize_pose(data[first])
    # # si = visualize_pose(data[second])
    # # ti = visualize_pose(data[third])
    
    # fi = generate_pose_img(data[first])
    # si = generate_pose_img(data[second])
    # ti = generate_pose_img(data[third])
    
    
    # firstIndex = int(firstKeyFrame * 196)
    # secondIndex = int(seconKeyFrame * 196)
    # thirdIndex = int(thirdKeyFrame * 196)
    
    # if random.random() > 0.5:
    #     randomChoice = random.choice([0,1,2])
    #     if randomChoice == 0:
    #         firstIndex = 0
    #     elif randomChoice == 1:
    #         secondIndex = 0
    #     else:
    #         thirdIndex = 0
    
    # firstweight = create_quadratic_pattern(196,firstIndex)
    # secondweight = create_quadratic_pattern(196,secondIndex)
    # thirdweight = create_quadratic_pattern(196,thirdIndex)
    
    # return torch.stack([fi,si,ti]), torch.stack([firstIndex,secondIndex,thirdIndex])




###start of plot_3d_motion

def create_quadratic_pattern(length, index):
    if index < 0 or index >= length:
        raise ValueError("Index must be within the array bounds")
    
    pattern = torch.zeros(length)
    # max_distance = max(index, length - index - 1)  # maximum distance to the edges
    
    
    for i in range(length):
        if i >=index:
            max_distance = length-index-1
        else:
            max_distance = index
        
        if index == -1:
            pattern[i] = 0
        distance = abs(i - index)
        pattern[i] = 1 - (distance / max_distance)**2
    
    pattern = torch.clamp(pattern, 0, 1)  # Ensure values are within [0, 1]
    return pattern  # Return as a PyTorch tensor

def plot_3d_motion(joints,title = '', radius=4, plotName = 'newplot.png'):
#     matplotlib.use('Agg')

    max_len = joints.shape[0]
    target_frame = int(max_len//2)
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    width_in_pixels = 480
    height_in_pixels = 480

    # Convert pixels to inches (1 inch = 2.54 cm)
    width_in_inches = width_in_pixels / plt.rcParams['figure.dpi']
    height_in_inches = height_in_pixels / plt.rcParams['figure.dpi']

    # Create the plot with the specified size
    # fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))

    fig = plt.figure(figsize=(width_in_inches, height_in_inches))

    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['green', 'pink', 'blue', 'yellow', 'red',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    


    # Set a seed value (e.g., 1234 for reproducibility)
    random.seed(232113)

    num_colors = len(colors)
    colors = [(random.random(), random.random(), random.random()) for _ in range(num_colors+10)]


    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index , trajectory_only = 'False'):
        #         print(index)
        # ax.lines.clear()
        ax.collections.clear()
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =

        # plot_xzPlane(MINS[0]
        #             #  -trajec[index, 0]
        #              , MAXS[0]
        #             #  -trajec[index, 0]
        #              ,0
        #              ,MINS[2]
        #             #    -trajec[index, 1]
        #                  ,MAXS[2]
        #                 #  -trajec[index, 1]
        #                  )


#         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        trajec_color = 'blue'
        if trajectory_only == 'True':
            trajec_color = 'red'

        if index > 1:
            ax.plot3D(trajec[:index, 0]
                    #   -trajec[index, 0] ## these center the trajectory
                      , np.zeros_like(trajec[:index, 0]),
                        trajec[:index, 1]
                        # -trajec[index, 1] ## these center the trajectory
                        , linewidth=1.0,
                      color=trajec_color
                      )
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        

        # cmap = cm.get_cmap('tab20')  # Choose a colormap
        # num_colors = 22  # Assuming kinematic_tree has a defined length

        cmap = cm.get_cmap('gist_ncar')
        num_colors = 26
        colors = cmap(np.linspace(0, 1, num_colors))
        if (trajectory_only == 'False'):
            
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
    #             print(color)
                # print(chain)
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0

                for j in range(len(chain) - 1):
                    start_idx = chain[j]
                    
                    norm = start_idx / (num_colors - 1)
                    color = cmap(norm)

                    end_idx = chain[j + 1]
                    # color = colors[start_idx]

                    ax.plot3D(
                        [data[index, start_idx, 0], data[index, end_idx, 0]],
                        [data[index, start_idx, 1], data[index, end_idx, 1]],
                        [data[index, start_idx, 2], data[index, end_idx, 2]],
                        linewidth=linewidth,
                        color=colors[start_idx]
            )

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_facecolor('black')
        
        # Set the color of the axes labels and ticks to white
        ax.w_xaxis.set_pane_color((0, 0, 0, 1))
        ax.w_yaxis.set_pane_color((0, 0, 0, 1))
        ax.w_zaxis.set_pane_color((0, 0, 0, 1))
        
        ax.xaxis._axinfo['grid'].update(color = 'w', linewidth = 0.5)
        ax.yaxis._axinfo['grid'].update(color = 'w', linewidth = 0.5)
        ax.zaxis._axinfo['grid'].update(color = 'w', linewidth = 0.5)
        
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        fig.patch.set_facecolor('black')

    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    # ani.save(save_path, fps=fps)
    update(max_len,'True')
    update(target_frame)
    
    file_path = './conditions/' + plotName

    buf = BytesIO()
    # plt.savefig(file_path)
    plt.savefig(buf, format='png')
    plt.close() # Prevents figure from being displayed
    buf.seek(0)
    # image = Image.open(buf)
    # Close the buffer
    image = plt.imread(buf)
    if image.shape[2] == 4:  # Check if the image has 4 channels
            image = image[..., :3] 
    image = np.transpose(image, (2, 0, 1))  # Change shape from (480, 480, 3) to (3, 480, 480)
    image = image / 255.0
    # image = image[np.newaxis, :]  
    # Close the buffer
    buf.close()
    # image.show()
    # buf.close()
    # plt.show()
    # plt.close()
    

    return image


###end of plot_3d_motion




'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len

class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        opt.text_dir = opt.text_dir.replace('././','./')
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(opt.text_dir+'/'+ name + '.txt') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None
        # fixed_length can be set from outside before sampling

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.mode = mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)
        print(pjoin(opt.data_root, 'Mean.npy'))
        print(mode)
        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)
