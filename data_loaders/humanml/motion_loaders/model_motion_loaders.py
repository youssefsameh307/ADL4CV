from torch.utils.data import DataLoader, Dataset
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate
import pickle
from torchvision.transforms.functional import to_tensor
import random

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

import matplotlib.pyplot as plt

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
    try:
        plt.xlim([joints[:, 0].min()-0.2, joints[:, 0].max()+0.2])
        plt.ylim([joints[:, 1].min()-0.05, joints[:, 1].max()+0.05])
    except:
        print("shit happened")
    
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

def Get_frames(data):
    firstKeyFrame = 0.25 + (random.random() - 0.5) * 0.2
    seconKeyFrame = 0.5 + (random.random() - 0.5) * 0.2
    thirdKeyFrame = 0.75 + (random.random() - 0.5) * 0.2
    
    first = int(len(data)*firstKeyFrame)
    second = int(len(data)*seconKeyFrame)
    third = int(len(data)*thirdKeyFrame)
    
    # fi = visualize_pose(data[first])
    si = visualize_pose(data[second])
    # ti = visualize_pose(data[third])
    
    firstIndex = int(firstKeyFrame * 196)
    secondIndex = int(seconKeyFrame * 196)
    thirdIndex = int(thirdKeyFrame * 196)
    
    if random.random() > 0.5:
        randomChoice = random.choice([0,1,2])
        if randomChoice == 0:
            firstIndex = 0
        elif randomChoice == 1:
            secondIndex = 0
        else:
            thirdIndex = 0
    
    # firstweight = create_quadratic_pattern(196,firstIndex)
    # secondweight = create_quadratic_pattern(196,secondIndex)
    # thirdweight = create_quadratic_pattern(196,thirdIndex)
    return si
    # return torch.stack([fi,si,ti])




class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        joints = data['joints']
        m_lens = []
        motions = []
        img_condition = Get_frames(joints)
        for idx,mm_motion in enumerate(mm_motions):
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            # We don't need the following logic because our sample func generates the full tensor anyway:
            # if len(motion) < self.opt.max_motion_length:
            #     motion = np.concatenate([motion,
            #                              np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            #                              ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
            
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens, img_condition



def get_motion_loader(opt_path, batch_size, ground_truth_dataset, mm_num_samples, mm_num_repeats, device):
    opt = get_opt(opt_path, device)

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    if 'v6' in opt.name:
        dataset = CompV6GeneratedDataset(opt, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    else:
        raise KeyError('Dataset not recognized!!')

    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=0)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1,collate_fn=collate_fn, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader

# our loader
def get_mdm_loader(model, diffusion, batch_size, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # dataset = CompMDMGeneratedDataset(opt, ground_truth_dataset, ground_truth_dataset.w_vectorizer, mm_num_samples, mm_num_repeats)
    dataset = CompMDMGeneratedDataset(model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale)

    try:
        file_path = 'dataset.pkl'

        # Open the file in binary write mode and dump the dataset
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
    except:
        print('Error saving dataset')

    mm_dataset = MMGeneratedDataset(opt, dataset, ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=0)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader