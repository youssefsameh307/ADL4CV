import torch
import torchvision.transforms.functional as F
from multiprocessing import Pool
import os
from torchvision.utils import save_image
from PIL import Image

import numpy as np
import time

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from multiprocessing import Pool

def interpolate_joints(joints, num_points=500):
    """
    Interpolate the joint coordinates using linear interpolation in a differentiable manner.

    Args:
        joints: A torch tensor of shape (22, 2) representing joint positions (x, y).
        num_points: The number of points to interpolate.

    Returns:
        Interpolated joint coordinates as a torch tensor of shape (num_points, 2).
    """
    x_coords = torch.linspace(0, 1, joints.shape[0], device=joints.device)
    x_interp = torch.linspace(0, 1, num_points, device=joints.device)

    y_coords = joints[:, 1]
    y_interp = F.interpolate(y_coords.unsqueeze(0).unsqueeze(0), size=num_points, mode='linear', align_corners=False).squeeze()

    interpolated_joints = torch.stack((x_interp, y_interp), dim=-1)
    return interpolated_joints


def apply_averaging(images, kernel_size=3):
    """
    Apply averaging (blurring) to each image in the batch.

    Args:
        images: A torch tensor of shape (batch_size, channels, height, width) representing the batch of images.
        kernel_size: The size of the averaging kernel. Default is 3.

    Returns:
        A torch tensor of the same shape as images, with the averaging applied.
    """
    # Create an averaging kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=images.device) / (kernel_size * kernel_size)
    
    # Expand the kernel to have the same number of channels as the input images
    kernel = kernel.expand(images.size(1), 1, kernel_size, kernel_size)
    
    # Apply the convolution
    blurred_images = F.conv2d(images, kernel, padding=kernel_size//2, groups=images.size(1))
    
    return blurred_images


def draw_line2(image, start, end, color):
    """
    Draw a line on the image from start to end with the given color.

    Args:
        image: A torch tensor of shape (batch_size, 3, height, width) representing the batch of images.
        start: A torch tensor of shape (batch_size, 2) representing the starting points (x, y) for each image in the batch.
        end: A torch tensor of shape (batch_size, 2) representing the ending points (x, y) for each image in the batch.
        color: A tuple (r, g, b) representing the color to draw the line with.

    Returns:
        None (modifies the input image tensor in-place).
    """
    batch_size = image.shape[0]
    height = image.shape[2]
    width = image.shape[3]

    # Convert start and end points to integer indices
    start_indices = torch.stack([
        (start[:, 0] * (width - 1)).long(),
        (start[:, 1] * (height - 1)).long()
    ], dim=-1)

    end_indices = torch.stack([
        (end[:, 0] * (width - 1)).long(),
        (end[:, 1] * (height - 1)).long()
    ], dim=-1)

    # Compute line coordinates using interpolation
    num_points = torch.max(torch.abs(end_indices - start_indices)) + 1
    t = torch.linspace(0, 1, num_points, device=image.device).unsqueeze(1)
    line_indices = start_indices.unsqueeze(1) * (1 - t) + end_indices.unsqueeze(1) * t
    line_indices = line_indices.round().long()

    # Ensure indices are within bounds
    line_indices[:, :, 0] = torch.clamp(line_indices[:, :, 0], 0, width - 1)
    line_indices[:, :, 1] = torch.clamp(line_indices[:, :, 1], 0, height - 1)

    # Set color at line coordinates
    color_tensor = torch.tensor(color, device=image.device, dtype=image.dtype).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    # image[:, :, line_indices[:, :, 1], line_indices[:, :, 0]] = color_tensor
    
    for i in range(batch_size):
        image[i, :, line_indices[i, :, 1], line_indices[i, :, 0]] = color_tensor.reshape(-1,1)
    # image=apply_averaging(image, kernel_size=3)



def draw_line(image, start, end, color):
    """
    Draw a line on the image from start to end with the given color.

    Args:
        image: A torch tensor of shape (3, height, width) representing the image.
        start: A tuple (x, y) representing the starting point.
        end: A tuple (x, y) representing the ending point.
        color: A tuple (r, g, b) representing the color to draw the line with.
    """
    start = (int(start[0] * (image.shape[2] - 1)), int(start[1] * (image.shape[1] - 1)))
    end = (int(end[0] * (image.shape[2] - 1)), int(end[1] * (image.shape[1] - 1)))

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        image[:, y0, x0] = torch.tensor(color, device=image.device)
        if (x0 == x1) and (y0 == y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
            
def create_pose_image(joints, kinematic_tree, image_size=480):
    """
    Create an image representing the pose by drawing lines between joints.

    Args:
        joints: A torch tensor of shape (22, 2) representing joint positions (x, y).
        kinematic_tree: A list of lists representing the parent-child relationships between joints.
        image_size: The size of the image (default: 480).

    Returns:
        A torch tensor representing the image of the pose.
    """
    # Initialize an image with white background (3 channels for RGB)
    image = torch.ones(3, image_size, image_size, device=joints.device)

    # Draw lines between joints based on the kinematic tree
    for parent_idx, segment in enumerate(kinematic_tree):
        for i in range(len(segment) - 1):
            start_joint = joints[segment[i]]
            end_joint = joints[segment[i + 1]]
            target_color = (0.1, 0.1, 0.8)
            if parent_idx == 1 or parent_idx == 4:
                target_color = (0.8, 0.1, 0.1)
            draw_line(image, start_joint, end_joint, color=target_color)  # Draw lines in black

    return image


def create_pose_image2(joints, kinematic_tree, image_size=480):
    """
    Create an image representing the pose by drawing lines between joints.

    Args:
        joints: A torch tensor of shape (22, 2) representing joint positions (x, y).
        kinematic_tree: A list of lists representing the parent-child relationships between joints.
        image_size: The size of the image (default: 480).

    Returns:
        A torch tensor representing the image of the pose.
    """
    # Initialize an image with white background (3 channels for RGB)
    image_b = torch.ones(joints.shape[0],3, image_size, image_size, device=joints.device)

    # Draw lines between joints based on the kinematic tree
    for parent_idx, segment in enumerate(kinematic_tree):
        for i in range(len(segment) - 1):
            start_joint = joints[:,segment[i]]
            end_joint = joints[:,segment[i + 1]]
            target_color = (0.1, 0.1, 0.8)
            if parent_idx == 1 or parent_idx == 4:
                target_color = (0.8, 0.1, 0.1)
            draw_line2(image_b, start_joint, end_joint, color=target_color)  # Draw lines in black

    return image_b



def calculate_pose_matrix(args):
    """
    Calculate the MSE for a single pair of joint positions.

    Args:
        args: Tuple containing (joints1, joints2, kinematic_tree, image_size).

    Returns:
        The mean squared error between the images of the poses.
    """
    joints1, kinematic_tree, image_size = args
    image1 = create_pose_image(joints1, kinematic_tree, image_size)

    return image1

def calculate_pose_matrix2(joints1, kinematic_tree):
    """
    Calculate the MSE for a single pair of joint positions.

    Args:
        args: Tuple containing (joints1, joints2, kinematic_tree, image_size).

    Returns:
        The mean squared error between the images of the poses.
    """
    image1 = create_pose_image2(joints1, kinematic_tree)

    return image1


def process_joints_batch(my_ten):
    my_ten[:,:,0] = -(my_ten[:,:,0] - my_ten[:,:,0].mean(1).unsqueeze(1).repeat(1,my_ten.shape[1]))
    my_ten[:,:,1] = -(my_ten[:,:,1] - 2.0)

    maxX = my_ten[:,:,0].max(1).values
    minX = -maxX

    maxAxis = 1.2
    xFactor = (maxX / maxAxis) * 480 
    xAdditive = -(xFactor - 480) /2

    maxX = maxX.unsqueeze(1).repeat(1,my_ten.shape[1])
    minX = minX.unsqueeze(1).repeat(1,my_ten.shape[1])
    xFactor = xFactor.unsqueeze(1).repeat(1,my_ten.shape[1]) /480
    xAdditive = xAdditive.unsqueeze(1).repeat(1,my_ten.shape[1]) /480

    my_ten[:,:,0] = ((my_ten[:,:,0] - minX) / (maxX - minX) * xFactor + xAdditive)
    my_ten[:,:,1] = ((my_ten[:,:,1]) / (2.0) * (440/480) + (20/480))
    return my_ten

def calculate_poses_batch(joints1_batch, image_size=480, workers=8):
    """
    Calculate the MSE between batches of joint positions in parallel.

    Args:
        joints1_batch: A torch tensor of shape (batch_size, 22, 2) representing the first batch of joint positions (x, y).
        joints2_batch: A torch tensor of shape (batch_size, 22, 2) representing the second batch of joint positions (x, y).
        kinematic_tree: A list of lists representing the parent-child relationships between joints.
        image_size: The size of the image (default: 480).
        workers: Number of parallel workers (default: 8).

    Returns:
        The mean squared error between the images of the poses for each pair in the batches.
    """
    
    # joints1_batch = joints1_batch.cpu()
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    joints1_batch = process_joints_batch(joints1_batch)


    batch_size = joints1_batch.shape[0]
    args = [(joints1_batch[i], kinematic_tree, image_size) for i in range(batch_size)]

    # with Pool(workers) as pool:
    #     images = pool.map(calculate_pose_matrix, args)
    images = calculate_pose_matrix2(joints1_batch,kinematic_tree)
    # print(images.shape)
    return images
    # return torch.stack(images)

def save_batch_images(images, directory='batch_img', prefix='image'):
    """
    Save a batch of tensor images as PNG files.

    Args:
        images: A torch tensor of shape (batch_size, 3, height, width) representing a batch of RGB images.
        directory: The directory where the images will be saved (default: './images').
        prefix: The prefix for the image filenames (default: 'image').

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    batch_size = images.shape[0]

    for i in range(batch_size):
        image_tensor = images[i]
        image_array = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and move to CPU
        image_array = (image_array * 255).astype('uint8')  # Convert to uint8 format

        image = Image.fromarray(image_array)
        image.save(os.path.join(directory, f'{prefix}_{i + 1}.png'))

# # Example usage
# kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# batch_size = 64*9
# joints1_batch = torch.rand(batch_size, 22, 2)
# joints2_batch = torch.rand(batch_size, 22, 2)

# joints = np.load("dataset/HumanML3D/new_joints/000003.npy")

# joints = joints[:8]

# my_ten = torch.from_numpy(joints)[:,:,:2]
# # my_ten = my_ten.cuda()
# # my_ten = my_ten.cpu()

# # my_ten = my_ten.to(torch.device('cpu'))

# joints1_batch = joints1_batch.cuda()

# # start_time = time.time()

# images = calculate_poses_batch(joints1_batch)

# # end_time = time.time()
# # print(images)

# # elapsed_time = end_time - start_time

# # # Print the elapsed time in seconds
# # print(f"Elapsed time: {elapsed_time:.6f} seconds")


# save_batch_images(images[:2])
# # print(images.shape)
# # print(f'MSE between poses: {mse_loss}')
