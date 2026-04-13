import torch
from torchvision.transforms import v2
import math
import numpy as np
from functools import partial
def create_visual_transforms(img_size = 224):
    train_transform = v2.Compose([
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.75, 1.0),
            ratio=(7.0 / 8.0, 8.0 / 7.0)
        ),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        # edge padding before + centercrop after rotation => corners filled.
        v2.Pad(padding_mode="edge", padding=math.ceil(img_size * 0.2)),
        v2.RandomRotation(
            degrees=15,
            interpolation=v2.InterpolationMode.BILINEAR
        ),
        v2.CenterCrop(img_size),

        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0, sigma=0.08),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])

    nontrain_transform = v2.Compose([
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, nontrain_transform


def create_mp_transform():
     return v2.Resize(size=(224,224))
 #yup, just image. MP's HandLandmarker will handle the rest of the processing internally.
#==============================
#=== KEYPOINT AUGMENTATIONS ===
#==============================

def to_points(x):
    new_shape = (*x.shape[:-1], 21, 2)
    return x.reshape(new_shape) if x.dim() > 1 else x.reshape(21, 2)

def to_flat(points):
    new_shape = (*points.shape[:-2], 42)
    return points.reshape(new_shape)

def rotate(x, angle_rad, in_place=False):

    if in_place:
        pts=to_points(x)  # directly modify input tensor without cloning
    else:
        pts = to_points(x).clone()
    center = pts[0]  # Wrist

    # Shift to origin
    pts = pts - center

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    
    # Create rotation matrix on the same device/dtype as x
    R = torch.tensor([[c, -s],
                      [s,  c]], device=x.device, dtype=x.dtype)

    # Matrix multiplication remains @
    pts = pts @ R.t()

    # Shift back
    pts = pts + center

    return to_flat(pts)

def rand_stretch_aligned(x, scale_x, scale_y, in_place=False):
    if in_place:
        pts=to_points(x)  # directly modify input tensor without cloning
    else:
        pts = to_points(x).clone()

    center = pts[0]  # Wrist
    
    # 1. Shift to origin
    pts = pts - center

    # 2. Find the hand's current angle (using middle finger MCP at index 9)
    direction_vector = pts[9]
    angle_rad = np.arctan2(direction_vector[1], direction_vector[0])

    # 3. Create Rotation Matrix
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = torch.tensor([[c, -s],
                      [s,  c]], device=x.device, dtype=x.dtype)
    
    # 4. Create Scaling Matrix
    S = torch.tensor([[scale_x, 0],
                      [0, scale_y]], device=x.device, dtype=x.dtype)

    # 5. Apply the "Sandwich"
    # Rotate to align with axis -> Scale -> Rotate back
    pts = pts @ R      # Note: pts @ R is equivalent to pts @ (R.T).T
    pts = pts @ S      
    pts = pts @ R.t()    

    # 6. Shift back
    pts = pts + center

    return to_flat(pts)

def per_point_jitter(x, strength, in_place=False):
    if in_place:
        pts=to_points(x)  # directly modify input tensor without cloning
    else:
        pts = to_points(x).clone()
    
    # Generate uniform noise directly on the target device
    # torch.rand generates [0, 1), so we scale it to [1-str, 1+str)
    low = 1.0 - strength
    high = 1.0 + strength
    noise = (high - low) * torch.rand(pts.shape, device=x.device, dtype=x.dtype) + low
    
    pts *= noise
    return to_flat(pts)
    
def create_keypoint_transforms(config):
    return partial(train_transform, config=config), nontrain_transform

def train_transform(input: torch.Tensor, config: dict): #x is (T, 42) tensor of keypoint coords for one video sample
        x = input.clone()  # Clone to avoid modifying the origin
        angle_rad = np.deg2rad(np.random.uniform(-config["rotation_max_angle_deg"], config["rotation_max_angle_deg"]))

        scale_x = np.random.uniform(config["stretch_scale_x"][0], config["stretch_scale_x"][1])
        scale_y = np.random.uniform(config["stretch_scale_y"][0], config["stretch_scale_y"][1])

        for i in range(x.shape[0]):
            x[i] = rotate(x[i], angle_rad, in_place=True)
            x[i] = rand_stretch_aligned(x[i], scale_x=scale_x, scale_y=scale_y, in_place=True)
        x = per_point_jitter(x, strength=config["jitter_strength"], in_place=True)
        return x

def nontrain_transform(input: torch.Tensor):
        return input