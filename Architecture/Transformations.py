import torch
import numpy as np
from scipy.ndimage import rotate

class ToTensor3D(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input_base']

        ret_input = input_data.transpose(3, 0, 1, 2)   # Pytorch supports N x C x X_dim x Y_dim
        ret_input = torch.from_numpy(ret_input).float()
        rdict['input_base'] = ret_input

        input_data = sample['input_follow']

        ret_input = input_data.transpose(3, 0, 1, 2)   # Pytorch supports N x C x X_dim x Y_dim
        ret_input = torch.from_numpy(ret_input).float()
        rdict['input_follow'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                ret_gt = torch.tensor(gt_data)

                rdict['gt'] = ret_gt
            if 'base_seg' in sample:
                seg_data = sample['base_seg']
                if seg_data is not None:
                    ret_seg = seg_data.transpose(3,0,1,2)
                    rdict['base_seg'] = torch.from_numpy(ret_seg).float()
                seg_data = sample['follow_seg']
                if seg_data is not None:
                    ret_seg = seg_data.transpose(3,0,1,2)
                    rdict['follow_seg'] = torch.from_numpy(ret_seg).float() 
 
        sample.update(rdict)
        return sample

class RandomRotation3D(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """

    def __init__(self, degrees, p=0.5, axis=0, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.p = 0.5
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data_base = sample['input_base']
        input_data_follow = sample['input_follow']

        if len(sample['input_base'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        if(torch.rand(1)<self.p):
            angle = self.get_params(self.degrees)

            input_rotated_base = np.zeros(input_data_base.shape, dtype=input_data_base.dtype)
            input_rotated_follow = np.zeros(input_data_follow.shape, dtype=input_data_follow.dtype)

            if 'base_seg' in sample:
                gt_data_base = sample['base_seg'] 
                gt_rotated_base = np.zeros(gt_data_base.shape, dtype=gt_data_base.dtype)
                gt_data_follow = sample['follow_seg'] 
                gt_rotated_follow = np.zeros(gt_data_follow.shape, dtype=gt_data_follow.dtype)

            input_rotated_base = rotate(input_data_base, float(angle), reshape=False, order=1,mode='nearest')
            input_rotated_follow = rotate(input_data_follow, float(angle), reshape=False, order=1,mode='nearest')

            if 'base_seg' in sample:
                gt_rotated_base = rotate(gt_data_base, float(angle), reshape=False, order=self.order,mode='nearest')
                gt_rotated_base = (gt_rotated_base > 0.5).astype(np.single)

                gt_rotated_follow = rotate(gt_data_follow, float(angle), reshape=False, order=self.order,mode='nearest')
                gt_rotated_follow = (gt_rotated_follow > 0.5).astype(np.single)

            # Update the dictionary with transformed image and labels
            rdict['input_base'] = input_rotated_base
            rdict['input_follow'] = input_rotated_follow

            if 'base_seg' in sample:
                rdict['base_seg'] = gt_rotated_base
                rdict['follow_seg'] = gt_rotated_follow

            sample.update(rdict)
        return sample
