import torch
import numpy as np
from scipy.ndimage import rotate

class ToTensor3D(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        ret_input = input_data.transpose(3, 0, 1, 2)   # Pytorch supports N x C x X_dim x Y_dim
        ret_input = torch.from_numpy(ret_input).float()
        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                ret_gt = torch.tensor(gt_data)

                rdict['gt'] = ret_gt
            if 'seg' in sample:
                seg_data = sample['seg']
                if seg_data is not None:
                    ret_seg = seg_data.transpose(3,0,1,2)
                    rdict['seg'] = torch.from_numpy(ret_seg).float() 
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
        input_data = sample['input']
        if len(sample['input'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        if(torch.rand(1)<self.p):
            angle = self.get_params(self.degrees)

            input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

            if 'seg' in sample:
                gt_data = sample['seg'] 
                gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype)

            input_rotated = rotate(input_data, float(angle), reshape=False, order=1,mode='nearest')
            if 'seg' in sample:
                gt_rotated = rotate(gt_data, float(angle), reshape=False, order=self.order,mode='nearest')
                gt_rotated = (gt_rotated > 0.5).astype(np.single)
            
            # Update the dictionary with transformed image and labels
            rdict['input'] = input_rotated

            if 'seg' in sample:
                rdict['seg'] = gt_rotated
            sample.update(rdict)
        return sample
