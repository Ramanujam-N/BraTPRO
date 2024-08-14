from torch.utils.data import Dataset,DataLoader
import nibabel as nib
import skimage.transform as skiform
import numpy as np
import pydicom as dicom
import pandas as pd

class BratPro_Reader(Dataset):
    def __init__(self,data_dict,size=128,transform=None,segment=False):
        self.data_dict = data_dict
        self.transform = transform
        self.size =size
        self.segment = segment
    def __getitem__(self,index):
        prev_image =  nib.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/BraTPRO/Lumiere'+self.data_dict[index]['prev_registered'][1:]+self.data_dict[index]['prev_registered'][19:]+'_0003.nii.gz').get_fdata()
        base_image =  nib.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/BraTPRO/Lumiere'+self.data_dict[index]['baseline_registered'][1:]+self.data_dict[index]['baseline_registered'][19:]+'_0003.nii.gz').get_fdata()
        follow_image = nib.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/BraTPRO/Lumiere'+self.data_dict[index]['followup_registered'][1:]+self.data_dict[index]['followup_registered'][19:]+'_0003.nii.gz').get_fdata()

        if(self.segment):
            prev_seg_image =  nib.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/BraTPRO/Lumiere'+self.data_dict[index]['prev_seg_registered'][1:]).get_fdata()
            base_seg_image =  nib.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/BraTPRO/Lumiere'+self.data_dict[index]['baseline_seg_registered'][1:]).get_fdata()
            follow_seg_image = nib.load('/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a/LabData/BraTPRO/Lumiere'+self.data_dict[index]['followup_seg_registered'][1:]).get_fdata()

        gt = np.zeros(4)
        gt[self.data_dict[index]['response']] = 1
        # gt = self.data_dict[index]['response']

        prev_image,img_crop_para_prev = self.tight_crop_data(prev_image)
        base_image,img_crop_para_base = self.tight_crop_data(base_image)
        follow_image,img_crop_para_follow = self.tight_crop_data(follow_image)

        if(self.segment):
            prev_seg_image = prev_seg_image[img_crop_para_prev[0]:img_crop_para_prev[0] + img_crop_para_prev[1], img_crop_para_prev[2]:img_crop_para_prev[2] + img_crop_para_prev[3], img_crop_para_prev[4]:img_crop_para_prev[4] + img_crop_para_prev[5]]
            base_seg_image = base_seg_image[img_crop_para_base[0]:img_crop_para_base[0] + img_crop_para_base[1], img_crop_para_base[2]:img_crop_para_base[2] + img_crop_para_base[3], img_crop_para_base[4]:img_crop_para_base[4] + img_crop_para_base[5]]
            follow_seg_image = follow_seg_image[img_crop_para_follow[0]:img_crop_para_follow[0] + img_crop_para_follow[1], img_crop_para_follow[2]:img_crop_para_follow[2] + img_crop_para_follow[3], img_crop_para_follow[4]:img_crop_para_follow[4] + img_crop_para_follow[5]]

            # print(base_seg_image.shape)
            prev_seg_image = skiform.resize(prev_image,(self.size,)*3,order=0,preserve_range=True)
            base_seg_image = skiform.resize(base_image,(self.size,)*3,order=0,preserve_range=True)
            follow_seg_image = skiform.resize(follow_image,(self.size,)*3,order=0,preserve_range=True)

        prev_image = skiform.resize(prev_image,(self.size,)*3,order=1,preserve_range=True)
        base_image = skiform.resize(base_image,(self.size,)*3,order=1,preserve_range=True)
        follow_image = skiform.resize(follow_image,(self.size,)*3,order=1,preserve_range=True)

        prev_image -=prev_image.mean()
        prev_image /=prev_image.std() + 1e-7

        base_image -=base_image.mean()
        base_image /=base_image.std() + 1e-7

        follow_image -=follow_image.mean()
        follow_image /=follow_image.std() + 1e-7


        image = np.stack([prev_image,base_image,follow_image],axis=-1)
    
        if(self.segment):
            seg = np.stack([prev_seg_image>0,base_seg_image>0,follow_seg_image>0],axis=-1)

        data_dict = {}
        data_dict['input'] = image 
        
        if(self.segment):
            data_dict['seg'] = seg
        
        data_dict['gt'] = gt
        if(self.transform!=None):
            self.transform(data_dict)
        return data_dict
    def __len__(self):
        return len(self.data_dict)
    
    def cut_zeros1d(self, im_array):
        '''
     Find the window for cropping the data closer to the brain
     :param im_array: input array
     :return: starting and end indices, and length of non-zero intensity values
        '''

        im_list = list(im_array > 0)
        start_index = im_list.index(1)
        end_index = im_list[::-1].index(1)
        length = len(im_array[start_index:]) - end_index
        return start_index, end_index, length

    def tight_crop_data(self, img_data):
        '''
     Crop the data tighter to the brain
     :param img_data: input array
     :return: cropped image and the bounding box coordinates and dimensions.
        '''

        row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
        col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
        stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
        rsid, reid, rlen = self.cut_zeros1d(row_sum)
        csid, ceid, clen = self.cut_zeros1d(col_sum)
        ssid, seid, slen = self.cut_zeros1d(stack_sum)
        return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen], [rsid, rlen, csid, clen, ssid, slen]
    
    