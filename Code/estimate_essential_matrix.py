import numpy as np
import torch
from kornia.geometry.ransac import RANSAC
from kornia.feature import match_mnn, DenseSIFTDescriptor
from kornia.geometry.epipolar import essential_from_fundamental
from torchvision.transforms import Grayscale

def GetEssentialMatrix(ref_imgs, intrinsics):
    with torch.no_grad():
        left = ref_imgs[0]
        right = ref_imgs[1]

        RGB2GRAY = Grayscale(1)
        SIFT = DenseSIFTDescriptor()
        ransac = RANSAC(model_type='fundamental')

        left_desc = SIFT(RGB2GRAY(left))
        right_desc = SIFT(RGB2GRAY(right))

        left_desc = left_desc.reshape(left_desc.shape[0], left_desc.shape[1], -1)
        right_desc = right_desc.reshape(right_desc.shape[0], right_desc.shape[1], -1)
        
        mat = list()
        for i in range(left_desc.shape[0]):
            l1 = left_desc[i]
            r1 = right_desc[i]

            scores, idxs = match_mnn(l1.T,r1.T)
            l_idxs = np.vstack(np.unravel_index(idxs[:,0].cpu().numpy(), left.shape[-2:])).T
            r_idxs = np.vstack(np.unravel_index(idxs[:,1].cpu().numpy(), right.shape[-2:])).T

            F,_ = ransac(torch.tensor(l_idxs).float().cuda(), torch.tensor(r_idxs).float().cuda())
            
            mat.append(F[None,:,:])

        mat = torch.cat(mat, axis=0)
        return mat