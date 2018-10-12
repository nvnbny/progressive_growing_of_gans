import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pdb


class PixelNormalization(nn.Module):
    """
    This is the per pixel normalization layer. This will devide each x, y by channel root mean square
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-8) ** 0.5

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then 
    it will multiply the scale for every forward pass
    """
    def __init__(self, inCh, outCh, kernelSize, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=stride, padding=padding)
        
        # new bias to use after wscale
        self.bias = self.conv.bias
        self.conv.bias = None
        
        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        self.wtScale = gain/np.sqrt(fanIn)
        
        # init
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))
        
    def forward(self, x):
        return self.conv(x) * self.wtScale + self.bias.view(1, self.bias.shape[0], 1, 1)

    def __repr__(self):
        return self.__class__.__name__ + self.name

    
class BatchStdConcat(nn.Module):
    """
    Add std to last layer group of disc to improve variance
    """
    def __init__(self, groupSize=4):
        super().__init__()
        self.groupSize=4

    def forward(self, x):
        shape = list(x.size())                                              # NCHW - Initial size
        xStd = x.view(self.groupSize, -1, shape[1], shape[2], shape[3])     # GMCHW - split batch as groups of 4  
        xStd -= torch.mean(xStd, dim=0, keepdim=True)                       # GMCHW - Subract mean of shape 1MCHW
        xStd = torch.mean(xStd ** 2, dim=0, keepdim=False)                  # MCHW - Take mean of squares
        xStd = (xStd + 1e-08) ** 0.5                                        # MCHW - Take std 
        xStd = torch.mean(xStd.view(int(shape[0]/self.groupSize), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
                                                                            # M111 - Take mean across CHW
        xStd = xStd.repeat(self.groupSize, 1, shape[2], shape[3])           # N1HW - Expand to same shape as x with one channel 
        return torch.cat([x, xStd], 1)
    
    def __repr__(self):
        return self.__class__.__name__ + '(Group Size = %s)' % (self.groupSize)
    
    
class ProcessGenLevel(nn.Module):
    """
    Based on the fade wt, this module will use relevant conv layer levels to return generated image
    """
    def __init__(self, chain, post):
        super().__init__()
        self.chain = chain
        self.post = post
        self.nLevels = len(self.chain)

    def forward(self, x, fadeWt):
        
        # Calculate levels invloved (in case of fade stage) and the wts for each level
        prevLevel, curLevel = int(np.floor(fadeWt-1)), int(np.ceil(fadeWt-1))
        curLevelWt = fadeWt-int(fadeWt); prevLevelWt = 1 - curLevelWt
        fade = False if prevLevel==curLevel else True
        
        # Loop through all levels till current level
        for lev in range(curLevel+1):
            x = self.chain[lev](x)
            
            # process prev level image only if fade stage
            if lev == prevLevel and fade: 
                prevLevel_x = self.post[lev](x)
            
            if lev == curLevel: 
                x = self.post[lev](x)
                
                # Calculate wted x if it is fade stage
                if fade: 
                    x = curLevelWt * x + prevLevelWt * F.upsample(prevLevel_x, scale_factor=2, mode='nearest')
        return x


class ProcessDiscLevel(nn.Module):
    """
    Based on the fade wt, this module will use relevant conv layer levels to return generated image
    """
    def __init__(self, pre, chain):
        super().__init__()
        self.pre = pre
        self.chain = chain
        self.nLevels = len(self.chain)

    def forward(self, x, fadeWt):
        
        # Calculate levels invloved (in case of fade stage) and the wts for each level
        curLevel = int(self.nLevels - np.ceil(fadeWt))
        prevLevel = int(self.nLevels - np.floor(fadeWt))
        curLevelWt = fadeWt-int(fadeWt); prevLevelWt = 1 - curLevelWt
        fade = False if prevLevel==curLevel else True
        
        # If it is stab phase
        if not fade:
            x = self.pre[curLevel](x)
            x = self.chain[curLevel](x)
        
        else:
            curLevel_x = self.pre[curLevel](x)
            curLevel_x = self.chain[curLevel](curLevel_x)
            
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            prevLevel_x = self.pre[prevLevel](x)
            x = curLevelWt * curLevel_x + prevLevelWt * prevLevel_x
            x = self.chain[prevLevel](x)
            
        # Loop through all levels
        for lev in range(prevLevel + 1, self.nLevels):
            x = self.chain[lev](x)

        return x

    
class ReshapeLayer(nn.Module):
    """
    Reshape latent vector layer
    """
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)
