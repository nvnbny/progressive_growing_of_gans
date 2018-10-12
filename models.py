import modelUtils, torch.nn as nn, numpy as np

##############################################################
# Generator
##############################################################

def genConvBlock(net, inCh, outCh, kernelSize, padding, stride=1, negSlope=0.2):
    """
    This funtion appends and returns LIST of conv blocks for gen
    """
    net += [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]    
    net += [nn.LeakyReLU(negative_slope=negSlope)]
    net += [modelUtils.PixelNormalization()]
    return net


def toRGBBlock(inCh, outCh, kernelSize=1, stride=1, padding=0):
    """
    This creates an sequential post processing block (ch to channel)
    """
    net = [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding, gain=1)]
    return nn.Sequential(*net)


class Generator(nn.Module):
    """
    Progressive growth of GAN generator
    """
    def __init__(self, nOutputChannels=3, resolution=256, fmapBase=8192, fmapDecay=1.0, fmapMax=512, latentSize=512):
        super().__init__()
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        nBlocks = int(np.log2(resolution))
        
        chain = nn.ModuleList()
        post = nn.ModuleList()
        net = []
        
        # First block 4x4
        inCh, outCh = latentSize, self.getNoChannels(1)
        net += [modelUtils.ReshapeLayer([latentSize, 1, 1])]
        net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=4, padding=3) 
        net = genConvBlock(net=net, inCh=outCh, outCh=outCh, kernelSize=3, padding=1)
        toRGB = toRGBBlock(inCh=outCh, outCh=nOutputChannels)
        
        chain.append(nn.Sequential(*net))
        post.append(toRGB)
        
        # Blocks 8x8 and up
        for i in range(2, nBlocks):
            inCh, outCh = self.getNoChannels(i-1), self.getNoChannels(i)
            net = [nn.Upsample(scale_factor=2, mode='nearest')]
            net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=3, padding=1) 
            net = genConvBlock(net=net, inCh=outCh, outCh=outCh, kernelSize=3, padding=1)
            toRGB = toRGBBlock(inCh=outCh, outCh=nOutputChannels)
            
            chain.append(nn.Sequential(*net))
            post.append(toRGB)
        
        self.net = modelUtils.ProcessGenLevel(chain, post)

    def getNoChannels(self, stage):
        """
        Get no. of filters based on below formulae
        """
        return min(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMax)

    def forward(self, x, fadeWt=None):
        return self.net(x, fadeWt)


##############################################################
# Discriminator
##############################################################

def discConvBLock(net, inCh, outCh, kernelSize, padding, stride=1,  negSlope=0.2):
    """
    This funtion appends and returns LIST of conv blocks for disc
    """
    net += [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]
    net += [nn.LeakyReLU(negative_slope=negSlope)]
    return net
    

def fromRGBBlock(inCh, outCh, kernelSize=1, stride=1, padding=0, negSlope=0.2):
    """
    This creates an preprocessing block (from RBG to ch)
    """
    net = [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]
    net += [nn.LeakyReLU(negative_slope=negSlope)]
    return nn.Sequential(*net)


class Discriminator(nn.Module):
    """
    Progressive growth of GAN Disc 
    """
    def __init__(self, nOutputChannels=3, resolution=256, fmapBase=8192, fmapDecay=1.0, fmapMax=512):
        super().__init__()
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        nBlocks = int(np.log2(resolution))
        
        chain = nn.ModuleList()
        pre = nn.ModuleList()

        # Last preproccesing layer (e.g. 256 x 256)
        inCh, outCh = nOutputChannels, self.getNoChannels(nBlocks-1)
        fromRGB = fromRGBBlock(inCh=inCh, outCh=outCh)
        pre.append(fromRGB)
        
        # Blocks 256 x 256 to 8 x 8
        for i in range(nBlocks-1, 1, -1):
            inCh, outCh = self.getNoChannels(i), self.getNoChannels(i-1)
            net = []
            net = discConvBLock(net, inCh=inCh, outCh=inCh, kernelSize=3, padding=1)
            net = discConvBLock(net, inCh=inCh, outCh=outCh, kernelSize=3, padding=1)
            net += [nn.AvgPool2d(kernel_size=2, stride=2)]
            chain.append(nn.Sequential(*net))
            
            fromRGB = fromRGBBlock(inCh=nOutputChannels, outCh=outCh)
            pre.append(fromRGB)
        
        # Ultimate conv Block with sigmoid (4x4)
        net = []
        inCh, outCh = self.getNoChannels(1), self.getNoChannels(1)
        net += [modelUtils.BatchStdConcat()]
        inCh += 1
        
        net = discConvBLock(net, inCh=inCh, outCh=outCh, kernelSize=3, padding=1)
        inCh, outCh = outCh, self.getNoChannels(0)
        
        net = discConvBLock(net, inCh=inCh, outCh=outCh, kernelSize=4, padding=0)
        inCh, outCh = outCh, 1
        
        net += [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=1, stride=1, padding=0, gain=1)]
        
        chain.append(nn.Sequential(*net))
        self.net = modelUtils.ProcessDiscLevel(pre=pre, chain=chain)
    
    def getNoChannels(self, stage):
        """
        Get no. of filters based on below formulae
        """
        return min(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMax)
    
    def forward(self, x, fadeWt=None):
        return self.net(x, fadeWt)

