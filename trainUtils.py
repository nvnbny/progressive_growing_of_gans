import torch.nn as nn, torch, models, dataUtils, numpy as np, os, pdb, PIL.Image as Image
from torch import FloatTensor as FT
from torch.autograd.variable import Variable
from torch.optim import Adam
from datetime import datetime
import config        


def switchTrainable(nNet, status):
    """
    This is used to switch models parameters to trainable or not
    """
    for p in nNet.parameters(): p.requires_grad = status


def modifyLR(optimizer, lr):
    """
    This function will change LR
    """
    for param in optimizer.param_groups:
        param['lr'] = lr

        
def writeFile(path, content, mode):
    """
    This will write content to a give file
    """
    file = open(path, mode)
    file.write(content); file.write('\n')
    file.close()


class Trainer:
    """
    Trainer class with hyperparams, log, train function etc.
    """
    def __init__(self):

        # Paths        
        self.LOG_DIR=config.LOG_DIR
        self.DATA_PATH=config.DATA_PATH
        self.logDir=config.logDir
        self.modelFname=config.modelFname
        
        # Hyperparams
        self.dLR=config.dLR; self.gLR=config.gLR
        self.latentSize = config.latentSize
        self.batchSizes =  config.batchSizes
        self.resolutions = config.resolutions
        self.startRes = config.startRes
        self.startStage = config.startStage

        # model 
        self.createModels()
        self.loss_mse = nn.MSELoss().cuda()
        self.genLoss = []; self.genDiscLoss = [] 
        self.discLoss = []; self.discRealLoss = []; self.discFakeLoss = []
        
        # Log
        if self.logDir==None: self.createLogDir()
        else: self.loadPretrainedWts()
        print(f'{datetime.now():%d-%m-%HH:%MM} - Logging in-' + self.logDir)
                
    def createModels(self):
        """
        This function will create models and their optimizers
        """
        self.gen = models.Generator().cuda()
        self.disc = models.Discriminator().cuda()
        self.gOptimizer = Adam(self.gen.parameters(), lr = self.gLR, betas=(0.0, 0.99))
        self.dOptimizer = Adam(self.disc.parameters(), lr = self.dLR, betas=(0.0, 0.99))
        
        print('Models Instantiated. # of trainable parameters Disc:%e; Gen:%e' 
              %(sum([np.prod([*p.size()]) for p in self.disc.parameters()]), 
                sum([np.prod([*p.size()]) for p in self.gen.parameters()])))
        
    def createLogDir(self):
        """
        Create log dir
        """
        self.logDir = self.LOG_DIR + 'pggan-log-' + f'{datetime.now():%d-%m-%HH:%MM}/'
        try: os.makedirs(name=self.logDir)
        except: print('WARNING: Logging in previously created folder')
        writeFile(self.logDir + 'log.txt', self.logParameters(), 'w')
    
    def loadPretrainedWts(self):
        """
        From log dir, load wts
        """
        self.logDir = self.LOG_DIR + self.logDir
        wtsDict = torch.load(self.logDir + self.modelFname, map_location=lambda storage, loc: storage)

        self.disc.load_state_dict(wtsDict['disc'])
        self.gen.load_state_dict(wtsDict['gen'])
        self.dOptimizer.load_state_dict(wtsDict['dOptimizer'])
        self.gOptimizer.load_state_dict(wtsDict['gOptimizer'])

    def logParameters(self):
        """
        This function will return hyperparameters and architecture as string
        """
        hyperParams = f'HYPERPARAMETERS - dLR-{self.dLR}|gLR-{self.gLR}'
        architecture = '\n\n' + str(self.disc) + '\n\n' + str(self.gen) + '\n\n'
        print(hyperParams)    
        return hyperParams + architecture
    
    def logTrainingStats(self):
        """
        Print and write mean losses, save images generated
        """
        # Average all stats and log
        genLoss_ = np.mean(self.genLoss[-self.logStep:])
        genDiscLoss_ = np.mean(self.genDiscLoss[-self.logStep:])
        discLoss_ = np.mean(self.discLoss[-self.logStep:])
        discRealLoss_ = np.mean(self.discRealLoss[-self.logStep:])
        discFakeLoss_ = np.mean(self.discFakeLoss[-self.logStep:])
        stats = f'{datetime.now():%HH:%MM}| {self.res}| {self.stage}| {self.n}/{self.nIterations}| {genLoss_:.4f}| {discLoss_:.4f}| {genDiscLoss_:.4f}| {discRealLoss_:.4f}| {discFakeLoss_:.4f}| {self.dLR:.2e}| {self.gLR:.2e}'
        print(stats); writeFile(self.logDir + 'log.txt', stats, 'a')
        
        # Loop through each image and process
        for _ in range(8):    
            # Fake
            z = self.getNoise(1)
            fake = self.gen(x=z, fadeWt=self.fadeWt)
            f = dataUtils.tensorToImage(fake[0])
            
            # real
            self.callDataIteration()
            r = dataUtils.tensorToImage(self.real[0])
            
            try: img = np.vstack((img, np.hstack((f, r))))
            except: img = np.hstack((f, r))

        # save samples
        Image.fromarray(img).save(self.logDir + str(self.res) + '_' + self.stage + '_' + str(self.n) + '.jpg')
    
    def saveModelCheckpoint(self):
        """
        Saves model Check point
        """
        torch.save({'disc':self.disc.state_dict(), 'dOptimizer':self.dOptimizer.state_dict(),
                    'gen':self.gen.state_dict(), 'gOptimizer':self.gOptimizer.state_dict()}, 
                   self.logDir + 'modelCheckpoint_'+str(self.res)+'_'+self.stage+'_'+str(self.n)+'.pth.tar')    
    
    def callDataIteration(self):
        """
        This function will call next value of dataiterator
        """        
        # Next Batch
        try: real = self.dataIterator.next()
        except StopIteration: 
            self.dataIterator = iter(self.dataloader)
            real = self.dataIterator.next()
        
        self.real = real.cuda()
    
    def getNoise(self, bs=None):
        """
        This function will return noise
        """
        if bs == None : 
            try: bs = self.batchSize
            except: bs = 1
        return FT(bs, self.latentSize).normal_().cuda()
          
    def trainDiscriminator(self):
        """
        Do one step for discriminator
        """
        self.dOptimizer.zero_grad()
        switchTrainable(self.disc, True)

        # Noisy Labels
        one = 1 + np.random.randn() * 0.3
        zero = np.random.randn() * 0.3
        
        # real
        dRealOut = self.disc(x=self.real.detach(), fadeWt=self.fadeWt)
        discRealLoss_ = 0.5 * torch.mean((dRealOut - one)**2)
        
        # fake
        self.z = self.getNoise()
        self.fake = self.gen(x=self.z, fadeWt=self.fadeWt)
        dFakeOut = self.disc(x=self.fake.detach(), fadeWt=self.fadeWt)
        discFakeLoss_ = 0.5 * torch.mean((dFakeOut - zero)**2)
        
        discLoss_ = discRealLoss_ + discFakeLoss_
        discLoss_.backward(); self.dOptimizer.step()
        return discLoss_.item(), discRealLoss_.item(), discFakeLoss_.item()
    
    def trainGenerator(self):
        """
        Train Generator for 1 step
        """
        self.gOptimizer.zero_grad()
        switchTrainable(self.disc, False)
        
        self.z = self.getNoise()
        self.fake = self.gen(x=self.z, fadeWt=self.fadeWt)
        genDiscLoss_ = torch.mean((self.disc(x=self.fake, fadeWt=self.fadeWt) - True)**2)
        
        genLoss_ = genDiscLoss_
        genLoss_.backward(); self.gOptimizer.step()
        return genLoss_.item(), genDiscLoss_.item()
    
    def train(self):
        """
        Train function 
        """ 
        samplesPerStage=config.samplesPerStage 
        self.logStep = config.logStep
        modifyLR(optimizer=self.gOptimizer, lr=self.gLR); modifyLR(optimizer=self.dOptimizer, lr=self.dLR)
        print('Time   |res |stage|It        |gLoss  |dLoss  |gDLoss |dRLoss |dFLoss |dLR      |gLR      ')
        
        # for every resolution
        for i, self.res in enumerate(self.resolutions):
            # in case starting from between
            if self.res < self.startRes: continue
            
            self.batchSize = self.batchSizes[self.res]
            self.nIterations = samplesPerStage // self.batchSize
            
            # for every stage
            for self.stage in ['fade', 'stab']:
                if self.stage == 'fade':
                    # load new dl if stage is fade or we have loaded data 
                    self.dataloader = dataUtils.loadData(path=self.DATA_PATH, batchSize=self.batchSize, res=self.res)
                    self.dataIterator = iter(self.dataloader)
                    
                    # No fade if this is the first res
                    if i + 1 == 1 and self.stage == 'fade': continue
                    
                    # No fade if continuing from stab after loading model  
                    if self.startStage == 'stab' and self.res == self.startRes: continue 
                
                # for every batch
                for self.n in range(1, self.nIterations+1):
                    self.fadeWt = i + 1 if self.stage == 'stab' else i + (self.n * self.batchSize)/ samplesPerStage
                    self.callDataIteration()

                    # Train Disc
                    discLoss_, discRealLoss_, discFakeLoss_ = self.trainDiscriminator()
                    self.discLoss.append(discLoss_); self.discRealLoss.append(discRealLoss_)
                    self.discFakeLoss.append(discFakeLoss_)

                    # Train Gen
                    genLoss_, genDiscLoss_ = self.trainGenerator()
                    self.genLoss.append(genLoss_); self.genDiscLoss.append(genDiscLoss_)

                    # log
                    if self.n % self.logStep == 0 or self.n % (self.nIterations) == 0 : self.logTrainingStats()
            
                # save model for every res and stage
                if config.saveModel: self.saveModelCheckpoint()

