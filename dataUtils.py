import os, PIL.Image as Image, numpy as np, torch, cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob

################################################################################
# Util Functions
################################################################################

def arrayToImage(arr):
    """
    Feed in numpy array in the range of -1 to 1 and return PIL image
    """
    return Image.fromarray(((arr*127)+127).astype('uint8'))


def tensorToImage(tensor):
    """
    Convert a flipped channel tensor to a PIL image
    """
    arr = np.transpose(np.array(tensor.detach()), (1,2,0))
    arr[arr>1]=1; arr[arr<-1]=-1
    arr = ((arr + 1) * 127.5).astype('uint8')
    return Image.fromarray(arr)


def arrayToTensor(array):
    """
    Convert numpy array to tensor after transposing and float 32 conversion  
    """
    return torch.from_numpy(np.transpose(array.astype('float32'), (2, 0, 1)))


################################################################################
# Dataloader functions
################################################################################

class CelebDataset(Dataset):
    """
    This class will create dataset for bodylessDataset
    """
    def __init__(self, path, res):
        self.paths = glob(os.path.join(path, '*.jpg'))
        self.res =  res
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Get data
        img = cv2.imread(self.paths[idx])
    
        # preprocess 
        img = img[20:198, 0:178]
        img = img[:, :, ::-1].astype('float32')
        img = cv2.resize(img, (self.res, self.res), cv2.INTER_NEAREST) 
        img = img/127.5 - 1
        
        # convert to tensor
        img = arrayToTensor(img)
        return img


def loadData(path, res, batchSize):
    """
    Function to load and preprocess data from path
    """
    dataset = CelebDataset(path, res)
    dataloader = DataLoader(dataset, batch_size=batchSize, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)

    # dataIterator =  iter(dataloader); img = dataIterator.next()
    # print(f'Data Loaded - Image Shape: {str(img.size())}')
    return dataloader

      
