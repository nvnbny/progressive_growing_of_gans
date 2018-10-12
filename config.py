
# Paths
LOG_DIR='./log/'                        # Path where Generated samples and model snapshots 
DATA_PATH='./data/img_align_celeba/'    # Path where celeb data is kept 
logDir=None                             # Path of previous log if training is to be continued; Ignore if training from scratch
modelFname=None                         # Name of pretrained model if training is to be continued; Ignore if training from scratch
saveModel=True                          # Save models at the end of each stage, turn off if you don't have space

# Hyperparameters
dLR=1e-3                                                        # Discriminator Learning Rate
gLR=1e-3                                                        # Generator Learning Rate
latentSize=512                                                  # Size of noise vector
batchSizes={4:16, 8:16, 16:16, 32:16, 64:8, 128:4, 256:4}       # Batch sizes for each resolution
resolutions=[4, 8, 16, 32, 64, 128, 256]                        # Dict of Resolutions  

# Other
samplesPerStage=600000                                          # real of samples for each stage
logStep=2000                                                    # log every x steps
startRes=4                                                      # Starting resolution if trained is to be continued;Ignore if training from scratch 
startStage='None'                                               # Starting stage if trained is to be continued;Ignore if training from scratch

