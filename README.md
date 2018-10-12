# PGGAN PyTorch - by **Naveen Benny**
This is a PyTorch implementation of the paper [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION by Karras et al.](https://arxiv.org/abs/1710.10196) 

### Contributions by Authors
* **Progressive Growing:** The primary contribution of the paper is a training methodology for GANs starting with low-resolution images, and then progressively increasing the resolution by adding layers to the networks as shown above

<p align="center">
  <img src="/samples/figure.png">
</p>

* **Mini-batch Discrimination:** To improve variance of generated samples, the authors add a Simplified Minibatch discrimination. Towards the end of the discriminator, batch standard deviation estimates across features and spatial locations are calculated to yeild one feature map that is then concatenated and fed to the next layer 
* To prevent escalation of signal magnitudes due to an unhealthy competition between discriminator and generator, the authors add
  * **Equalized learning rates:** Addition weight scaling to every layer using normalization constant from He’s initializer (He et al., 2015) and avoiding explicit weight initialization
  * **Pixel Normalization:** Normalize the feature vector in each pixel to unit length in the generator after each convolutional layer. They do this using a variant of “local response normalization” (Krizhevsky et al., 2012)

### Implementation
* Below are some of the samples generated from this implementation. The people on the left column do not exist, ones on the right are from the celeb dataset

<p align="center">
  <img src="/samples/compiled.png">
</p>

* More samples are available in the samples folder and in the log files
* Training Time ~ 2 days on a single Nvidia GTX 1080Ti (11GB VRAM)
* The code is well structured, modular and well commented, therefore not creating another doc. Below are the modules that are part of this code
  * **main.py** - main function
  * **trainUtils.py** - Trainer/Solver class 
  * **modelUtils.py** - Custom Layers for Gen and Disc
  * **models.py** - Generator and Discriminator Class
  * **dataUtils.py** - Dataloader and dataset 
  * **config.py** - Paths, Hyperparameters, configurations etc.
  * **log Folder** - Contains training statistics and losses, logged hyperparams, model architectures and generated samples
  * **samples Folder** - Has a few generated samples from this implementaion
  * **data Folder** - Should place the celeba dataset here. There are already a few samples placed for testing 
  * **architectures.txt** - Elaborates Gen and Disc architectures
* Differences from the official [Tensorflow Implementation](https://github.com/tkarras/progressive_growing_of_gans#progressive-growing-of-gans-for-improved-quality-stability-and-variation-official-tensorflow-implementation-of-the-iclr-2018-paper) 
  * Unlike the original implementation that uses Wasserstein - GP Loss (Gulrajani et al., 2017), this uses a simple MSE Loss. This makes training slightly unstable and requires using noisy labels, a hack that is used often with GANs. At 256 x 256 sizes, the training might become unstable. Using WGAN-GP should help solve this
  * The original implementation uses celeba-HQ dataset while this implementation uses the standard celeba dataset. The HQ dataset would probably take weeks on my machine (Original implementation takes 20 days on NVIDIA Tesla P100 GPU!!)
  * Other minor differences such as absence of linear layers in the initial layers etc.

### Steps to execute
1. Clone repo 
```
git clone https://github.com/nvnbny/progressive_growing_of_gans.git
```
2. Install dependencies 
    * PyTorch (0.4.0>)
    * PIL
    * numpy
    * cv2
3. Download and unzip [celeba dataset](https://www.kaggle.com/jessicali9530/celeba-dataset/version/2#_=_). Place the 'img_align_celeba' folder inside the ./data folder
4. Run the code below
```
python main.py
```
5. Check log folder for stats and samples

**Please feel free to raise issues/PRs**
