# Tristar Skincancer Problem

## My approach
I approached this problem as a scientific experiment, aiming to test a new model against data not previously explored in the literature. To establish a strong baseline, I first employed the state-of-the-art FixCaps pretrained model, which, after 45 epochs, achieved an impressive 97.95% accuracy on the test dataset.

https://github.com/Woodman718/FixCaps

Next, I selected META’s foundational vision transformer model, DINOv2, which was pretrained on hundreds of millions of images, though none were from biomedical datasets. My goal was to demonstrate that by using DINOv2 as a backbone and training only the classification head, I could still obtain reasonable results. Indeed, this approach yielded an 87% accuracy, highlighting the versatility and effectiveness of foundational models across diverse domains.

https://github.com/facebookresearch/dinov2

### My Code

My Dinov2 approach can be found here. The notebook will walk you through how to run it. I recommend you use the Colab link to view as I was unable to test the IPYNB on my personal computer without CUDA capabilities.

#### [COLAB LINK](https://colab.research.google.com/drive/1Josh_DwnrULzeYMhMl9o3kEbVWeCjfjg?usp=sharing)

#### [IPYNB FILE](DINOv2_approach.ipynb)


The Baseline CNN code can be found here. It is a very simple implementation mostly drawn from the FixCaps repo. 

#### [COLAB LINK](https://colab.research.google.com/drive/1pbS2njIJpkicWqDbdXBJI3dvTFwDG1mP?usp=sharing)

#### [IPYNB FILE](CNN_baseline.ipynb)


## Future Work


There are two things that, if given time to implement, would likely improve model performance by 5-10%. First, implementing segmentation to mask unaffected skin and background has been proven to improve vision transformers' performance immensely.

Secondly, collecting more data and setting up a fine-tuning model for DINOv2 would greatly enhance performance. This would allow the model to better separate classes when embedding into latent space.

