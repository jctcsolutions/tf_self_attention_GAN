Tensorflow implementation of Generative Adversarial Network (GAN) with spectral normalization and self-attention (SA) layer.
*(Note: this implementation was done with Google's Colab environment which allows for FREE-GPU acceleration up to 12-hours.)*

### Dataset
The dataset used is the CelebFaces Attributes (CelebA) Dataset which consists of over 200k images of celebrities with 40 binary attribute annotations (although the attribute annotations were not used during training). This dataset can be downloaded here: https://www.kaggle.com/jessicali9530/celeba-dataset/downloads/img_align_celeba.zip/2. The unzipped folder consists of a series of individual JPEG files of celebrity portraits. 

In order to run training over this dataset using the code, they need to be converted to the tfRecord format (a serialized format). The `create_tfdataset` function provides this functionality. It first calls the `parse_img_example` function to standardize all images to the same HxW dimension using the `tf.image.resize_image_with_crop_or_pad` method, then the `normalizer` function to normalize the pixel intensity to [-1,1] (and add some random noise to the image to help training). 

### Training
The training process can be expected to take over 10+ hours to get too 100K steps, but the image output from the generator can be expected to show silhouette of portraits (albeit ghastly) very early on, as illustrated by the GIF below that shows the training progression of the generator over the same set of random noise vectors.

![alt text](https://github.com/jctcsolutions/tf_self_attention_GAN/blob/master/sagan_progression.gif?raw=true)

### Sample Results
Sample training result @ 100K training step using hinge loss function with 4x discriminator LR and spectral normalization. Self attention layer was applied after the 3rd generator convolution layer. 

![alt text](https://github.com/jctcsolutions/tf_self_attention_GAN/blob/master/sample_grid.jpg?raw=true)


---
Happy Training!
