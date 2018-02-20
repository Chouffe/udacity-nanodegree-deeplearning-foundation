# Face Generation

## How to train a GAN - GAN Hacks

* [https://github.com/soumith/ganhacks](GAN Hacks)
* Use Xavier Initialization to break symmetry: [https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/](Understanding Xavier Initialization)
* Use `Dropout`
  * Generator: Improve performance by increasing the noise and making the network more noise tolerant
  * [Pix2Pix Paper](https://arxiv.org/pdf/1611.07004.pdf)
  * Discriminator
* Use `Conv2D` for downsampling: avoid making sparse gradients as the stability of the GAN game suffers if you have sparse gradients
* Use `sigmoid` as the output layer
* Use Leaky ReLU instead of ReLU
* Use Batch Norm
* To prevent the discriminator from being too strong as well as to help it generalise better the discriminator labels are reduced from 1 to 0.9. This is called label smoothing (one-sided). A possible TensorFlow implementation is `labels = tf.ones_like(tensor) * (1 - smooth)`
* [Paper: Improved techniques for training GANs](https://arxiv.org/pdf/1606.03498.pdf)
* One can train twice the `generator` when the `discriminator` is trained once so that the discriminator loss does not go to zero

## Good Hyperparameters for DCGAN

* `beta1`: between `0.1` and `0.3`
* `batch_size`: between `16` and `32`
* `lr`: between `0.0001` and `0.0008`
