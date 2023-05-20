# V-Net (2016)
Vnet extends Unet to process 3D MRI volumes. In contrast to processing the input 3D volumes slice-wise, they proposed to use 3D convolutions. In the end, medical images have an inherent 3D structure, and slice-wise processing is sub-optimal. The main modifications of Vnet are:

- Motivated by similar works on image classification, they replaced max-pooling operations with strided convolutions. This is performed through convolution with 2 × 2 × 2 kernels applied with stride 2.

- 3D convolutions with padding are performed in each stage using 5×5×5 kernels.

- Short residual connections are also employed in both parts of the network.

- They use 3D transpose convolutions in order to increase the size of the inputs, followed by one to three conv layers. Feature maps are halved in every decoder layer.

- All the above can be illustrated in this image:

![vnet-model](https://github.com/Er-Divyesh-Sethiya/U-Net-Architecture/assets/103837830/c1965528-9d5f-4fad-a495-9f25997e6b92)

- Finally, in this work, the Dice loss was introduced which is a common loss function in segmentation. You can find the implementation of Vnet in our open-source library.
