# Deep-Feature-Tansformer
Deep Feature Transformer module is used to estimate the transformation parameters  in the feature domain of the image.

![image](https://user-images.githubusercontent.com/44599319/56183503-6a12b200-6049-11e9-8f4b-8f51e96698ad.png)

# Street View Housing Number (SVHN) Experiment 
To illustrate that our proposed DFTN can also be applied to other types of data classification, we carried out an additional experiment by using the real-world dataset, Street View House Numbers (SVHN) [1]. To maintain a fair comparison, we followed the experiment design given in [2] and performed pre-processing on the dataset to crop the data by taking both 64×64 and 128×128 crops around each digit sequence. We then repeated the training of the baseline [3] for both STN+CNN [4] (single and multiple STN) and the proposed DFTN+CNN to complete the classification experiment, and the results are given in the next table. From the results, it can still be concluded that our proposed DFTN achieves a better performance than the existing STN.

![image](https://user-images.githubusercontent.com/44599319/56183391-0092a380-6049-11e9-9edf-f180654550dd.png)



## Robustness test on missing information in corrupted images

In this section, we carry out an experiment to show the power of the proposed DFTN in achieving a certain level of robustness against the missing information inside corrupted images, simulating the scenarios that some parts of images are lost during transformations or transmissions over noisy communication networks. Architecturally, we still use the same combination that the proposed DFTN integrates with VDSR, and hence the evaluations are conducted between
such a combination DFTN+VDSR and the combination of the existing state of the arts: STN+VDSR. To simulate the missing information inside corrupted images, we use the dataset CelebA faces [5] and mask all the input images at the center by using a mask of 16×16 pixels. We then add spatial transformation effects upon all the masked images by using five different transformations, including R, S, RS, T, and RTS. To increase the uncertainty, making the testing more challenging on purpose, we introduced a random mechanism in selecting the transformation effects, including :(i )the rotation angle is randomly selected from the range of [-45, +45]; (ii) the scaling factor is randomly selected from the range of [0.6, 1]; (iii) the translation values are randomly selected from the range of [3, 9]; and finally (iv) combinations of the above values are also randomly selected to formulate the effect of RS and RTS. Some examples of
such corrupted images are illustrated in part (b) of Fig.1.

![image](https://user-images.githubusercontent.com/44599319/56183474-4d767a00-6049-11e9-8214-02bf67ea07ef.png)


All the experimental results are shown in the next Table, from which it can be seen that our proposed DFTN still outperforms the existing STN in terms of both PSNR and SSIM measurements.

![image](https://user-images.githubusercontent.com/44599319/56183545-8c0c3480-6049-11e9-9efa-b03db505df69.png)

As shown in Fig.1, in addition, our proposed DFTN also outperforms STN significantly in terms of subjective visual inspections. While the existing STN can successfully correct the spatial transformations, it fails to recover the missing information in the masked area. In contrast, the proposed DFTN not only successfully overcomes the distorted effect of spatial transformation but also successfully recover the lost information for the masked regions. Further examination of the part (d) in Fig.1 reveals that the output images produced by STN+VDSR also introduced some extra distortion around the four corners of most sample images. This is due to the fact that STN used bilinear interpolation, which inevitably incurs the extrapolation process in the pixel domain.
In contrast, our proposed DFTN does not have any such distortion as shown in part (c) of Fig.1. This is due to the fact that our proposed DFTN works in deep feature domain and hence the extrapolation in grid sampling can only affect the feature map, not the pixels.

Compared with the existing STN, in summary, our proposed DFTN operates in deep feature domain with not only a multi-channel feature map, but also further feature refinement, to provide detailed hierarchies and facilitate multi-channel affine transformations. In essence, the proposed DFTN can be regarded as a weighted sum of 512 STNs, providing the foundation for such overwhelmingly better performances than that of the existing STN.



### REFERENCES
[1] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, BoWu, and Andrew Y. Ng. 2011. Reading Digits in Natural Imageswith Unsupervised Feature Learning.


[2]Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. 2015. Spatial transformer networks. In Advances in neural information processing systems. 2017–2025.


[3]Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. 2014. Multiple Object Recognition with Visual Attention. Computer Science (2014)


[5]Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2014. Deep Learning Face Attributes in the Wild. (2014), 3730–3738.
