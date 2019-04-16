# Deep-Feature-Tansformer
Deep Feature Transformer module is used to estimate the transformation parameters  in the feature domain of the image.

# Street View Housing Number (SVHN) Experiment 
To illustrate that our proposed DFTN can also be applied to other types of data classification, we carried out an additional experiment by using the real-world dataset, Street View House Numbers (SVHN) [1]. To maintain a fair comparison, we followed the experiment design given in [2] and performed pre-processing on the dataset to crop the data by taking both 64×64 and 128×128 crops around each digit sequence. We then repeated the training of the baseline [3] for both STN+CNN [4] (single and multiple STN) and the proposed DFTN+CNN to complete the classification experiment, and the results are given in the next table. From the results, it can still be concluded that our proposed DFTN achieves a better performance than the existing STN.


































REFERENCES
[1] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, BoWu, and Andrew Y. Ng. 2011. Reading Digits in Natural Imageswith Unsupervised Feature Learning.
[2]Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. 2015. Spatial transformer networks. In Advances in neural information processing systems. 2017–2025.
[3]Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. 2014. Multiple Object Recognition with Visual Attention. Computer Science (2014)
