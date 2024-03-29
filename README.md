# UDA-TFL
Paper title: Unsupervised Domain-Adaptation-Based Tensor Feature Learning With Structure Preservation

Domain adaptation (DA) is widely used in computer vision and pattern recognition applications. It is an effective process where a model is trained on objects from the source domain to predict the categories of the objects in the target domain. The aim of feature extraction in domain adaptation is to learn the best representation of the data in a certain domain and use it in other domains. However, the main challenge here is the difference between the data distributions of the source and target domains. Also, in computer vision, the data are represented as tensor objects such as 3-D images and video sequences. Most of the existing methods in DA apply vectorization to the data, which leads to information loss due to failure to preserve the natural tensor structure in a low-dimensional space. Thus, in this article, we propose unsupervised DA-based tensor feature learning (UDA-TFL) as a novel adapted feature extraction method that aims to avoid vectorization during transfer knowledge simultaneously; retain the structure of the tensor objects; reduce the data discrepancy between source and target domains; and represent the original tensor object in a lower dimensional space that is resistant to noise. Therefore, multilinear projections are determined to learn the tensor subspace without vectorizing the original tensor objects via an alternating optimization strategy. We integrate maximum mean discrepancy in the objective function to reduce the difference between source and target distributions. Extensive experiments are conducted on 39 cross-domain datasets from different fields, including images and videos. The promising results indicate that UDA-TFL significantly outperforms the state-of-the-art

Cite the paper
Braytee, A., Naji, M. and Kennedy, P.J., 2022. Unsupervised Domain-Adaptation-Based Tensor Feature Learning With Structure Preservation. IEEE Transactions on Artificial Intelligence, 3(3), pp.370-380.

Link:
https://ieeexplore.ieee.org/abstract/document/9744422?casa_token=St9UeyLS4wEAAAAA:1m79siX24VEHmuT6kih9Te-2O7GmKXVWKOKKVN4i-zVEb4h6o5YdnTZscccis1axjFhCigk5cw

Published in: IEEE Transactions on Artificial Intelligence ( Volume: 3, Issue: 3, June 2022)
