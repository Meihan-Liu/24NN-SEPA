# Structure enhanced prototypical alignment for unsupervised cross-domain node classification
This is the source code of Neural Networks-2024 paper "[Structure enhanced prototypical alignment for unsupervised cross-domain node classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608024003204)" (SEPA).

<img src="https://github.com/Meihan-Liu/24NN-SEPA/blob/main/fig/model.png" alt="image" width="500">

# Requirements
This code requires the following:
* torch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.1.0
* numpy==1.19.2
* scikit-learn==0.24.2

# Dataset
Datasets used in the paper are all publicly available datasets. You can find [Twitch](https://github.com/benedekrozemberczki/datasets#twitch-social-networks) and [Citation](https://github.com/yuntaodu/ASN/tree/main/data) via the links.

# Cite
If you compare with, build on, or use aspects of SEPA framework, please consider citing the following paper:

```
@article{liu2024structure,
  title={Structure enhanced prototypical alignment for unsupervised cross-domain node classification},
  author={Liu, Meihan and Zhang, Zhen and Ma, Ning and Gu, Ming and Wang, Haishuai and Zhou, Sheng and Bu, Jiajun},
  journal={Neural Networks},
  volume={177},
  pages={106396},
  year={2024},
  publisher={Elsevier}
}
```
