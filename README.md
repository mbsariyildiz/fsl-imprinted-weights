Pytorch implementation of 
[Low-Shot Learning with Imprinted Weights](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf). 

I started this repository as the clone of [this repository](https://github.com/YU1ut/imprinted-weights). But then I realized that
I had changed almost everything, therefore decided to make a separate repo on it's own. But still there may be some intact code snippets, for
which I give credit to @YU1ut.

## Important note
In the paper, Inception V1 is used as the feature extractor. However, since there is no pre-trained Inception V1 model in 
torchvision.models package, in this repo I use ResNet-50 as the feature extractor. Besides, somewhat surprisingly 
fine-tuning ResNet-50 with RMSProp (with the exact same parameters as in the paper) results in poor generalization.
Perhaps, this is yet another case where SGD with momentum is superior than RMSProp with momentum, in terms of generalization.

## Development environment
- ubuntu 18.04
- cuda 9.0
- conda 4.5.11
- python 3.6.4
- pytorch 0.4.1
- torchvision 0.2.1
- sklearn 0.19.1
- matplotlib 3.0.1
- numpy 1.15.4
- tqdm

## Dataset
Download [CUB_200_2011 Dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
and unzip it into ```data``` directory under the repository folder.

## Usage
Currently, I don't have enough time to explain every step in detail. 
Please see the scripts to understand how things work.

## Results
Please see the followings for how scores are computed.
- Each n shot setting is performed 5 times such that a different seed is used before sampling of n samples from novel classes.
- In each imprinting experiment, a sample from a novel class is augmented 10 times.
- For each score, confusion matrix is computed and then per-class recall scores are averaged accordingly.

### Average per-class recalls of novel classes in CUB-200-2011
#### w/o FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand-noFT (paper) |0.17 |0.17 |0.17 |0.17 |0.17 |
|Imprinting (paper)|21.26 |28.69 |39.52 |45.77 |49.32|
|Imprinting + Aug (paper) |21.40 |30.03 |39.35 |46.35 |49.80|
|**Rand-noFT**|**0.00** |**0.00** |**0.00** |**0.00** |**0.01** |
|**Imprinting + Aug** |**20.6** |**28.2** |**39.5** |**46.8** |**50.6**|

#### w/ FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand + FT (paper) |5.25 |13.41 |34.95| 54.33 |65.60|
|Imprinting + FT (paper)|18.67 |30.17| 46.08 |59.39 |68.77|
|AllClassJoint (paper) |3.89 |10.82 |33.00 |50.24 |64.88|
|**Rand + FT**|**3.8** |**11.6** |**32.9** |**51.7** |**66.8** |
|**Imprinting + Aug + FT** |**19.3** |**31.4** |**50.4** |**61.7** |**66.9** |
|**AllClassJoint** |**5.6** |**16.0** |**41.5** |**59.6** |**71.7** |
|**AllClassJoint** - ***Cosine Similarity*** |**6.6** |**19.5** |**47.8** |**65.6** |**76.7** |

### Average per-class recalls of all classes in CUB-200-2011
#### w/o FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand-noFT (paper) |37.36| 37.36| 37.36| 37.36 |37.36|
|Imprinting (paper)|44.75| 48.21| 52.95| 55.99 |57.47|
|Imprinting + Aug (paper) |44.60| 48.48| 52.78 |56.51| 57.84|
|**Rand-noFT**|**41.2** |**41.2** |**41.2** |**41.2** |**41.2** |
|**Imprinting + Aug** |**50.6** |**54.2** |**59.6** |**63.0** |**64.9**|

#### w/ FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand + FT (paper) |39.26 |43.36| 53.69| 63.17| 68.75|
|Imprinting + FT (paper)|45.81 |50.41 |59.15| 64.65| 68.73|
|AllClassJoint (paper) |38.02 |41.89| 52.24| 61.11| 68.31|
|**Rand + FT**|**42.4** |**45.9** |**56.2** |**65.1** |**72.5** |
|**Imprinting + Aug + FT** |**50.3** |**56.0** |**65.1** |**70.6** |**72.6** |
|**AllClassJoint** |**42.1** |**46.9** |**59.9** |**68.8** |**74.4** |
|**AllClassJoint** - ***Cosine Similarity*** |**44.3** |**50.6** |**64.5** |**73.7** |**78.6** |