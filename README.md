Pytorch implementation of [Low-Shot Learning with Imprinted Weights](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf). 

I started this repository as the clone of [this repository](https://github.com/YU1ut/imprinted-weights). But then I realized that
I had changed almost everything, therefore decided to make a separate repo on it's own. But still there may be some intact code snippets, for
which I give credit to @YU1ut.

## Important note
In the paper, Inception V1 is used as the feature extractor. However, since there is no pre-trained Inception V1 model in 
torchvision.models package, in this repo I use ResNet-50 as the feature extractor. Besides, somewhat surprisingly 
fine-tuning ResNet-50 with RMSProp (with the exact same parameters as in the paper) results in poor generalization.
Perhaps, this is yet another case where SGD with momentum is superior than RMSProp with momentum, in terms of generalization.

## Development environment
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
I will publish the results of experiments very soon.

## References
- [1]: H. Qi, M. Brown and D. Lowe. "Low-Shot Learning with Imprinted Weights", in CVPR, 2018.