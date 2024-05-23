# How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks

Repository for the replication of the paper [How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks](https://arxiv.org/abs/2305.15508), published in [The 40th Conference on Uncertainty in Artificial Intelligence](https://www.auai.org/uai2024/). 

## MaxLogit-pNorm

From a vector $\mathbf{z}$ of logits with $C$ classes, the MaxLogit-pNorm is defined as:

$$\text{MaxLogit-pNorm}(\mathbf{z}) := \max_{k} \frac{z_k-\mu(\mathbf{z})}{||\mathbf{z}-\mu(\mathbf{z})||_p}$$

where $`||\mathbf{z}||_p = \left(\sum\limits_{j=1}^{C} |z_j|^p\right)^{1/p}`$ is the $p$-norm of $\mathbf{z}$. 
It can be calculated with:

```python
import torch
import post_hoc
g = post_hoc.MaxLogit_pNorm(logits,p)
```
The optimization of `p` can be made with a grid search using
```python
p = post_hoc.optimize.p(logits,risk, metric = metric)
```
where `risk` is a tensor with the defined risk for each prediction and `metric` is the metric to be minimized. This procedure allows the fallback to the Maximum Softmax Probability (MSP), considered the baseline for confidence estimation, in cases where the MaxLogit-pNorm harms the confidence estimation. Alternatively, the optimization can be made directly with the calculation of the confidence:

```python
import torch
import post_hoc
g = post_hoc.MaxLogit_pNorm(logits,p = 'optimal', **kwargs_optimize)
```


## Paper Experiments

All conducted experiments are available in [experiments/notebooks](experiments/notebooks). Functions for all confidence estimators can be found in [utils/measures](utils/measures), while metrics, such as the Normalized Area Under the Risk Coverage Curve (NAURC), are in [utils/metrics](utils/metrics).

All models considered in the experiments can be found in [experiments/models](experiments/models). Cifar100 and OxfordPets models were trained using the receipe present in [experiments/train.py](experiments/train.py), while ImageNet pre-trained models are forked from [torchvision](https://github.com/pytorch/vision) and [timm](https://github.com/huggingface/pytorch-image-models) repositories.
 

## Citation

To cite this paper, please use

```bibtex
@misc{cattelan2024fix,
      title={How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks}, 
      author={Luís Felipe P. Cattelan and Danilo Silva},
      booktitle={The 40th Conference on Uncertainty in Artificial Intelligence},
      year={2024},
      url={https://openreview.net/forum?id=IJBWLRCvYX}
}
```
