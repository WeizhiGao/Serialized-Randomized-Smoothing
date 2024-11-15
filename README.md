# Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing
The official code of the NeurIPS 2024 paper: [Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing](https://arxiv.org/abs/2411.00899) by [Weizhi Gao](), [Zhichao Hou](https://weizhigao.github.io/), [Han Xu](https://sites.google.com/view/han-xu-123/home), and [Xiaorui Liu](https://sites.google.com/ncsu.edu/xiaorui/).

Serialized Randomized Smoothing is the first work studying randomized smoothing for Deep Equilibrium Models and accelerte it. We make use of randomized smoothing to certify DEQ, showing SOTA certified robustness. Our serialized randomized smoothing can accelerate randomized smoothing on DEQ up to 7x almost without sacrificing the certified accuracy. It is also the first work that extend the certification robustness of DEQ to ImageNet. 

The results on CIFAR-10.
| Model \ Radius | 0.0  | 0.25 | 0.5  | 0.75 | 1.0  | 1.25 | 1.5  | Time (s)    |
|----------------|------|------|------|------|------|------|------|-------------|
| MDEQ-1A        | 28%  | 19%  | 13%  | 8%   | 5%   | 3%   | 1%   | 1.06        |
| MDEQ-5A        | 50%  | 41%  | 32%  | 21%  | 15%  | 10%  | 6%   | 2.59        |
| MDEQ-30A       | **67%** | **55%** | 45%  | 33%  | 23%  | 16%  | 12%  | 12.89       |
| SRS-MDEQ-1N    | 61%  | 52%  | 44%  | 31%  | 22%  | 15%  | 11%  | 1.02 (**13×**) |
| SRS-MDEQ-1A    | 63%  | 53%  | **45%** | 32%  | 22%  | 16%  | **12%** | 1.79 (**7×**) |
| SRS-MDEQ-3A    | **66%** | 54%  | **45%** | **33%** | 23%  | 16%  | 11%  | 2.55 (**5×**) |

