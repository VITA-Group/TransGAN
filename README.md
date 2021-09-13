# TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up
Code used for [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074). 

## Implementation
- [ ] checkpoint gradient using torch.utils.checkpoint
- [ ] 16bit precision training
- [x] Distributed Training (Faster!)
- [x] IS/FID Evaluation
- [x] Gradient Accumulation

## Main Pipeline
![Main Pipeline](assets/TransGAN_1.png)

## Representative Visual Results
![Cifar Visual Results](assets/cifar_visual.jpg)
![Visual Results](assets/teaser_examples.jpg)


README waits for updated
## Acknowledgement
Codebase from [AutoGAN](https://github.com/VITA-Group/AutoGAN), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## Citation
if you find this repo is helpful, please cite
```
@article{jiang2021transgan,
  title={TransGAN: Two Transformers Can Make One Strong GAN},
  author={Jiang, Yifan and Chang, Shiyu and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2102.07074},
  year={2021}
}
```
