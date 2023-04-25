# Weakly-supervised Biomechanically-constrained CT/MRI Registration of the Spine

This is the official Pytorch implementation of my paper:
["Weakly-supervised Biomechanically-constrained CT/MRI Registration of the Spine" (MICCAI 2022)](https://arxiv.org/abs/2205.07568)

---

## Guidelines

1. Check [Registry](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html)
   and [Config](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html)
   of [MMEngine](https://mmengine.readthedocs.io/en/latest/).
2. Check the Config files
   for [dataset](/configs/_base_/dataset.py), [network](/configs/_base_/voxelmorph_half_res.py), [training](/configs/voxelmorph_half_res_mind_oc_pc.py)
   and [inference](/configs/inference_voxelmorph_half_res.py).
3. Construct the data files following the structure shown below. Change the parameters in
   the [dataset](/configs/_base_/dataset.py) Config.
4. Train the model with only [image similarity](/models/losses/mind.py)
   and [smoothness regularization](/models/losses/diffusion_regularizer.py) (e.g. 400 epochs).
5. Train the model with additionally rigidity losses to finetune. Check
   the [supplementary materiel](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_22#Sec5) for weights of
   different rigidity losses.
6. Inference on test data.

---

## Prerequisites

- `Python 3.8.8`
- `Numpy 1.23.4`
- `Scipy 1.9.3`
- `Pytorch 1.13.0`
- `MONAI 1.0.1`
- `MMCV 1.7.0`
- `MMENGINE 0.5.0`
- `WANDB 0.13.10`
- `SimpleITK 2.0.2`

---

## Data structure

```bash
├──data_dir
│   ├──train
│   │   ├──001
│   │   │   ├──ct.nii.gz
│   │   │   ├──ct_mask.nii.gz
│   │   │   ├──mr_t1.nii.gz
│   │   │   ├──mr_mask.nii.gz
│   │   ├──002
...
│   ├──val
│   │   ├──...
...
│   ├──test
│   │   ├──...
```

---

## Dataset

Since the dataset used in the paper is confidential and belongs to the hospital, we cannot share it.
If you are also doing spine registration and trying to construct your own dataset, it's recommended to use the free
spine CT vertebra segmentation webtool [Anduin](https://anduin.bonescreen.de/).

---
## Pretrained Model

The pretrained model weights are also unable to share due to the confidentiality of the dataset.

---

## Logging

I use [Weight&Biases](https://wandb.ai/site) for logging in training. It's convenient to use (view the plot without ssh
to port) and free. If you prefer to use tensorboard, you could also construct your own logger and add it
to [run_epoch](/scripts/epoch_run.py).

---

## Baseline method

Check [Elastix](https://elastix.lumc.nl/modelzoo/par0004/).

---

## Citation

If you find this repository useful in your research, please consider to cite use in your work by:

```
@inproceedings{jian2022weakly,
  title={Weakly-supervised Biomechanically-constrained CT/MRI Registration of the Spine},
  author={Jian, Bailiang and Azampour, Mohammad Farid and De Benetti, Francesca and Oberreuter, Johannes and Bukas, Christina and Gersing, Alexandra S and Foreman, Sarah C and Dietrich, Anna-Sophia and Rischewski, Jon and Kirschke, Jan S and others},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part VI},
  pages={227--236},
  year={2022},
  organization={Springer}
}
```

---

## Acknowledgement

Many thanks to the following repositories for providing helpful resources to my work:

- [Voxelmorph](https://github.com/voxelmorph/voxelmorph)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [MMCV](https://github.com/open-mmlab/mmcv)

---

## License & Copyright

© Bailiang Jian
Licensed under the [MIT Licensce](LICENSCE)
