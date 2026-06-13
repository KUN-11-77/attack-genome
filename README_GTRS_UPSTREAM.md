# GTRS: Generalized Trajectory Scoring for End-to-end Multi-modal Planning

![](./assets/gtrs.png)

### [arXiv](https://arxiv.org/abs/2506.06664) | [Blog](https://blogs.nvidia.com/blog/auto-research-cvpr-2025/?ncid=so-nvsh-677066) | [Challenge](https://opendrivelab.com/challenge2025/#navsim-e2e-driving)

This is the official repository of GTRS, an end-to-end planning framework featuring a robust trajectory scorer for evaluating diverse trajectory candidates.

## News
🏆GTRS wins the [E2E Driving Track](https://opendrivelab.com/challenge2025/#navsim-e2e-driving) at [CVPR25 Autonomous Grand Challenge](https://opendrivelab.com/challenge2025/).
![](assets/e2e.png)

## Getting Started
Please refer to [doc/install.md](doc/install.md) for downloading the dataset and installing the devkit.

## Project documentation layout
This repository currently uses two documentation layers:
- [`doc/`](doc/README.md): operational docs for experiment status, runbooks, setup lookups, and code navigation
- [`docs/`](docs/): research planning docs for method design, experiment planning, and paper writing

If you are starting a new experiment session, the shortest path is:
- [`doc/ops/EXPERIMENT_STATUS.md`](doc/ops/EXPERIMENT_STATUS.md)
- the active experiment’s `SUMMARY.md` / `STATUS.md`
- [`doc/README.md`](doc/README.md) only when you need operational lookup docs

## Model Checkpoints
| Model                                                                                     |                                   Resolution                                    | Vocab. Size  |                                    Backbone                                    | EPDMS | Checkpoint                                   |
|:------------------------------------------------------------------------------------------|:----------:|:------------:|:------------------------------------------------------------------------------:|:-----:|:----------:|
| [Diffusion Policy](navsim/planning/script/config/common/agent/gtrs_diffusion_policy.yaml) |512x2048|      -       |[V2-99](https://drive.google.com/file/d/1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A/view) | 25.6  |     [Link](https://huggingface.co/Zzxxxxxxxx/gtrs/blob/main/gtrs_dp.ckpt)     |
| [GTRS-Dense](navsim/planning/script/config/common/agent/gtrs_dense_vov.yaml)              |512x2048|    16384     |[V2-99](https://drive.google.com/file/d/1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A/view) | 41.7  | [Link](https://huggingface.co/Zzxxxxxxxx/gtrs/blob/main/gtrs_dense_vov.ckpt)  |
| [GTRS-Aug](navsim/planning/script/config/common/agent/gtrs_aug_vov.yaml)                  |512x2048|     8192     |[V2-99](https://drive.google.com/file/d/1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A/view) | 42.1  |  [Link](https://huggingface.co/Zzxxxxxxxx/gtrs/blob/main/gtrs_aug_vov.ckpt)   |
| [Hydra-MDP (img only)](navsim/planning/script/config/common/agent/hydra_mdp_vov.yaml)     |512x2048|    16384     |[V2-99](https://drive.google.com/file/d/1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A/view) | 37.5  |  [Link](https://huggingface.co/Zzxxxxxxxx/gtrs/blob/main/hydra_mdp_vov.ckpt)   |

## Training and Inference
Please refer to [doc/gtrs_training.md](doc/gtrs_training.md) and [doc/gtrs_inference.md](doc/gtrs_inference.md) respectively.

## Citation
Consider citing our work if you find it useful. Thanks!
```
@article{li2025generalized,
  title={Generalized Trajectory Scoring for End-to-end Multimodal Planning},
  author={Li, Zhenxin and Yao, Wenhao and Wang, Zi and Sun, Xinglong and Chen, Joshua and Chang, Nadine and Shen, Maying and Wu, Zuxuan and Lan, Shiyi and Alvarez, Jose M},
  journal={arXiv preprint arXiv:2506.06664},
  year={2025}
}
```
## Acknowledgement
Many thanks to the following great open-source repositories:
+ [NAVSIM](https://github.com/autonomousvision/navsim)
+ [VAD](https://github.com/hustvl/VAD)
+ [Transfuser](https://github.com/autonomousvision/transfuser)
