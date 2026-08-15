# [CVPR2025] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective

> **Replication branch:** This branch adds an unofficial, auditable reproduction
> scaffold for [FT-NCFM (arXiv:2511.16233)](https://arxiv.org/abs/2511.16233) on
> top of the official NCFM code. It is not code released by either paper's authors.

## FT-NCFM reproduction

Start with the [replication roadmap](docs/REPLICATION_ROADMAP.md), then see the
[paper-to-code map](docs/PAPER_TO_CODE.md) and
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md). The first executable gate is
a 5% MNIST-based VLA-shaped proxy that tests LiSSA influence scoring, visual-only
counterexamples, influence-weighted characteristic-function matching, and min-max
loss convergence. It is a plumbing test, not a robotics benchmark result.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ft.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ft_ncfm.experiment `
  --config configs/ft_ncfm/minivla_5pct.yaml
```

Linux/CUDA users can build the checked-in container instead. Every experiment
writes a manifest, raw metric stream, influence artifact, coreset, and summary
under the ignored `artifacts/` directory.

Official PyTorch implementation of the paper ["Dataset Distillation with Neural Characteristic Function"](https://arxiv.org/abs/2502.20653) (NCFM) in CVPR 2025.


## :fire: News

- [2025/03/02] The code of our paper has been released.  
- [2025/02/27] Our NCFM paper has been accepted to CVPR 2025 (Rating: 555). Thanks!  


## :rocket: Pipeline

Here's an overview of the process behind our **Neural Characteristic Function Matching (NCFM)** method:

![Figure 1](./asset/figure1.png?raw=true)




## 🛠️ Getting Started

To get started with NCFM, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/gszfwsb/NCFM.git
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```
3. Pretrain the models yourself, or download the **pretrained_models** from [huggingface](https://huggingface.co/maomaocun/NCFM). 
```sh
cd pretrain
torchrun --nproc_per_node={n_gpus} --nnodes=1 pretrain_script.py --gpu={gpu_ids} --config_path=../config/{ipc}/{dataset}.yaml

```

4. Condense
```sh
cd condense 
torchrun --nproc_per_node={n_gpus} --nnodes=1 condense_script.py --gpu={gpu_ids} --ipc={ipc} --config_path=../config/{ipc}/{dataset}.yaml

```
5. Evaluation or or download the **condensed dataset** from [huggingface](https://huggingface.co/maomaocun/NCFM)
```sh
cd evaluation 
torchrun --nproc_per_node={n_gpus} --nnodes=1 evaluation_script.py --gpu={gpu_ids} --ipc={ipc} --config_path=../config/{ipc}/{dataset}.yaml --load_path={distilled_dataset.pt}
```

### :blue_book: Example Usage

1. CIFAR-10

```sh
#ipc50
cd condense
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=50 --config_path=../config/ipc50/cifar10.yaml
```

2. CIFAR-100

```sh
#ipc10
cd condense
torchrun --nproc_per_node=8 --nnodes=1 --master_port=34153 condense_script.py --gpu="0,1,2,3,4,5,6,7" --ipc=10 --config_path=../config/ipc10/cifar100.yaml
```



## :postbox: Contact
If you have any questions, please contact [Shaobo Wang](https://gszfwsb.github.io/)(`shaobowang1009@sjtu.edu.cn`).

## :pushpin: Citation
If you find NCFM useful for your research and applications, please cite using this BibTeX:

```bibtex
@inproceedings{wang2025NCFM,
      title={Dataset Distillation with Neural Characteristic Function: A Minmax Perspective}, 
      author={Shaobo Wang and Yicun Yang and Zhiyuan Liu and Chenghao Sun and Xuming Hu and Conghui He and Linfeng Zhang},
 booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2025}
}
```

## Acknowledgement
We sincerely thank the developers of the following projects for their valuable contributions and inspiration: [MTT](https://github.com/GeorgeCazenavette/mtt-distillation), [DATM](https://github.com/NUS-HPC-AI-Lab/DATM), [DC/DM](https://github.com/VICO-UoE/DatasetCondensation), [IDC](https://github.com/snu-mllab/Efficient-Dataset-Condensation), [SRe2L](https://github.com/VILA-Lab/SRe2L), [RDED](https://github.com/LINs-lab/RDED), [DANCE](https://github.com/Hansong-Zhang/DANCE). We draw inspiration from these fantastic projects!
