# NAS-Bench-x11

[NAS-Bench-x11 and the Power of Learning Curves](https://arxiv.org/abs/2111.03602)\
Shen Yan, Colin White, Yash Savani, Frank Hutter.\
NeurIPS 2021.

## Surrogate NAS benchmarks for multi-fidelity algorithms
We present a method to create surrogate neural architecture search (NAS) benchmarks, `NAS-Bench-111`, `NAS-Bench-311`, and `NAS-Bench-NLP11`, that output the **full training information** for each architecture, rather than just the final validation accuracy. This makes it possible to benchmark multi-fidelity techniques such as successive halving and learning curve extrapolation (LCE). Then we present a framework for converting popular single-fidelity algorithms into LCE-based algorithms.

<p align="center">
<img src="img/nbx11.png" alt="nas-bench-x11" width="70%">
</p>

## Installation
Clone this repository and install its requirements.
```bash
git clone https://github.com/automl/nas-bench-x11
cd nas-bench-x11
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install -e .
```

## Download pre-trained models
Download [surrogate models](https://figshare.com/articles/dataset/pretrained_models_zip/17131955) and save them in ``checkpoints/``. The current model used in paper is version v0.5. We will continue to improve the surrogate model (i.e. sliding window approach).

## Using the API
TODO: simple example to show how to query the api, similar to this: https://github.com/google-research/nasbench#using-the-dataset

Now you are ready to benchmark multi-fidelity NAS algorithms on a CPU.

## Run NAS experiments from our paper
Download the nas-bench-301 runtime model [lgb_runtime_v1.0](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510) in ``nasbench301/nb_models``.
```
# Supported optimizers: (rs re ls bananas)-{svr, lce}, hb, bohb 

bash naslib/benchmarks/nas/run_nb311.sh 
bash naslib/benchmarks/nas/run_nb201.sh 
bash naslib/benchmarks/nas/run_nb201_cifar100.sh 
bash naslib/benchmarks/nas/run_nb201_imagenet16-200.sh
bash naslib/benchmarks/nas/run_nb111.sh 
bash naslib/benchmarks/nas/run_nbnlp.sh 
```
Results will be saved in ``results/``.



## Citation
```bibtex
@inproceedings{yan2021bench,
  title={NAS-Bench-x11 and the Power of Learning Curves},
  author={Yan, Shen and White, Colin and Savani, Yash and Hutter, Frank},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
