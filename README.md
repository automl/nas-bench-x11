# NAS-Bench-x11

We present a method using singular value decomposition and noise modeling to create surrogate benchmarks, NAS-Bench-111, NAS-Bench-311, and NAS-Bench-NLP11, that output the full training information for each architecture, rather than just the final validation accuracy.

# Data Preparation
Download each nasbenchmarks and save them in ``data/``.

# Model Training
```
# Supported search spaces: nb101, darts, nlp, nb201

python fit_model.py --search_space darts --model svd_lgb
```
models will be saved in ``experiments/``.


# Run NAS experiments
```
# Supported optimizers: (rs re ls bananas)-{svr, lce}, hb, bohb 

bash naslib/benchmarks/nas/run_nb311.sh 
bash naslib/benchmarks/nas/run_nb201.sh 
bash naslib/benchmarks/nas/run_nb201_cifar100.sh 
bash naslib/benchmarks/nas/run_nb201_imagenet16-200.sh
bash naslib/benchmarks/nas/run_nb211.sh 
bash naslib/benchmarks/nas/run_nbnlp.sh 
```
results will be saved in ``results/``.

## Citation
```bibtex
@inproceedings{yan2021bench,
  title={NAS-Bench-x11 and the Power of Learning Curves},
  author={Yan, Shen and White, Colin and Savani, Yash and Hutter, Frank},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
