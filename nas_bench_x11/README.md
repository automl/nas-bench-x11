## Train your own surrogate benchmark
The previous section already lets you run NAS algorithms using the pre-trained surrogate benchmarks. Now we give instructions in case you want to train your own surrogate benchmark.

Download the nas benchmark datasets (either with the terminal commands below,
or from their respective websites
([NAS-Bench-101](https://github.com/google-research/nasbench),
[NAS-Bench-301](https://github.com/automl/nasbench301), and
[NAS-Bench-NLP](https://github.com/fmsnew/nas-bench-nlp-release)).
```bash
cd $HOME/nas-bench-x11
# these files are 2GB, 300MB, and 21MB, respectively
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
wget https://figshare.com/ndownloader/files/25594868 -O nasbench301_models_v0.9.zip
wget https://drive.google.com/file/d/1DtrmuDODeV2w5kGcmcHcGj5JXf2qWg01
unzip nasbench301_full_data.zip
```

## Model Training
```
# Supported search spaces: nb101, darts, nlp, nb201
cd $HOME/nas-bench-x11
export PYTHONPATH=$HOME/nas-bench-x11:$PYTHONPATH
python nas_bench_x11/fit_model.py --search_space darts --model svd_lgb
```
Trained models will be saved in ``checkpoints/``.
