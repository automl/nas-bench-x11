## Train your own surrogate benchmark
The [main readme](https://github.com/automl/nas-bench-x11) already shows how to use the pretrained surrogate benchmarks and run NAS algorithms. This readme is for training your own surrogate benchmark.

Download the NAS benchmark datasets corresponding to the surrogates you wish to train:
[NAS-Bench-101](https://storage.googleapis.com/nasbench/nasbench_full.tfrecord),
[NAS-Bench-301](https://figshare.com/ndownloader/files/25594868),
[NAS-Bench-NLP](https://drive.google.com/file/d/1DtrmuDODeV2w5kGcmcHcGj5JXf2qWg01), and
[NAS-Bench-201](https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ).

## Model Training
For NAS-Bench-NLP, put `nas-bench-x11/nasbenchnlp` into your PYTHONPATH\ 
(`export PYTHONPATH=$HOME/nas-bench-x11:$PYTHONPATH`)
```bash
# Supported search spaces: nb101, darts, nlp, nb201
cd $HOME/nas-bench-x11
python nas_bench_x11/fit_model.py --search_space darts --model svd_lgb
```
Trained models will be saved in ``checkpoints/``.
