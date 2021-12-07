# NAS-Bench-NLP

This is the nas-bench-x11 fork of [NAS-Bench-NLP](https://github.com/fmsnew/nas-bench-nlp-release).
Changes were made to [`main_one_model_train.py`](nas-bench-x11/nasbenchnlp/main_one_model_train.py) and a few other places, so that if needed, the NAS-Bench-NLP11 surrogate will train the first three epochs of architectures when given queries.