from setuptools import setup, find_packages

setup(
    name = 'nas_bench_x11',
    version = '1.0',
    author = 'Shen Yan, Colin White',
    author_email = 'yanshen6@msu.edu, crwhite@cs.cmu.edu',
    description = 'benchmark for multi-fidelity NAS algorithms',
    license = 'Apache 2.0',
    keywords = ['AutoML', 'NAS', 'Deep Learning'],
    url = "https://github.com/automl/nas-bench-x11",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires = [
        'autograd>=1.3',
        'click',
        'Cython',
        'ConfigSpace==0.4.12',
        'ipython',
        'lightgbm>=2.3.1',
        'matplotlib',
        'numpy==1.20.3',
        'pandas',
        'pathvalidate',
        'Pillow>=7.1.2',
        'psutil',
        'scikit-image',
        'scikit-learn>=0.23.1',
        'scipy==1.2',
        'seaborn==0.11.1',
        'statsmodels',
        'torch==1.5.0',
        'torchvision==0.6.0',
        'tqdm',
        'xgboost',
        'tensorflow==1.15.4'
    ]
)
