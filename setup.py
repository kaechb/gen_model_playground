from setuptools import setup, find_packages

setup(
    name='gen_model_playground',
    version='0.0.1',
    package_dir={},
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib',
        'tqdm',
        'lightning',
        'lightning-bolts',
        'torchcfm',
        'torchdyn',
        'nflows',
        'scipy',
        'scikit-learn',
        'wandb',
    ]


)