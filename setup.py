from setuptools import setup, find_packages

setup(
    name='gen_model_playground',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    insall_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib',
        'tqdm',
        'lightning'
        'lightning-bolts',
        'torchcfm',
        'torchdyn',
        'nflows',
        'scipy',
        'sklearn',
        'wandb',
    ]


)