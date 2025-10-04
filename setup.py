from setuptools import setup, find_packages

setup(
    name='nlp_model_merging',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "datasets",
        "numpy",
        "tqdm",
        "pyyaml",
    ],
    entry_points={
        'console_scripts': [
            'nlpmerge=src.main:main',
        ],
    },
)
