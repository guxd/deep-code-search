#nsml: nsml/ml:cuda10.1-cudnn7-pytorch1.3keras2.3
from distutils.core import setup
setup(
    author='Xiaodong Gu',
    author_email='xiaodong.gu@navercorp.com',
    name='DeepCS',
    version='0.1',
    description='Hyperparameter tuning',
    install_requires = [
        'numpy',
        'protobuf',
        'six',
        'tables',
        'tensorboardX',
        'tqdm',
        'transformers',
    ]
)
