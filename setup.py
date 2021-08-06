from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv', 'torch-geometric', 'transformers', 'google-cloud-storage']

setup(
    name='ChineseKG2TextTrainer',
    version='0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='ChineseKG2Text training application.'
)