
from setuptools import setup
from setuptools import find_packages

long_description = '''
DeepLinkPrediction is a package that facilitates the usage of
 a Link Prediction model based on Deep Learning architecture. This package is distributed under the MIT license.
'''

setup(name='DeepLinkPrediction',
      version='0.0.1',
      description='DLP analysis and interaction network manipulation',
      long_description=long_description,
      author='Pieter-Paul Strybol, Maarten Larmuseau',
      author_email='pieterpaul.strybol@ugent.be, maarten.larmuseau@ugent.be',
      url='https://github.ugent.be/PSTRYBOL/DepMap_DeepLinkPrediction',
      license='MIT',
      install_requires=['numpy==1.16.5',
                        'pandas>=0.24.2',
                        'scipy==1.6.1',
                        'tensorflow==1.13.1',
                        'scikit-learn==0.20.3',
                        'keras==2.2.4',
                        'matplotlib>=3.0.3',
                        'lifelines>=0.21.2',
                        'networkx==2.5.1',
                        'umap',
                        'statsmodels>=0.10.1',
                        'gensim==3.8.1',
                        'seaborn>=0.10.1',
                        'venn==0.1.3',
                        'ndex2',
                        'requests',
                        'tables'],

      classifiers=[
         "Development Status :: 3 - Alpha",

      ],
packages=find_packages())
