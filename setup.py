import re
import io
import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read()

with open('README.md') as f:
    readme = f.read()

# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()
    
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
    
setup(
      name='buildings_bench',
      version=find_version('buildings_bench', '__init__.py'),
      description='Large-scale pretraining and benchmarking for short-term load forecasting.',
      author='Patrick Emami',
      author_email='Patrick.Emami@nrel.gov',
      url="https://nrel.github.io/BuildingsBench/",
      long_description=readme,
      long_description_content_type='text/markdown',
      install_requires=requirements,
      packages=find_packages(include=['buildings_bench',
                                      'buildings_bench.data',
                                      'buildings_bench.evaluation',
                                      'buildings_bench.models'],
                             exclude=['test']),
      package_data={'buildings_bench': ['configs/*.toml']},
      license='BSD 3-Clause',
      python_requires='>=3.8',
      extras_require={
            'benchmark': ['transformers', 'wandb', 'properscoring', 'matplotlib', 'seaborn', 'jupyterlab']
      },
      keywords=['forecasting', 'energy', 'buildings', 'benchmark'],
      classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]
)
