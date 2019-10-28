#!/usr/bin/env python
# coding: utf8

"""  Distribution script. """

import sys

from os import path
from setuptools import setup

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

# Default project values.
project_name = 'spleeter'
project_version = '1.4.1'
device_target = 'cpu'
tensorflow_dependency = 'tensorflow'
tensorflow_version = '1.14.0'
here = path.abspath(path.dirname(__file__))
readme_path = path.join(here, 'README.md')
with open(readme_path, 'r') as stream:
    readme = stream.read()

# Check if GPU target is specified.
if '--target' in sys.argv:
    target_index = sys.argv.index('--target') + 1
    target = sys.argv[target_index].lower()
    sys.argv.remove('--target')
    sys.argv.pop(target_index)

# GPU target compatibility check.
if device_target == 'gpu':
    project_name = '{}-gpu'.format(project_name)
    tensorflow_dependency = 'tensorflow-gpu'

# Package setup entrypoint.
setup(
    name=project_name,
    version=project_version,
    description='''
        The Deezer source separation library with
        pretrained models based on tensorflow.
    ''',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Deezer Research',
    author_email='research@deezer.com',
    url='https://github.com/deezer/spleeter',
    license='MIT License',
    packages=[
        'spleeter',
        'spleeter.commands',
        'spleeter.model',
        'spleeter.model.functions',
        'spleeter.model.provider',
        'spleeter.resources',
        'spleeter.utils',
        'spleeter.utils.audio',
    ],
    package_data={'spleeter.resources': ['*.json']},
    python_requires='>=3.6, <3.8',
    include_package_data=True,
    install_requires=[
        'importlib_resources ; python_version<"3.7"',
        'musdb==0.3.1',
        'museval==0.3.0',
        'norbert==0.2.1',
        'pandas==0.25.1',
        'requests',
        '{}=={}'.format(tensorflow_dependency, tensorflow_version),
    ],
    entry_points={
        'console_scripts': ['spleeter=spleeter.__main__:entrypoint']
    },
    classifiers=[
        'Environment :: Console',
        'Environment :: MacOS X',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Artistic Software',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Conversion',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities']
)
