# !/usr/bin/env python3

import os
from setuptools import setup, find_packages

package = "crunch_synth"

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, package, '__version__.py')) as f:
    exec(f.read(), about)

with open("requirements.txt") as fd:
    requirements = fd.read().splitlines()

with open('requirements.test.txt') as fd:
    test_requirements = fd.read().splitlines()

with open('README.md') as fd:
    readme = fd.read()

setup(
    name=about['__title__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    version=about['__version__'],
    author=", ".join(about["__author__"]),
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=find_packages(
        exclude=["tests"],
    ),
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        'test': test_requirements,
    },
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.12',
    ],
    keywords=[
        'crunchdao',
        'crunch',
        'crunch-synth',
    ],
)
