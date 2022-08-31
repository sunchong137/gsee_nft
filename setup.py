from setuptools import setup, find_packages
import os
import sys



def read_requirements(fname):
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements

NAME = 'gsee'
AUTHOR = 'Chong Sun'
DESCRIPTION = "A python package to perform ground state energy estimation with the algorithm proposed by Lin & Tong."
AUTHOR_EMAIL = 'chong.sun@zapatacomputing.com'
REQUIREMENTS = read_requirements('requirements.txt')

setup(
    name=NAME,
    description=DESCRIPTION,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=['*test*', '*examples*']),
    include_package_data=True,
)
