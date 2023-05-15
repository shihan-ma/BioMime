from setuptools import setup, find_packages

from BioMime import __version__

setup(
    name='BioMime',
    version=__version__,
    author='Shihan Ma',
    author_email='mmasss1205@gmail.com',
    description='BioMime is a package for simulating MUAPs from changes of physiological parameters',

    url='https://github.com/shihan-ma/BioMime',

    packages=find_packages(
        where="BioMime"
    )
)
