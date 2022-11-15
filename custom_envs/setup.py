from setuptools import setup

setup(
    name='custom_envs',  # name of the package
    version='0.0.1',  # version of this release
    install_requires=['gym==0.26.2', 'numpy'], # specifies the minimal list of libraries required to run the package correctly
    packages=['custom_cartpole', 'grid_v0']
)
