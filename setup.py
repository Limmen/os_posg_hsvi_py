from setuptools import setup

setup(name='os_posg_hsvi_py',
      version='0.0.1',
      install_requires=['numpy', 'pulp'],
      author='Kim Hammar',
      author_email='hammar.kim@gmail.com',
      description='Implementation of HSVI for OS-POSGs in python',
      license='Creative Commons Attribution-ShareAlike 4.0 International',
      keywords='Stochastic-games OS-POSG partially-observed-stochastic-games hsvi',
      url='https://github.com/Limmen/os_posg_hsvi_py',
      download_url='https://github.com/Limmen/os_posg_hsvi_py/archive/0.0.1.tar.gz',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'Programming Language :: Python :: 3.8'
      ]
      )