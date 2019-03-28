from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='rdf',
      version='0.1',
      description='Matrix factorization model with regularization differentiation function',
      long_description='''This package implements several matrix factorization model with regularization differentiation function, including SVD, SVD++, and NMF''',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='matrix factorization, regularization, SVD, SVD++, NMF',
      url='https://github.com/ncu-dart/rdf',
      author='Hung-Hsuan Chen',
      author_email='hhchen@ncu.edu.tw',
      license='MIT',
      packages=['rdf'],
      install_requires=[
            'numpy',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=[
            'bin/svd-train.py',
            'bin/svdpp-train.py',
            'bin/nmf-train.py',
            'bin/rdfsvd-train.py',
            'bin/rdfsvdpp-train.py',
            'bin/rdfnmf-train.py',
            'bin/rdf-test.py',
      ],
)
