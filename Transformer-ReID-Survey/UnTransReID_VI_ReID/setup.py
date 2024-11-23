from setuptools import setup, find_packages


setup(name='ClusterContrast',
      version='1.0.0',
      description='Cluster Contrast for Unsupervised Person Re-Identification',
      author='GuangYuan wang',
      author_email='yixuan.wgy@alibaba-inc.com',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])
