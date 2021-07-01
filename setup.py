from distutils.core import setup

setup(name='neegpred',
      packages = ['neegpred'],
      version='0.2',
      description='predicit liking and BDM results from EEG signals',
      url='https://github.com/KeshetTadmor/neegpred', 
      download_url = 'https://github.com/KeshetTadmor/neegpred/archive/refs/tags/0.1.tar.gz', #FILL IN LATER
      author='Adi Cohen, Nimrod Leberstein & Keshet Tadmor',
      author_email='nimrodlbr@gmail.com',
      keywords = ['EEG', 'ML', 'neuro economy'],
      license='GNU General Public License v3.0',
      long_description='This package is intended to be used by Prof. Dino Levy, It offers a platform to train\n'
                       'models that can predict the effect of a given commercial by EEG signal processing'
                       'and Machine learning'

      install_requires=['pandas','numpy','mat73','matplotlib','scipy','math','tqdm','seaborn','sklearn','shap','pytest'], 
  

      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.7',
        ],
)
