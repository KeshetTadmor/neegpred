from distutils.core import setup

setup(name='neegpred',
      packages = ['neegpred'],
      version='0.1',
      description='predicit liking and BDM results from EEG signals',
      url='https://github.com/KeshetTadmor/neegpred', 
      download_url = 'https://github.com/KeshetTadmor/neegpred/archive/refs/tags/0.1.tar.gz', #FILL IN LATER
      author='Adi Cohen, Nimrod Leberstein & Keshet Tadmor',
      author_email='nimrodlbr@gmail.com',
      keywords = ['EEG', 'ML', 'neuro economy'],
      license='GNU General Public License v3.0',

      install_requires=['pandas','numpy','mat73','matplotlib','scipy','math','tqdm','seaborn','sklearn','shap'], 
  

      classifiers=[
        'Development Status :: 3 - Alpha',      # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Intended Audience :: researchers',      
        'Topic :: data science :: EEG analysis',
        'License :: OSI Approved :: GNU General Public License v3.0'', # Your License Here  
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.4',
        ],
)
