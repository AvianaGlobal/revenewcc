from setuptools import setup, find_packages
import os

setup(
    name='revenewCC',
    version='0.2',
    packages=find_packages(),
    include_package_data=True,
    install_dir=os.getcwd(),
    install_requires=[
        'pandas',
        'sqlalchemy',
        'fuzzywuzzy',
        'xlsxwriter',
        'pyodbc',
        'wxpython',
    ],
    entry_points={
        'console_scripts': [
            'ranking=revenewCC.ranking:main'
        ]
    }
)
