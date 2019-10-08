from setuptools import setup, find_packages
import os

setup(
    name='revenewCC',
    version='0.3',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'sqlalchemy',
        'fuzzywuzzy',
        'xlsxwriter',
        'pyodbc',
        'wxpython',
        'python-Levenshtein',
    ],
    entry_points={
        'gui_scripts': [
            'ranking=revenewCC.ranking:main'
        ]
    }
)
