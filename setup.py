from setuptools import setup, find_packages

setup(
    name='revenewCC',
    version='0.5',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fuzzywuzzy',
        'pandas',
        'sqlalchemy',
        'pillow',
        'pyodbc',
        'python-Levenshtein',
        'tqdm',
        'wxpython',
        'xlsxwriter',
    ],
    entry_points={
        'console_scripts': [
            'ranking=revenewCC.ranking:main'
        ]
    }
)
