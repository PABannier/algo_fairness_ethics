from setuptools import setup, find_packages

setup(
    name="algo_fairness",
    install_requires=[
        "numpy>=1.12",
        "scikit-learn>=0.23",
        "lightgbm",
        "statsmodels"
    ],
    packages=find_packages()
)
