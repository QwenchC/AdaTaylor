"""
AdaTaylor安装配置
"""

from setuptools import setup, find_packages

setup(
    name="adataylor",
    version="0.1.0",
    description="泰勒展开自适应逼近器",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "sympy>=1.8.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "plotly>=5.4.0",
        "dash>=2.0.0",
        "pywavelets>=1.1.0",
        "pandas>=1.3.0",
        "dash-bootstrap-components>=1.0.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "adataylor=main:run_cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)