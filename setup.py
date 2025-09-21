# setup.py
from setuptools import setup, find_packages

setup(
    name="multiperspective-narrative-memory-system",
    version="1.0.0",
    description="Multi-character memory system for narrative AI applications",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0", 
        "scikit-learn>=1.0.0",
        "networkx>=2.8.0",
        "typing-extensions>=4.0.0",
        "flask>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0"
    ],
    python_requires=">=3.8",
    author="Narrative Memory Team",
    author_email="contact@example.com",
    license="MIT",
)