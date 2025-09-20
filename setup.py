# setup.py
from setuptools import setup, find_packages

setup(
    name="multiperspective_narrative_memory_system",
    version="1.0.0",
    description="Multi-character memory system for narrative AI applications",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0", 
        "scikit-learn>=1.0.0",
        "networkx>=2.8.0",
        "typing-extensions>=4.0.0"
    ],
    python_requires=">=3.8",
    author="Sekai Memory Team",
    author_email="contact@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)