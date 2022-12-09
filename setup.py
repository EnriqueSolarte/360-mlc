from setuptools import setup, find_packages

setup(
    name='360_mlc',
    version='0.1',
    packages=find_packages(),
    author = "Enrique Solarte and Justin Wu (吳京軒)",
    author_email = "enrique.solarte.pardo@gmail.com",
    description = ("Multi-view layout consistency MLC - NeurIPS 2022"),
    python_requires='>=3.7',
    license = "BSD",
)
