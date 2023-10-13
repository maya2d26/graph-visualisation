from setuptools import find_packages, setup  # type: ignore

DESCRIPTION = "Graph Visualisation with Deep Learning"

setup(
    name="dl-graphviz",
    version="0.0",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.py"]},
    install_requires=[],
)