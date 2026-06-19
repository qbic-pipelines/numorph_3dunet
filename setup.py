from setuptools import setup, find_packages

setup(
    name="numorph_3dunet",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    author="Carolin Schwitalla",
    author_email="carolin.schwitalla@uni-tuebingen.de",
    description="Numorph segmentation of cell nuclei using a 3dunet.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=["3dunet", "segmentation", "microscopy", "lightsheet"],
    url="https://github.com/qbic-pipelines/numorph_3dunet",
    entry_points={
        'console_scripts': [
            'numorph_3dunet.predict=numorph_3dunet.cli:predict',
        ],
    },
)