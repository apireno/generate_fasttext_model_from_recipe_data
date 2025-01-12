from setuptools import setup, find_packages


setup(
    name="generate_fasttext_model_from_recipe_data",
    version="0.1.0",
    packages=find_packages(),  # Use find_packages() to automatically find packages
    install_requires=[
        "torch"
    ]
)