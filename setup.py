from setuptools import setup

setup(
    name = "fieldgen",
    version = "0.1.0",    
    description = "Generating simple fields for cosmological applications.",
    url = "https://github.com/Saladino93/fieldgen",
    author = "Omar Darwish",
    author_email = "o.darwish@protonmail.com",
    packages = ["fieldgen"],
    install_requires = ["numpy", "plancklens", "healpy", "lenspyx", "numba"]
)