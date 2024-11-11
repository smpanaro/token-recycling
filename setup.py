from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="token-recycling",
    version="0.0.01",
    author="Stephen Panaro",
    description="Implementation of Token Recycling for LLM inference acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smpanaro/token-recycling",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
)
