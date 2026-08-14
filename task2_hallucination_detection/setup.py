from pathlib import Path
from setuptools import setup, find_namespace_packages


def parse_requirements():
    req = Path("requirements.txt")
    if not req.exists():
        return []

    return [
        line.strip()
        for line in req.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="task2-hallucination-detection",
    version="0.1.0",
    packages=find_namespace_packages(include=["src*"]),
    include_package_data=True,
    # install_requires=parse_requirements(),
    python_requires=">=3.10",

)