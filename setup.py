import os

from setuptools import find_packages, setup


def get_version() -> str:
    version = "unknown"
    with open("zefir_analytics/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                _, _, version = line.replace('"', "").split()
                break
    return version


def load_req(r_file: str) -> list[str]:
    with open(os.path.join(os.getcwd(), r_file)) as f:
        return [
            r for r in (line.split("#", 1)[0].strip() for line in f.readlines()) if r
        ]


setup(
    name="zefir_analytics",
    packages=find_packages(".", exclude=["*tests*"]),
    version=get_version(),
    install_requires=load_req("requirements.txt"),
    setup_requires="",
    python_requires=">=3.11",
    author="IDEA",
    author_email="office@idea.edu.pl",
    url="idea.edu.pl",
)
