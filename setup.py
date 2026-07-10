from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

PROJECT_ROOT = Path(__file__).parent


def read_requirements(path):
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if requirement.startswith(("-e ", "--editable ")):
            continue
        if requirement.startswith(("-r ", "--requirement ")):
            nested_path = requirement.split(maxsplit=1)[1]
            requirements.extend(read_requirements(path.parent / nested_path))
            continue
        requirements.append(requirement)
    return requirements


requirements = read_requirements(PROJECT_ROOT / "requirements.txt")

setup(
    name='LongTermBiosignals',
    version='2.0.2',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=requirements,
    url='https://github.com/jomy-kk/LTBio',
    license='',
    author='João Saraiva, Mariana Abreu',
    author_email='joaomiguelsaraiva@tecnico.ulisboa.pt',
    description='Python library for easy managing and processing of large Long-Term Biosignals.',
    long_description = long_description,
    long_description_content_type = "text/markdown",

    python_requires = ">=3.10.4",

)
