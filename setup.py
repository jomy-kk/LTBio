from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name='LongTermBiosignals',
    version='1.0.0',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    url='https://github.com/jomy-kk/IT-LongTermBiosignals',
    license='',
    author='João Saraiva, Mariana Abreu',
    author_email='joaomiguelsaraiva@tecnico.ulisboa.pt',
    description='Python library for easy managing and processing of large Long-Term Biosignals.',
    long_description = long_description,
    long_description_content_type = "text/markdown",

    python_requires = ">=3.10.4",

)
