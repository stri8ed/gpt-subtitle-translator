from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gpt_subtitle_translator',
    version='0.1.0',
    description='Translate subtitles using GPT-like models',
    author='stri8ed',
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        'gpt_subtitle_translator': ['prompt.txt']
    },
    include_package_data=True,
    install_requires=requirements
)