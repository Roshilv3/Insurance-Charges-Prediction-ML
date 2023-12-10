from setuptools import find_packages, setup
from typing import List

hypen_e_dot = "-e ."

def get_requirements(file_path:str) -> List[str]:

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace('\n', '') for i in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

setup(
    name= "Insurance-Charges-Prediction-ML Project",
    version= "0.0.1",
    author= "Roshil Verma",
    author_email= "roshil.verma.3@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements("requirements.txt")
)