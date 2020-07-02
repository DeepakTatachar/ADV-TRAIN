import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="advtrain",
    version="0.0.2",
    author="Deepak Tatachar, Sangamesh Kodge",
    author_email="dravikum@purdue.edu",
    description="ADV-TRAIN is Deep Vision TRaining And INference framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepakTatachar/ADV-TRAIN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)