from pathlib import Path

from setuptools import Extension, setup, find_packages

ROOT = Path(__file__).parent

extensions = [
    Extension("straw.lpc.ext_lpc", ["src/straw/lpc/ext_lpc.pyx"]),
    Extension("straw.rice.ext_rice", ["src/straw/rice/ext_rice.pyx"]),
    Extension("straw.io.ext_io", ["src/straw/io/ext_io.pyx"]),
]

with (ROOT / "requirements.txt").open("r") as f:
    requirements = [
        line.strip()
        for line in f.readlines()
        if not line.startswith("#")
    ]

with (ROOT / "README.md").open("r") as f:
    README = f.read()

setup(
    # This is deprecated and replaced by pyproject.toml
    # setup_requires=[
    #     # Setuptools 18.0 properly handles Cython extensions.
    #     "setuptools>=18.0",
    #     "cython",
    # ],
    name="straw-codec",
    version="0.2",
    description="Lossless codec for multichannel audio",
    author="KLZ-0",
    author_email="adrian@kalazi.com",
    url="https://github.com/KLZ-0/straw/",
    packages=[pkg for pkg in find_packages(ROOT / "src") if "figures" not in pkg],
    install_requires=requirements,
    package_dir={"": "src"},
    ext_modules=extensions,
    python_requires=">=3.6",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["straw=straw.standalone:main"],
    },
)
