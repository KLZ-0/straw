from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("straw.lpc.ext_lpc", ["src/straw/lpc/ext_lpc.pyx"]),
    Extension("straw.rice.ext_rice", ["src/straw/rice/ext_rice.pyx"]),
    Extension("straw.io.ext_io", ["src/straw/io/ext_io.pyx"]),
]

setup(
    name="Straw",
    version="0.1",
    description="Lossless codec for multichannel audio",
    author="Adrian Kalazi",
    author_email="adrian@kalazi.com",
    url="https://github.com/KLZ-0/straw/",
    packages=["straw"],
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, gdb_debug=True)
)
