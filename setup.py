"""
    Setup file primarily for compilation of C extensions (for now)
"""

import setuptools

spam = setuptools.Extension("straw.compiled.spam", sources=["extensions/spammodule.c"],
                            include_dirs=["/usr/local/lib"])

setuptools.setup(name="straw",
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 version="0.2",
                 description="Lossless audio codec",
                 ext_modules=[spam])
