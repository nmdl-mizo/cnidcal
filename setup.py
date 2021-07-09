import setuptools

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setuptools.setup(
    name="cnidcal",
    version="0.0.1",
    license="MIT",
    author="Yaoshu Xie",
    author_email="ysxie@iis.u-tokyo.ac.jp",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "matplotlib"],
)
