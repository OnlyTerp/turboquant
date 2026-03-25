"""
TurboQuant vLLM Plugin — Setup

Installs the package and registers it as a vLLM platform plugin so that
``vllm serve <model> --attention-backend turboquant`` works out of the box.
"""

from pathlib import Path

from setuptools import setup, find_packages

HERE = Path(__file__).resolve().parent

# Read long description from README
long_description = ""
readme_path = HERE / "vllm_plugin" / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="turboquant-vllm-plugin",
    version="0.1.0",
    description=(
        "TurboQuant: 3.5-bit KV cache compression for vLLM — "
        "PolarQuant + QJL with near-zero accuracy loss"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TurboQuant Contributors",
    license="Apache-2.0",
    url="https://github.com/your-org/turboquant-vllm-plugin",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
    ],
    extras_require={
        "vllm": ["vllm>=0.4.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "ruff",
            "mypy",
        ],
    },
    entry_points={
        "vllm.platform_plugins": [
            "turboquant = vllm_plugin.platform:TurboQuantPlatform",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="vllm kv-cache compression quantization turboquant polarquant qjl",
)
