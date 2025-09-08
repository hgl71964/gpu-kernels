from setuptools import setup

setup(
    name="triton_kernel",
    version="0.1",
    #description="A Triton GEMM example",
    author="Gemini",
    #py_modules=["gemm"],
    install_requires=[
        "torch",
        "triton"
    ],
)
