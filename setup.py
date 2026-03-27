from set_utils import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='KudaHitamCUDA',
    ext_modules=[
        CUDAExtension('KudaHitamCUDA', [
            'KudaHitamCUDA.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
