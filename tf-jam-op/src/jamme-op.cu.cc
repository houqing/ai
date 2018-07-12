
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "jamme-op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"


#define HALF_MANTISSA_BIT_MASK	(0x3ff)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void JamMeCudaKernel(const int nthreads, const T* in, T* jam, T* out, float *sim) {
  const unsigned short a = 3;

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    unsigned short b;
    b = __half_as_ushort(in[idx]) & HALF_MANTISSA_BIT_MASK;
    out[idx] = in[idx];
    if ((b + a) > b) {
      jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + a);
    }
  }

  sim[0] = 0.5;
}

template <typename T>
__global__ void JamMeCudaRandKernel(const int nthreads, const T* in, T* jam, T* out, float *sim) {
  const unsigned short a = 3;

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    unsigned short b;
    b = __half_as_ushort(in[idx]) & HALF_MANTISSA_BIT_MASK;
    out[idx] = in[idx];
    if ((b + a) > b) {
      jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + a);
    }
  }

  sim[0] = 0.5;
}

template <typename T>
__global__ void JamMeGradCudaKernel(const int nthreads, const T* in, T* jam) {
  const unsigned short a = 3;

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    unsigned short b;
    b = __half_as_ushort(in[idx]) & HALF_MANTISSA_BIT_MASK;
    if ((b - a) < b) {
      jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) - a);
    }
  }
}

}	// namespace


namespace functor {

template <typename T>
struct JamMeFunctor<GPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam, T* out, float *sim) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();

    CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
    JamMeCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, out, jam, sim);
  };
};

template <typename T>
struct JamMeGradFunctor<GPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();

    CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
    JamMeGradCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam);
  };
};

#define DEFINE_GPU_SPECS(T)	\
  template struct JamMeFunctor<GPUDevice, T>;	\
  template struct JamMeGradFunctor<GPUDevice, T>;

//TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_half(DEFINE_GPU_SPECS);
//TF_CALL_float(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

} // functor

} // tensorflow


#endif	// GOOGLE_CUDA

