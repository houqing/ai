
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "jamme-op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "cuda/include/cuda_fp16.h"


#define HALF_MANTISSA_BIT_MASK	(0x3ff)

// 
#define JAM_BIT_NUM	0x2

// TODO
#define JAM_BIT_MASK	0x3
#define JAM_FLAG_MASK	0x8
#define JAM_BIT_RATIO	100

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {
#include "cuda/include/cuda_fp16.h"

// 0x3b00: 0.875
// 0x3c00: 1.
// 0x3d00: 1.25
// 0x3e00: 1.5
// 0x3f00: 1.75

template <typename T>
__global__ void JamMeRandCudaKernel(const int nthreads, const T* in, T* jam, T* out, float *sim) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
	unsigned short tmp;
        unsigned short mask = (1<<JAM_BIT_NUM) -1;

        if ((((__half_as_ushort(in[idx])) << 1) >> 11) == 0x1f) { //NaN or Infinit
		jam[idx] = in[idx];
        } else if (in[idx] == __ushort_as_half(0)) {
		jam[idx] = in[idx];
        } else if ((__half_as_ushort(out[idx])%100) < JAM_BIT_RATIO) {
            tmp = ((__half_as_ushort(in[idx])) & (~mask)) | ((__half_as_ushort(out[idx])) & mask);
            jam[idx] = __ushort_as_half(tmp);
	} else {
		jam[idx] = in[idx];
	}



#ifdef DAMAGE
        float16 data = in[idx];
        unsigned int r = *(unsigned int*)&rand[idx];
        unsigned short tmp;
        unsigned short mask = (1<<JAM_BIT_NUM) -1;
#if 0
        float a;
        float b;
#endif
        if (((( *(unsigned short*)&data) << 1) >> 11) == 0x1f) { //NaN or Infinit
//              atomicAdd(p, 1);
        } else if (data == 0) {
//              atomicAdd(p+1, 1);
        } else if ((r%100) < noise_ratio) {
            tmp = ((*(unsigned short*)&data) & (~mask)) | (((unsigned short)r) & mask);
            out[idx] = *(float16*)&tmp;
	}
#endif
  }

  sim[0] = 1;
}

template <typename T>
__global__ void JamMeRandCudaKernel_TODO(const int nthreads, const T* in, T* jam, T* out, float *sim) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
#if 0
    jam[idx] = in[idx];
#else
    if (((in[idx] != __ushort_as_half(0)) && ((__half_as_ushort(out[idx]) % 100) < JAM_BIT_RATIO))) {
	    unsigned short mi;
	    unsigned short jv;
	    mi = __half_as_ushort(in[idx]) & HALF_MANTISSA_BIT_MASK;
	    jv = __half_as_ushort(out[idx]) & JAM_BIT_MASK;
	    if (__half_as_ushort(out[idx]) & JAM_FLAG_MASK) {
		    if ((mi - jv) < mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) - jv);
		    } else if ((mi + jv) > mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + jv);
		    } else {
			    jam[idx] = in[idx];
			    jam[idx] = __ushort_as_half(0x3e00);
		    }
	    } else {
		    if ((mi + jv) > mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + jv);
		    } else if ((mi - jv) < mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) - jv);
		    } else {
			    jam[idx] = in[idx];
			    jam[idx] = __ushort_as_half(0x3f00);
		    }
	    }
	    //out[idx] = in[idx];
    } else {
	    jam[idx] = in[idx];
	    jam[idx] = __ushort_as_half(0x3b00);
    }
#endif

  }

  sim[0] = 2;
}

template <typename T>
__global__ void JamMeCudaKernel(const int nthreads, const T* in, T* jam, T* out, float *sim) {
  const unsigned short a = 3;

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    unsigned short b;
    b = __half_as_ushort(in[idx]) & HALF_MANTISSA_BIT_MASK;
    if ((b + a) > b) {
      jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + a);
    }
    //out[idx] = in[idx];
  }

  sim[0] = 3;
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
    //JamMeRandCudaKernel_TODO<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam, out, sim);
    JamMeRandCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam, out, sim);
    //JamMeCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam, out, sim);
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

