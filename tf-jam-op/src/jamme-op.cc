
#define EIGEN_USE_THREADS

#include "jamme-op.h"

#include <stdlib.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include <functional>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


#if 1
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/stream_executor.h"

#endif	// GOOGLE_CUDA
#endif



namespace tensorflow {

REGISTER_OP("JamMe")
.Input("in: T")
.Output("jam: T")
.Output("out: T")
.Output("sim: float")
.Attr("T: {half}")
.SetShapeFn(shape_inference::UnchangedShape);


#if 0
#if 1
REGISTER_OP("JamMeGrad")
.Input("in: T")
.Output("jam: T")
.Attr("T: {half}")
.SetShapeFn(shape_inference::UnchangedShape);
#else
typedef FunctionDefHelper FDH;

Status JamMeGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {half}"}},
      // Nodes
      {
        {{"dx"}, "ReluGrad", {"dy", "x"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}

REGISTER_OP_GRADIENT("JamMe", JamMeGrad);
#endif
#endif

}


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class JamMeOp : public OpKernel {
  public:
  explicit JamMeOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);

    Tensor* jam = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &jam));
    Tensor* out = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, in.shape(), &out));
    Tensor* sim = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &sim));

#if 1
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream avalible"));

//    Bitcast(out, float)
    auto rand_data_float = perftools::gputools::DeviceMemory<float>::MakeFromByteSize(out->template flat<T>().data(), out->template flat<T>().size() * sizeof(T));
#if 0
    if (is_rand_initialized_ == false) {
        int i;
        const uint8 seed_data[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        uint64 seed_bytes = 16;
        bool launch_status;
        launch_status = stream->ThenSetRngSeed(seed_data, seed_bytes).ok();
        OP_REQUIRES(ctx, launch_status, errors::Internal("JamMe rand seed failed"));
        is_rand_initialized_ = true;
    }
#endif

    bool launch_status;
    launch_status = stream->ThenPopulateRandUniform(&rand_data_float).ok();
    OP_REQUIRES(ctx, launch_status, errors::Internal("JamMe rand gen failed"));

//    Bitcast(out, T)
    auto rand_data_orig = perftools::gputools::DeviceMemory<T>::MakeFromByteSize(out->template flat<T>().data(), out->template flat<T>().size() * sizeof(T));
#endif

    functor::JamMeFunctor<Device, T>()(
        ctx,
        static_cast<const int>(in.NumElements()),
        in.flat<T>().data(),
        jam->flat<T>().data(),
        out->flat<T>().data(),
	(float*)sim->flat<float>().data());
  }

  private:
  int is_rand_initialized_ = false;
};

template <typename Device, typename T>
class JamMeGradOp : public OpKernel {
  public:
  explicit JamMeGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);

    Tensor* jam = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &jam));

    functor::JamMeGradFunctor<Device, T>()(
        ctx,
        static_cast<const int>(in.NumElements()),
        in.flat<T>().data(),
        jam->flat<T>().data());
  }
};


namespace functor {

template <typename T>
struct JamMeFunctor<CPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam, T* out, float *sim) {
  // XXX CPU version is not implemented
    ;
  }
};

template <typename T>
struct JamMeGradFunctor<CPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam) {
  // XXX CPU version is not implemented
    ;
  }
};

}	// namespace functor

#if 0
// XXX CPU version is not implemented
#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("JamMe")	\
		  .Device(DEVICE_CPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeOp<CPUDevice, T>);	\
	\
  REGISTER_KERNEL_BUILDER(Name("JamMeGrad")	\
		  .Device(DEVICE_CPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeGradOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
#endif


#if GOOGLE_CUDA


#if 1
#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("JamMe")	\
		  .Device(DEVICE_GPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeOp<GPUDevice, T>);	\
	\
  REGISTER_KERNEL_BUILDER(Name("JamMeGrad")	\
		  .Device(DEVICE_GPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeGradOp<GPUDevice, T>);

//TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);

#undef REGISTER_KERNEL
#endif


#endif	// GOOGLE_CUDA

}	// namespace tensorflow


