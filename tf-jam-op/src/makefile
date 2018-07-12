
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


all: a b

test: t

clean:
	rm -f jamme-op.cu.o jamme-op.so

t:
	python jamme-op-example.py

a:
	nvcc -std=c++11 -c -o jamme-op.cu.o jamme-op.cu.cc ${TF_CFLAGS[@]} $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
		-I/usr/local \
		--expt-relaxed-constexpr

b:
	g++ -std=c++11 -shared -o jamme-op.so jamme-op.cc jamme-op.cu.o ${TF_CFLAGS[@]} $(TF_CFLAGS) -fPIC -lcudart ${TF_LFLAGS[@]} $(TF_LFLAGS) \
		-L/usr/local/cuda/lib64 -D GOOGLE_CUDA=1





1:
	nvcc -std=c++11 -c -o jamme-op.cu.o jamme-op.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
		-I/home/l00395915/_virtualenv/tf-gpu-1.8.0--build/lib/python3.5/site-packages/tensorflow/include \
		-I/usr/local \
		--expt-relaxed-constexpr

1bad:
	nvcc -std=c++11 -c -o jamme-op.cu.o jamme-op.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
		-I/home/l00395915/_virtualenv/tf-gpu-1.8.0--build/lib/python3.5/site-packages/tensorflow/include \
		-I/usr/local

#		-I../../../ -I../../../third_party/eigen3 -I/usr/local -I/usr/local/include \

2:
	g++ -std=c++11 -shared -o jamme-op.so jamme-op.cc jamme-op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} \
		-I/home/l00395915/_virtualenv/tf-gpu-1.8.0--build/lib/python3.5/site-packages/tensorflow/include \
		-I/usr/local \
		-L/usr/local/cuda/lib64

2bad:
	g++ -std=c++11 -shared -o jamme-op.so jamme-op.cc jamme-op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} \
		-I/home/l00395915/_virtualenv/tf-gpu-1.8.0--build/lib/python3.5/site-packages/tensorflow/include \
		-I/usr/local \
		-L/usr/local/cuda/lib64


#		-I../../../ -I../../../third_party/eigen3 -I/usr/local -I/usr/local/include \
