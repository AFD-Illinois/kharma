FROM nvcr.io/nvidia/nvhpc:22.9-devel-cuda_multi-rockylinux8

# NVIDIA container has PowerTools, EPEL, dev tools installed

COPY . /app/

ENV PREFIX_PATH="/app/external/hdf5" DEVICE_ARCH=VOLTA70 C_NATIVE=nvc CXX_NATIVE=nvc++

RUN cd /app && bash -ic './make.sh clean cuda hdf5'

CMD /app/kharma.cuda
