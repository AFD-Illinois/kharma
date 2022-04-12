FROM centos:stream8

# Need Dev, PowerTools + EPEL, MPI + PHDF5
RUN dnf -y groupinstall "Development Tools" && dnf -y install epel-release && dnf config-manager --set-enabled powertools && \
    dnf -y install cmake environment-modules openmpi-devel hdf5-openmpi-devel hdf5-openmpi-static

COPY . /app/

ENV PREFIX_PATH=/usr/lib64/mpich EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON"

RUN cd /app && bash -ic 'module load mpi/openmpi-x86_64 && ./make.sh clean'

CMD /app/kharma.host
