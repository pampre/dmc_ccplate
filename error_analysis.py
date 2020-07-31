from math import pi, fabs, log10
import numpy as np
from dmcpy.charge import ccplate as sigma_disk
from mpi4py import MPI

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        value = 0.
        num = 10000000

        analytic = 1 / (4. * pi)

        # Distribute works
        for worker in range(1, size):
            comm.send(num, dest=worker)

        # Initialize processes
        received_processes = 0
        count = 0

        # Await results
        while received_processes < (size - 1):
            count += comm.recv(source=MPI.ANY_SOURCE)
            received_processes += 1

        value = count / (num * 100)

        numerical = value
        error = fabs(analytic - numerical)
        log_error = log10(error)

        data_np = np.array([log10(num), log_error])
        print(log_error)
        np.save('error7.npy', data_np)

    else:
        num = comm.recv(source=0)
        count = sigma_disk(num)
        comm.send(count, dest=0)















