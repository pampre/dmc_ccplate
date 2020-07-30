from dmcpy.charge import ccplate as sigma_disk
from mpi4py import MPI
from time import time

# Number of points to simulate
import numpy as np
n = 10000000
p0 = np.array([0., 0., 0., 0.])
r0 = 1.
p1 = np.array([0., 0., 0., 0.])
r1 = 2.

def master():
    t0 = time()
    points_per_node = int(n / (size - 1))

    for worker in range(1, size):
        comm.send(points_per_node, dest = worker)

    received_processes = 0
    count = 0

    # Await results
    while received_processes < (size -1):
        count += comm.recv(source=MPI.ANY_SOURCE)
        received_processes += 1

    count /= n
    
    print('Computed charge density : {}'.format(count))
    t1 = time()
    print('time: {}'.format(t1 - t0))

def slave():
    points_to_calculate = comm.recv(source = 0)
    count = sigma_disk(points_to_calculate, p0, r0, p1, r1)
    comm.send(count, dest = 0)

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        # Master node
        master()
    else:
        # Slave nodes
        slave()

