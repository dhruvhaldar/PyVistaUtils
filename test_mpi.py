from mpi4py import MPI
print(f"Rank: {MPI.COMM_WORLD.Get_rank()}, Size: {MPI.COMM_WORLD.Get_size()}")