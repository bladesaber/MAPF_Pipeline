from typing import Literal, List, Any

from mpi4py import MPI
import numpy as np


class OpenMpiUtils(object):
    @staticmethod
    def get_comm_world():
        return MPI.COMM_WORLD
    
    @staticmethod
    def get_mpi_rank() -> int:
        """rank of the current MPI process"""
        comm_world = MPI.COMM_WORLD
        rank = comm_world.Get_rank()
        return rank

    @staticmethod
    def synchronize_all_ranks() -> None:
        """wait for all MPI processes to reach this point"""
        OpenMpiUtils.get_comm_world().Barrier()

    @staticmethod
    def get_mpi_size() -> int:
        """total number of MPI processes"""
        return OpenMpiUtils.get_comm_world().Get_size()

    @staticmethod
    def is_master_rank() -> bool:
        """check if the current process is the master process"""
        return OpenMpiUtils.get_mpi_rank() == 0
    
    @staticmethod
    def print(text: str, rank: int = -1) -> None:
        if rank >= 0:
            print(f'Rank {rank}: {text}', flush=True)
        else:
            print(text, flush=True)
    
    @staticmethod
    def get_time() -> float:
        return MPI.Wtime()
    
    # ------ Reduce operations ------
    @staticmethod
    def get_opeartion(mode: Literal['max', 'min', 'sum', 'prod', 'bool_and', 'bool_or', 'bool_xor']) -> MPI.Op:
        if mode == 'max':
            return MPI.MAX
        elif mode == 'min':
            return MPI.MIN
        elif mode == 'sum':
            return MPI.SUM
        elif mode == 'prod':
            return MPI.PROD
        elif mode == 'bool_and':
            return MPI.LAND
        elif mode == 'bool_or':
            return MPI.LOR
        elif mode == 'bool_xor':
            return MPI.LXOR
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    @staticmethod
    def allreduce(
        x: np.ndarray, 
        mode: Literal['max', 'min', 'sum', 'prod', 'bool_and', 'bool_or', 'bool_xor']
    ) -> np.ndarray:
        """each process reduce the array x from all the other processes.
        it will return a global array (merge results from all processes) to current process.
        faster process will wait for slower process to finish.
        """
        return OpenMpiUtils.get_comm_world().allreduce(x, op=OpenMpiUtils.get_opeartion(mode))

    @staticmethod
    def Allreduce(
        local_array: np.ndarray, global_array: np.ndarray,
        mode: Literal['max', 'min', 'sum', 'prod', 'bool_and', 'bool_or', 'bool_xor']
    ) -> None:
        """each process reduce the local array from all the other processes to the its own global array.
        it will store the result in the global array to current process.
        faster process will wait for slower process to finish.
        """
        OpenMpiUtils.get_comm_world().Allreduce(local_array, global_array, op=OpenMpiUtils.get_opeartion(mode))

    @staticmethod
    def reduce(
        x: np.ndarray, 
        mode: Literal['max', 'min', 'sum', 'prod', 'bool_and', 'bool_or', 'bool_xor'], 
        root: int = 0
    ):
        """reduce the array x from all the other processes to the root process.
        only rank equal to root will get the result, the other processes will get None.
        faster process will wait for slower process to finish.
        """
        return OpenMpiUtils.get_comm_world().reduce(x, op=OpenMpiUtils.get_opeartion(mode), root=root)
    
    @staticmethod
    def Reduce(
        local_array: np.ndarray, global_array: np.ndarray,
        mode: Literal['max', 'min', 'sum', 'prod', 'bool_and', 'bool_or', 'bool_xor'],
        root: int = 0
    ):
        """reduce the local array from all the other processes to the its own global array.
        only rank equal to root will get the result, the other processes will get None.
        faster process will wait for slower process to finish.
        """
        OpenMpiUtils.get_comm_world().Reduce(local_array, global_array, op=OpenMpiUtils.get_opeartion(mode), root=root)

    # ------ Communication operations ------
    @staticmethod
    def broadcast(data: Any, root: int = 0):
        """broadcast the data from the root process to all processes
        data will be copied to all processes and casue large memory usage and time cost.
        Advised to use file-based method instead.
        process must allocate same size memory of data before broadcast.
        """
        OpenMpiUtils.get_comm_world().Bcast(data, root=root)
    
    @staticmethod
    def send(data: Any, dest: int, tag: int = 0) -> None:
        """send the data to the destination process"""
        OpenMpiUtils.get_comm_world().send(data, dest=dest, tag=tag)
    
    @staticmethod
    def send_to_all(data: Any, tag: int = 0) -> None:
        rank = OpenMpiUtils.get_mpi_rank()
        for i in range(OpenMpiUtils.get_mpi_size()):
            if i == rank:
                continue
            OpenMpiUtils.send(data, dest=i, tag=tag)
    
    @staticmethod
    def recv(source: int, tag: int = 0) -> Any:
        """receive the data from the source process"""
        return OpenMpiUtils.get_comm_world().recv(source=source, tag=tag)

    @staticmethod
    def recv_from_all(tag: int = 0) -> Any:
        return OpenMpiUtils.get_comm_world().recv(source=MPI.ANY_SOURCE, tag=tag)

    # ------ Gather operations ------
    @staticmethod
    def gather(data: Any, root: int = 0) -> List[Any]:
        """gather the data from all processes to the root process follow the order of the processes"""
        return OpenMpiUtils.get_comm_world().gather(data, root=root)  # type: ignore

    @staticmethod
    def allgather(data: Any) -> List[Any]:
        """gather the data from all processes to all processes follow the order of the processes"""
        return OpenMpiUtils.get_comm_world().allgather(data)  # type: ignore
    
    @staticmethod
    def Gather(send_array: np.ndarray, recv_array: np.ndarray | None = None, root: int = 0) -> None:
        """gather the local array from all processes to the root process follow the order of the processes"""
        if recv_array is None:
            recv_array = np.zeros(shape=(OpenMpiUtils.get_mpi_size() * np.prod(send_array.shape), ), dtype=send_array.dtype)
        OpenMpiUtils.get_comm_world().Gather(send_array, recv_array, root=root)

    @staticmethod
    def Allgather(send_array: np.ndarray, recv_array: np.ndarray | None = None) -> None:
        """gather the local array from all processes to all processes follow the order of the processes"""
        if recv_array is None:
            recv_array = np.zeros(shape=(OpenMpiUtils.get_mpi_size() * np.prod(send_array.shape), ), dtype=send_array.dtype)
        OpenMpiUtils.get_comm_world().Allgather(send_array, recv_array)

    @staticmethod
    def Scatter(send_array: np.ndarray, recv_array: np.ndarray|None, root: int = 0) -> Any:
        """distribute the data from the root process to all processes"""
        if recv_array is None:
            send_shape = np.prod(send_array.shape)
            size = OpenMpiUtils.get_mpi_size()
            assert send_shape % size == 0, "[Error]: send_array shape must be divisible by the number of processes"
            recv_shape = send_shape // size
            recv_array = np.zeros(shape=(recv_shape, ), dtype=send_array.dtype)
        OpenMpiUtils.get_comm_world().Scatter(send_array, recv_array, root=root)
    
    @staticmethod
    def scatter(data: List[Any], root: int = 0) -> Any:
        assert isinstance(data, list), "[Error]: data must be a list"
        assert len(data) == OpenMpiUtils.get_mpi_size(), "[Error]: data must have the same size as the number of processes"
        return OpenMpiUtils.get_comm_world().scatter(data, root=root)

    @staticmethod
    def Scatterv(send_array: np.ndarray, recv_array: np.ndarray|None, root: int = 0) -> Any:
        """distribute the data from the root process to all processes, using variable length array"""
        OpenMpiUtils.get_comm_world().Scatterv(send_array, recv_array, root=root)
