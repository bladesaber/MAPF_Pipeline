import matplotlib.pyplot as plt

from thirdparty.pyadjoint.pyadjoint.tape import Tape
from thirdparty.pyadjoint.pyadjoint.tape import get_working_tape
from thirdparty.pyadjoint.pyadjoint import tape


class start_annotation(object):
    def __enter__(self):
        tape._annotation_enabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        tape._annotation_enabled = False


class TapeUtils(object):
    @staticmethod
    def get_working_tape():
        return get_working_tape()

    @staticmethod
    def visualise_tape(tape: Tape):
        plt.figure(figsize=(10, 10))
        tape.visualise()
        plt.show()

