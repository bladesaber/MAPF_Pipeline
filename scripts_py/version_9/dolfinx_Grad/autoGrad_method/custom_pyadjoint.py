import time

from Thirdparty.pyadjoint.pyadjoint import *
from Thirdparty.pyadjoint.pyadjoint.enlisting import Enlist


class RecordTape(Tape):
    def evaluate_adj_rewrite(self, last_block=0, markings=False):
        module_time0 = time.time()
        for i in self._bar("Evaluating adjoint").iter(range(len(self._blocks) - 1, last_block - 1, -1)):
            block_time0 = time.time()
            self._blocks[i].evaluate_adj(markings=markings)
            block_time_cost = time.time() - block_time0
            print(f"Block {i} Cost Time: {block_time_cost} block_Type:{type(self._blocks[i])}")

        module_time_cost = time.time() - module_time0
        print(f"Adj Module Cost Time: {module_time_cost}")


class RecordFunctional(ReducedFunctional):
    def __call__(self, values):
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError("values should be a list of same length as controls.")

        # Call callback.
        self.eval_cb_pre(self.controls.delist(values))

        for i, value in enumerate(values):
            self.controls[i].update(value)

        self.tape.reset_blocks()
        blocks = self.tape.get_blocks()
        with self.marked_controls():
            with stop_annotating():
                module_time0 = time.time()

                for i in self.tape._bar("Evaluating functional").iter(range(len(blocks))):
                    block_time0 = time.time()
                    blocks[i].recompute()
                    block_time_cost = time.time() - block_time0
                    print(f"Block {i} Cost Time: {block_time_cost} block_Type:{type(blocks[i])}")

                module_time_cost = time.time() - module_time0
                print(f"Recompute Module Cost Time: {module_time_cost}")

        # ReducedFunctional can result in a scalar or an assembled 1-form
        func_value = self.functional.block_variable.saved_output
        # Scale the underlying functional value
        func_value *= self.scale

        # Call callback
        self.eval_cb_post(func_value, self.controls.delist(values))

        return func_value
