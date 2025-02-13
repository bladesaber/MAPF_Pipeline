from Thirdparty.pyadjoint.pyadjoint import *
from Thirdparty.pyadjoint import pyadjoint


class start_annotation(object):
    def __enter__(self):
        pyadjoint.tape._annotation_enabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        pyadjoint.tape._annotation_enabled = False


class AuxiliaryType(OverloadedType):
    # reference pyadjoint/FloatingType

    def __init__(self, *args, **kwargs):
        self._ad_floating_active = kwargs.pop("_ad_floating_active", False)

        self.block_class = kwargs.pop("block_class", None)
        self._ad_args = kwargs.pop("_ad_args", [])
        self._ad_kwargs = kwargs.pop("_ad_kwargs", {})
        self.ad_block_tag = kwargs.pop("ad_block_tag", None)
        self.block = None

        self.output_block_class = kwargs.pop("output_block_class", None)
        self._ad_output_args = kwargs.pop("_ad_output_args", [])
        self._ad_output_kwargs = kwargs.pop("_ad_output_kwargs", {})
        self._ad_outputs = kwargs.pop("_ad_outputs", [])
        OverloadedType.__init__(self, *args, **kwargs)

    def create_block_variable(self):
        block_variable = OverloadedType.create_block_variable(self)
        # block_variable.floating_type = True
        return block_variable

    def _ad_will_add_as_dependency(self):
        if self._ad_floating_active:
            self._ad_annotate_block()
        self.block_variable.save_output(overwrite=False)

    def _ad_will_add_as_output(self):
        if self._ad_floating_active:
            self._ad_annotate_output_block()
        return True

    def _ad_annotate_block(self):
        if self.block_class is None:
            return

        tape = get_working_tape()
        block = self.block_class(*self._ad_args, **self._ad_kwargs)
        block.tag = self.ad_block_tag
        self.block = block
        tape.add_block(block)
        block.add_output(self.create_block_variable())

    def _ad_annotate_output_block(self):
        if self.output_block_class is None:
            return

        tape = get_working_tape()
        block = self.output_block_class(self, *self._ad_output_args, **self._ad_output_kwargs)
        tape.add_block(block)
        for output in self._ad_outputs:
            block.add_output(output.create_block_variable())
