import numpy as np

from thirdparty.pyadjoint.pyadjoint.overloaded_type import (
    register_overloaded_type,
    get_overloaded_class,
    OverloadedType
)


class BaseOverloadedType(OverloadedType):
    def __init__(self, *args, **kwargs):
        self.block = None
        OverloadedType.__init__(self, *args, **kwargs)
    
    def _ad_will_add_as_dependency(self):
        super()._ad_will_add_as_dependency()

    def _ad_will_add_as_output(self):
        super()._ad_will_add_as_output()