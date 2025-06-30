import dolfinx

from Thirdparty.pyadjoint.pyadjoint.overloaded_type import OverloadedType, register_overloaded_type


@register_overloaded_type
class Mesh(OverloadedType, dolfinx.mesh.Mesh):
    def __init__(self, domain: dolfinx.mesh.Mesh, *args, **kwargs):
        super(Mesh, self).__init__(*args, **kwargs)
        dolfinx.mesh.Mesh.__init__(self, mesh=domain._cpp_object, domain=domain._ufl_domain)

    def _ad_create_checkpoint(self):
        return self.geometry.x.copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.geometry.x[:] = checkpoint
        return self
