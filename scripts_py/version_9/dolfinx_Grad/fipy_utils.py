import fipy
import numpy as np
import pandas as pd
from fipy.tools import numerix
from fipy.variables.faceGradVariable import _FaceGradVariable


class FipyUtils(object):
    @staticmethod
    def solve_steady_navier_stoke_eq_3d(
            domain: fipy.Mesh,
            velocity_x: fipy.CellVariable,
            velocity_y: fipy.CellVariable,
            velocity_z: fipy.CellVariable,
            pressure: fipy.CellVariable,
            viscosity: float,
            inlet_velocity_x: float, inlet_velocity_y: float, inlet_velocity_z: float,
            bry_fix: np.ndarray, bry_inlet: np.ndarray, bry_outlet: np.ndarray,
            velocity_relaxation=0.5, pressure_relaxation=0.8, max_it=100, tol=1e-6
    ):
        """
        boundary should be applied first
        """
        pressure_correction = fipy.CellVariable(mesh=domain)
        velocity = fipy.FaceVariable(mesh=domain, rank=1)  # todo need to check here

        eq_velocity_x = fipy.DiffusionTerm(coeff=viscosity) - fipy.ConvectionTerm(velocity) - pressure.grad[0]
        eq_velocity_y = fipy.DiffusionTerm(coeff=viscosity) - fipy.ConvectionTerm(velocity) - pressure.grad[1]
        eq_velocity_z = fipy.DiffusionTerm(coeff=viscosity) - fipy.ConvectionTerm(velocity) - pressure.grad[2]

        ap = fipy.CellVariable(mesh=domain, value=1.)
        coeff = 1. / ap.arithmeticFaceValue * domain._faceAreas * domain._cellDistances
        eq_pressure_correction = fipy.DiffusionTerm(coeff=coeff) - velocity.divergence  # todo poisson of p equal divergence of velocity ?

        volume = fipy.CellVariable(mesh=domain, value=domain.cellVolumes, name='Volume')
        control_volume = volume.arithmeticFaceValue

        velocity_x.constrain(0., bry_fix)
        velocity_y.constrain(0., bry_fix)
        velocity_z.constrain(0., bry_fix)

        velocity_x.constrain(inlet_velocity_x, bry_inlet)
        velocity_y.constrain(inlet_velocity_y, bry_inlet)
        velocity_z.constrain(inlet_velocity_z, bry_inlet)

        pressure_correction.constrain(0., bry_outlet)

        step = 0
        while True:
            step += 1

            # solve the Stokes equations to get starred values
            eq_velocity_x.cacheMatrix()
            res_x = eq_velocity_x.sweep(var=velocity_x, underRelaxation=velocity_relaxation)
            mat_x = eq_velocity_x.matrix

            res_y = eq_velocity_y.sweep(var=velocity_y, underRelaxation=velocity_relaxation)
            res_z = eq_velocity_z.sweep(var=velocity_z, underRelaxation=velocity_relaxation)

            # update the ap coefficient from the matrix diagonal
            ap[:] = -numerix.asarray(mat_x.takeDiagonal())  # todo how to change

            # update the face velocities based on starred values with the Rhie-Chow correction.
            # cell pressure gradient
            grad_pressure = pressure.grad

            # face pressure gradient
            grad_face_pressure = _FaceGradVariable(pressure)

            velocity[0] = (
                    velocity_x.arithmeticFaceValue
                    + control_volume / ap.arithmeticFaceValue
                    * (grad_pressure[0].arithmeticFaceValue - grad_face_pressure[0])
            )
            velocity[1] = (
                    velocity_y.arithmeticFaceValue +
                    control_volume / ap.arithmeticFaceValue *
                    (grad_pressure[1].arithmeticFaceValue - grad_face_pressure[1])
            )
            velocity[2] = (
                    velocity_z.arithmeticFaceValue +
                    control_volume / ap.arithmeticFaceValue *
                    (grad_pressure[2].arithmeticFaceValue - grad_face_pressure[2])
            )

            # todo need to change here
            velocity[..., bry_fix] = 0.
            velocity[0, bry_inlet] = inlet_velocity_x
            velocity[1, bry_inlet] = inlet_velocity_y
            velocity[2, bry_inlet] = inlet_velocity_z

            # solve the pressure correction equation
            eq_pressure_correction.cacheRHSvector()
            # left bottom point must remain at pressure 0, so no correction
            res_p = eq_pressure_correction.sweep(var=pressure_correction)
            rhs = eq_pressure_correction.RHSvector

            # update the pressure using the corrected value
            pressure.setValue(pressure + pressure_relaxation * pressure_correction)

            # update the velocity using the corrected pressure
            velocity_x.setValue(velocity_x - pressure_correction.grad[0] / ap * domain.cellVolumes)
            velocity_y.setValue(velocity_y - pressure_correction.grad[1] / ap * domain.cellVolumes)
            velocity_z.setValue(velocity_z - pressure_correction.grad[2] / ap * domain.cellVolumes)

            print(f"sweep:{step} res_x:{res_x} res_y:{res_y} res_z:{res_z} res_p:{res_p} "
                  f"continuity:{np.max(np.abs(rhs))}")

            if step > max_it:
                break

            if np.max([res_x, res_y, res_p, np.max(np.abs(rhs))]) < tol:
                break
