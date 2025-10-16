/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "topoIncompressibleFluid.H"
#include "localEulerDdtScheme.H"
#include "linear.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(topoIncompressibleFluid, 0);
    addToRunTimeSelectionTable(solver, topoIncompressibleFluid, fvMesh);
}
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::solvers::topoIncompressibleFluid::correctCoNum()
{
    fluidSolver::correctCoNum(phi);
}


void Foam::solvers::topoIncompressibleFluid::continuityErrors()
{
    fluidSolver::continuityErrors(phi);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solvers::topoIncompressibleFluid::topoIncompressibleFluid(fvMesh& mesh)
:
    fluidSolver(mesh),

    p_
    (
        IOobject
        (
            "p",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    ),

    pressureReference(p_, pimple.dict()),

    U_
    (
        IOobject
        (
            "U",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    ),

    phi_
    (
        IOobject
        (
            "phi",
            runTime.name(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        linearInterpolate(U_) & mesh.Sf()
    ),

    viscosity(viscosityModel::New(mesh)),

    momentumTransport
    (
        incompressible::momentumTransportModel::New
        (
            U_,
            phi_,
            viscosity
        )
    ),

    MRF(mesh),

    topoWeight_
    (
        IOobject
        (
            "topoWeight",
            runTime.name(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    ),

    p(p_),
    U(U_),
    phi(phi_),
    topoWeight(topoWeight_)
{
    mesh.schemes().setFluxRequired(p.name());

    momentumTransport->validate();

    if (transient())
    {
        correctCoNum();
    }
    else if (LTS)
    {
        Info<< "Using LTS" << endl;

        trDeltaT = tmp<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    fv::localEulerDdt::rDeltaTName,
                    runTime.name(),
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE
                ),
                mesh,
                dimensionedScalar(dimless/dimTime, 1),
                extrapolatedCalculatedFvPatchScalarField::typeName
            )
        );
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solvers::topoIncompressibleFluid::~topoIncompressibleFluid()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::topoIncompressibleFluid::preSolve()
{
    if ((mesh.dynamic() || MRF.size()) && !Uf.valid())
    {
        Info<< "Constructing face momentum Uf" << endl;

        // Ensure the U BCs are up-to-date before constructing Uf
        U_.correctBoundaryConditions();

        Uf = new surfaceVectorField
        (
            IOobject
            (
                "Uf",
                runTime.name(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            fvc::interpolate(U)
        );
    }

    fvModels().preUpdateMesh();

    if (transient())
    {
        correctCoNum();
    }
    else if (LTS)
    {
        setRDeltaT();
    }

    // Update the mesh for topology change, mesh to mesh mapping
    mesh_.update();
}


void Foam::solvers::topoIncompressibleFluid::prePredictor()
{}


void Foam::solvers::topoIncompressibleFluid::momentumTransportPredictor()
{
    momentumTransport->predict();
}


void Foam::solvers::topoIncompressibleFluid::thermophysicalTransportPredictor()
{}


void Foam::solvers::topoIncompressibleFluid::thermophysicalPredictor()
{}


void Foam::solvers::topoIncompressibleFluid::pressureCorrector()
{
    while (pimple.correct())
    {
        correctPressure();
    }

    tUEqn.clear();
}


void Foam::solvers::topoIncompressibleFluid::momentumTransportCorrector()
{
    viscosity->correct();
    momentumTransport->correct();
}


void Foam::solvers::topoIncompressibleFluid::thermophysicalTransportCorrector()
{}


void Foam::solvers::topoIncompressibleFluid::postSolve()
{}


// ************************************************************************* //
