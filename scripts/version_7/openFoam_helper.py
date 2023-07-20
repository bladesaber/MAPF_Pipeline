import os
import shutil
import numpy as np
import pandas as pd
from typing import Dict
import json
import math

class TopoSetDictFun(object):
    @staticmethod
    def faceSetFun_patchToFace(sourceName, outputName, style='new'):
        contant = f'''    {{
        name    {outputName};
        type    faceSet;
        action  {style};
        source  patchToFace;
        sourceInfo
	{{
            patch {sourceName};
	}}
    }}
'''
        return contant

    @staticmethod
    def faceSetFun_cellToFace(sourceName, outputName, style='new'):
        contant = f'''    {{
        name    {outputName};
        type    faceSet;
        action  {style};
        source  cellToFace;
        option  all;
        set     {sourceName};
    }}
'''
        return contant

    @staticmethod
    def faceSetFun_boxToFace(xmin, ymin, zmin, xmax, ymax, zmax, outputName):
        contant = f'''    {{
        name    {outputName};
        type    faceSet;
        action  new;
        source  boxToFace;
        sourceInfo
	{{
	        box ({xmin:.2f} {ymin:.2f} {zmin:.2f}) ({xmax:.2f} {ymax:.2f} {zmax:.2f});
	}}
    }}
'''
        return contant

    @staticmethod
    def faceSetFun_faceToFace(sourceName, outputName, style='new'):
        ### style: new subset delete
        contant = f'''    {{
        name    {outputName};
        type    faceSet;
        action  {style};
        source  faceToFace;
        sourceInfo
	{{
            set {sourceName};
	}}
    }}
'''
        return contant

    @staticmethod
    def cellSetFun_cylinderToCell(x0, y0, z0, x1, y1, z1, radius, outputName):
        contant = f'''    {{
        name    {outputName};
        type    cellSet;
        action  new;
        source  cylinderToCell;
        point1  ({x0:.2f} {y0:.2f} {z0:.2f});
        point2  ({x1:.2f} {y1:.2f} {z1:.2f});
        radius  {radius};
    }}
'''
        return contant

class CreatePatchDictFun(object):
    @staticmethod
    def createPatch(objectName):
        contant = f'''    {{
        // Name of new patch
        name {objectName};

        // Dictionary to construct new patch from
        patchInfo
        {{
            type patch;
        }}

        // How to construct: either from 'patches' or 'set'
        constructFrom set;

        // If constructFrom = set : name of faceSet
        set {objectName};
    }}
'''
        return contant

class OpenFoamSimHelper(object):
    def create_openfoam_case(self, config, save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        
        os.mkdir(save_dir)

        ### ------ create Base
        timeStep_dir = os.path.join(save_dir, '0')
        os.mkdir(timeStep_dir)

        constant_dir = os.path.join(save_dir, 'constant')
        os.mkdir(constant_dir)

        system_dir = os.path.join(save_dir, 'system')
        os.mkdir(system_dir)
        
        self.output_AllrunBash(os.path.join(save_dir, 'Allrun'))
        self.output_AllcleanBash(os.path.join(save_dir, 'Allclean'))

        ### ------ copy stl
        geometry_dir = os.path.join(constant_dir, 'geometry')
        os.mkdir(geometry_dir)

        stl_name:str = os.path.basename(config['designMesh_stl'])
        stl_name = stl_name.lower()
        shutil.copy(config['designMesh_stl'], os.path.join(geometry_dir, stl_name))

        ### ------ create blockMeshDict
        self.output_BlockMeshDict(
            xmin = round(config['xmin'] - 0.1, 1), xmax = round(config['xmax'] + 0.1, 1),
            ymin = round(config['ymin'] - 0.1, 1), ymax = round(config['ymax'] + 0.1, 1),
            zmin = round(config['zmin'] - 0.1, 1), zmax = round(config['zmax'] + 0.1, 1),
            xGridNum = 50, yGridNum = 50, zGridNum = 50,
            convertToMeters = 1,
            output_file=os.path.join(system_dir, 'blockMeshDict')
        )

        self.output_meshQualityDict(os.path.join(system_dir, 'meshQualityDict'))

        self.output_surfaceFeaturesDict(stl_name ,os.path.join(system_dir, 'surfaceFeaturesDict'))
        
        objectName = stl_name.replace('.stl', '')
        self.output_snappyHexMeshDict(
            objectName=objectName, stl_baseName=stl_name,
            inside_pose=config['inside_pose'],
            nSurfaceLayers=2,
            output_file=os.path.join(system_dir, 'snappyHexMeshDict')
        )

        create_ObjectNames = self.output_topoSetDict(
            objectName=objectName, inOutletSetting=config, 
            output_file=os.path.join(system_dir, 'topoSetDict')
        )

        self.output_createPatchDict(
            create_ObjectNames=create_ObjectNames, 
            output_file=os.path.join(system_dir, 'createPatchDict')
        )

        self.output_controlDict(
            output_file=os.path.join(system_dir, 'controlDict')
        )

    def output_BlockMeshDict(
            self, xmin, xmax, ymin, ymax, zmin, zmax, xGridNum, yGridNum, zGridNum, convertToMeters, output_file
        ):
        contant = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  dev
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters {convertToMeters};

vertices
(
    ( {xmin} {ymin} {zmin} )
    ( {xmax} {ymin} {zmin} )
    ( {xmax} {ymax} {zmin} )
    ( {xmin} {ymax} {zmin} )
    ( {xmin} {ymin} {zmax} )
    ( {xmax} {ymin} {zmax} )
    ( {xmax} {ymax} {zmax} )
    ( {xmin} {ymax} {zmax} )
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({xGridNum} {yGridNum} {zGridNum}) simpleGrading (1 1 1)
);

boundary
(
    allBoundary
    {{
        type patch;
        faces
        (
            (3 7 6 2)
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

// ************************************************************************* //
'''
        with open(output_file, 'w') as f:
            f.write(contant)

    def output_meshQualityDict(self, output_file):
        contant = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  7
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      meshQualityDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Include defaults parameters from master dictionary
#includeEtc "caseDicts/mesh/generation/meshQualityDict"

//- minFaceWeight (0 -> 0.5)
minFaceWeight 0.02;

// ************************************************************************* //
'''
        with open(output_file, 'w') as f:
            f.write(contant)

    def output_surfaceFeaturesDict(self, stl_baseName, output_file):
        contant = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  7
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaces ("{stl_baseName}");
includedAngle       150;


// ************************************************************************* //
'''
        with open(output_file, 'w') as f:
            f.write(contant)

    def output_snappyHexMeshDict(self, objectName, stl_baseName:str, inside_pose, nSurfaceLayers, output_file):
        contant = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  dev
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Which of the steps to run
castellatedMesh true;
snap            true;
addLayers       true;

// Geometry. Definition of all surfaces. All surfaces are of class
// searchableSurface.
// Surfaces are used
// - to specify refinement for any mesh cell intersecting it
// - to specify refinement for any mesh cell inside/outside/near
// - to 'snap' the mesh boundary to the surface
geometry
{{
    {objectName}
    {{
        type triSurfaceMesh;
        file "{stl_baseName}";
    }}
	
    /*
    //- Refine a bit extra around the small centre hole
    refineHole
    {{
        type searchableSphere;
        centre (0 0 -0.012);
        radius 0.003;
    }}
    */
}};

// Settings for the castellatedMesh generation.
castellatedMeshControls
{{

    // Refinement parameters
    // ~~~~~~~~~~~~~~~~~~~~~

    // If local number of cells is >= maxLocalCells on any processor
    // switches from from refinement followed by balancing
    // (current method) to (weighted) balancing before refinement.
    maxLocalCells 100000;

    // Overall cell limit (approximately). Refinement will stop immediately
    // upon reaching this number so a refinement level might not complete.
    // Note that this is the number of cells before removing the part which
    // is not 'visible' from the keepPoint. The final number of cells might
    // actually be a lot less.
    maxGlobalCells 200000;

    // The surface refinement loop might spend lots of iterations refining just a
    // few cells. This setting will cause refinement to stop if <= minimumRefine
    // are selected for refinement. Note: it will at least do one iteration
    // (unless the number of cells to refine is 0)
    minRefinementCells 0;

    // Number of buffer layers between different levels.
    // 1 means normal 2:1 refinement restriction, larger means slower
    // refinement.
    nCellsBetweenLevels 1;

    // Explicit feature edge refinement
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Specifies a level for any cell intersected by its edges.
    // This is a featureEdgeMesh, read from constant/geometry for now.
    features
    (
        {{
            file "{objectName}.extendedFeatureEdgeMesh";
            level 0;
        }}
    );

    // Surface based refinement
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    // Specifies two levels for every surface. The first is the minimum level,
    // every cell intersecting a surface gets refined up to the minimum level.
    // The second level is the maximum level. Cells that 'see' multiple
    // intersections where the intersections make an
    // angle > resolveFeatureAngle get refined up to the maximum level.

    refinementSurfaces
    {{
        {objectName}
        {{
            // Surface-wise min and max refinement level
            level (1 1);
        }}
    }}

    features
    (
        {{
            file "{objectName}.eMesh";
            level 1;
        }}
    );

    resolveFeatureAngle 30;

    // Region-wise refinement
    // ~~~~~~~~~~~~~~~~~~~~~~

    // Specifies refinement level for cells in relation to a surface. One of
    // three modes
    // - distance. 'levels' specifies per distance to the surface the
    //   wanted refinement level. The distances need to be specified in
    //   descending order.
    // - inside. 'levels' is only one entry and only the level is used. All
    //   cells inside the surface get refined up to the level. The surface
    //   needs to be closed for this to be possible.
    // - outside. Same but cells outside.

    refinementRegions
    {{
	/*
        refineHole
        {{
            mode    inside;
            level   3;
        }}
	*/
    }}

    // Mesh selection
    // ~~~~~~~~~~~~~~

    // After refinement patches get added for all refinementSurfaces and
    // all cells intersecting the surfaces get put into these patches. The
    // section reachable from the insidePoint is kept.
    // NOTE: This point should never be on a face, always inside a cell, even
    // after refinement.
    // This is an outside point insidePoint (-0.033 -0.033 0.0033);
       insidePoint ({inside_pose[0]} {inside_pose[1]} {inside_pose[2]}); // Inside point

    // Whether any faceZones (as specified in the refinementSurfaces)
    // are only on the boundary of corresponding cellZones or also allow
    // free-standing zone faces. Not used if there are no faceZones.
    allowFreeStandingZoneFaces true;
}}

// Settings for the snapping.
snapControls
{{
    //- Number of patch smoothing iterations before finding correspondence
    //  to surface
    nSmoothPatch 3;

    //- Relative distance for points to be attracted by surface feature point
    //  or edge. True distance is this factor times local
    //  maximum edge length.
    tolerance 1.0;

    //- Number of mesh displacement relaxation iterations.
    nSolveIter 300;

    //- Maximum number of snapping relaxation iterations. Should stop
    //  before upon reaching a correct mesh.
    nRelaxIter 5;

    // Feature snapping

    //- Number of feature edge snapping iterations.
    //  Leave out altogether to disable.
    nFeatureSnapIter 10;

    //- Detect (geometric) features by sampling the surface
    implicitFeatureSnap false;

    //- Use castellatedMeshControls::features
    explicitFeatureSnap true;

    //- Detect features between multiple surfaces
    //  (only for explicitFeatureSnap, default = false)
    multiRegionFeatureSnap true;
}}

// Settings for the layer addition.
addLayersControls
{{
    // Are the thickness parameters below relative to the undistorted
    // size of the refined cell outside layer (true) or absolute sizes (false).
    relativeSizes true;

    // Per final patch (so not geometry!) the layer information
    layers
    {{
        {objectName}
        {{
            nSurfaceLayers {nSurfaceLayers};
        }}
    }}

    // Expansion factor for layer mesh
    expansionRatio 1.1;

    // Wanted thickness of final added cell layer. If multiple layers
    // is the thickness of the layer furthest away from the wall.
    // Relative to undistorted size of cell outside layer.
    // See relativeSizes parameter.
    finalLayerThickness 0.3;

    // Minimum thickness of cell layer. If for any reason layer
    // cannot be above minThickness do not add layer.
    // See relativeSizes parameter.
    minThickness 0.25;

    // If points get not extruded do nGrow layers of connected faces that are
    // also not grown. This helps convergence of the layer addition process
    // close to features.
    nGrow 0;

    // Advanced settings

    // When not to extrude surface. 0 is flat surface, 90 is when two faces
    // are perpendicular
    featureAngle 30;

    // Maximum number of snapping relaxation iterations. Should stop
    // before upon reaching a correct mesh.
    nRelaxIter 5;

    // Number of smoothing iterations of surface normals
    nSmoothSurfaceNormals 1;

    // Number of smoothing iterations of interior mesh movement direction
    nSmoothNormals 3;

    // Smooth layer thickness over surface patches
    nSmoothThickness 3;

    // Stop layer growth on highly warped cells
    maxFaceThicknessRatio 0.5;

    // Reduce layer growth where ratio thickness to medial
    // distance is large
    maxThicknessToMedialRatio 0.5;

    // Angle used to pick up medial axis points
    minMedianAxisAngle 90;

    // Create buffer region for new layer terminations
    nBufferCellsNoExtrude 0;


    // Overall max number of layer addition iterations. The mesher will exit
    // if it reaches this number of iterations; possibly with an illegal
    // mesh.
    nLayerIter 50;

    // Max number of iterations after which relaxed meshQuality controls
    // get used. Up to nRelaxIter it uses the settings in meshQualityControls,
    // after nRelaxIter it uses the values in meshQualityControls::relaxed.
    nRelaxedIter 20;
}}

// Generic mesh quality settings. At any undoable phase these determine
// where to undo.
meshQualityControls
{{
    #include "meshQualityDict"

    // Optional : some meshing phases allow usage of relaxed rules.
    // See e.g. addLayersControls::nRelaxedIter.
    relaxed
    {{
        //- Maximum non-orthogonality allowed. Set to 180 to disable.
        maxNonOrtho 75;
    }}
}}

// Advanced

// Write flags
writeFlags
(
    scalarLevels    // write volScalarField with cellLevel for postprocessing
    layerSets       // write cellSets, faceSets of faces in layer
    layerFields     // write volScalarField for layer coverage
);

// Merge tolerance. Is fraction of overall bounding box of initial mesh.
// Note: the write tolerance needs to be higher than this.
mergeTolerance 1e-6;


// ************************************************************************* //
'''
        with open(output_file, 'w') as f:
            f.write(contant)

    def output_topoSetDict(self, objectName, inOutletSetting:Dict, output_file):
        create_ObjectNames = []
        
        actions = []
        inOutletInfos: Dict = inOutletSetting['inOutlet']

        wall_FaceName = '%s_wall' % objectName
        newWall_action = TopoSetDictFun.faceSetFun_patchToFace(objectName, outputName=wall_FaceName, style='new')
        wholeMesh_FaceName = '%s_all' % objectName
        newWholeMesh_action = TopoSetDictFun.faceSetFun_patchToFace(objectName, outputName=wholeMesh_FaceName, style='new')
        actions.extend([newWholeMesh_action, newWall_action])
        create_ObjectNames.append({
            'objectName': wall_FaceName,
            'type': 'wall'
        })

        for inOutletIdx_str in inOutletInfos.keys():
            info = inOutletInfos[inOutletIdx_str]
            inOutletIdx = info['idx']

            pose = np.array(info['pose'])
            vec = np.array(info['vec'])

            cylinderPose0 = pose - vec * 0.05
            cylinderPose1 = pose + vec * 0.05
            
            cylinderCellName = 'CylinderCell_%d' % inOutletIdx
            newCylinderCell_action = TopoSetDictFun.cellSetFun_cylinderToCell(
                cylinderPose0[0], cylinderPose0[1], cylinderPose0[2],
                cylinderPose1[0], cylinderPose1[1], cylinderPose1[2],
                radius=info['radius'] * 1.02,
                outputName=cylinderCellName
            )

            cylinderFaceName = 'CylinderFace_%d' % inOutletIdx
            cylinderCell2face_action = TopoSetDictFun.faceSetFun_cellToFace(
                sourceName=cylinderCellName, outputName=cylinderFaceName
            )

            inOutlet_name = '%s_%d' % (info['type'], inOutletIdx)
            new_inOutlet_action = TopoSetDictFun.faceSetFun_faceToFace(wholeMesh_FaceName, inOutlet_name)
            extract_inOutlet_action = TopoSetDictFun.faceSetFun_faceToFace(cylinderFaceName, inOutlet_name, style='subset')
            remove_inOutlet_action = TopoSetDictFun.faceSetFun_faceToFace(inOutlet_name, wall_FaceName, style='delete')

            create_ObjectNames.append({
                'objectName': inOutlet_name,
                'type': 'patch'
            })

            actions.extend([
                newCylinderCell_action, cylinderCell2face_action, 
                new_inOutlet_action, extract_inOutlet_action, remove_inOutlet_action,
                "\n // ------------------------------------------------- \n"
            ])

        actions_contant = '\n'.join(actions)

        contant = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  dev
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
{actions_contant}
);
'''
        with open(output_file, 'w') as f:
            f.write(contant)
        
        return create_ObjectNames

    def output_createPatchDict(self, create_ObjectNames, output_file):
        actions = []

        for objectInfo in create_ObjectNames:
            createPatch_contant = CreatePatchDictFun.createPatch(objectInfo['objectName'])
            actions.extend([createPatch_contant])

        actions_contant = '\n'.join(actions)

        contant = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  dev
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

patches
(
{actions_contant}
);

'''
        with open(output_file, 'w') as f:
            f.write(contant)

    def output_controlDict(self, output_file):
        contant = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  7
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom      startTime;

startTime       0;

stopAt          endTime;

endTime       1;

deltaT          0.0005;

writeControl    timeStep;

writeInterval   50;

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

adjustTimestep yes;

maxCo 1;

maxDeltaT 0.001;

functions
{{
    #includeFunc residuals
}}


// ************************************************************************* //
        '''

        with open(output_file, 'w') as f:
            f.write(contant)

    def output_AllrunBash(self, output_file):
        with open(output_file, 'w') as f:
            f.write("#!/bin/sh" + "\n")
            f.write("\n")

            f.write("# Source tutorial run functions \n . $WM_PROJECT_DIR/bin/tools/RunFunctions" + "\n")
            f.write("\n")

            f.write("touch vis.foam" + "\n")
            f.write("runApplication blockMesh" + "\n")
            f.write("runApplication surfaceFeatures" + "\n")
            f.write("runApplication snappyHexMesh -overwrite" + "\n")
            f.write("runApplication topoSet" + "\n")
            f.write("runApplication createPatch -overwrite" + "\n")
            f.write("cp -f ${PWD}/constant/polyMesh/cellLevel ${PWD}/0/cellLevel" + "\n")
            f.write("cp -f ${PWD}/constant/polyMesh/pointLevel ${PWD}/0/pointLevel" + "\n")
            f.write("# decomposePar" + "\n")
            f.write("# mpirun --allow-run-as-root -np 10 icoFoam -parallel" + "\n")
            f.write("# reconstructPar")

    def output_AllcleanBash(self, output_file):
        with open(output_file, 'w') as f:
            f.write("#!/bin/sh" + "\n")
            f.write("\n")

            f.write("cd ${0%/*} || exit 1" + "\n")
            f.write("\n")

            f.write("# Source tutorial clean functions\n. $WM_PROJECT_DIR/bin/tools/CleanFunctions")
            f.write("\n")

            f.write("cleanCase" + "\n")
            f.write("cleanExplicitFeatures")

def main():
    openfoam_helper = OpenFoamSimHelper()

    stl_dir = '/home/quan/Desktop/tempary/application_pipe/stl_case'
    save_dir = '/home/quan/Desktop/tempary/application_pipe/openfoam_case'

    for groupTag in os.listdir(stl_dir):
        groupDir = os.path.join(stl_dir, groupTag)
        inOutletSetting_jsonFile = os.path.join(groupDir, 'inOutletSetting.json')
        with open(inOutletSetting_jsonFile, 'r') as f:
            inOutletSetting = json.load(f)

        openfoam_helper.create_openfoam_case(
            inOutletSetting, 
            save_dir=os.path.join(save_dir, groupTag)
        )

if __name__ == '__main__':
    main()  

