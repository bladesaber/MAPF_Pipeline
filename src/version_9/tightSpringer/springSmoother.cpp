//
// Created by admin123456 on 2023/8/31.
//

#include "tightSpringer/springSmoother.h"

/*
namespace TightSpringNameSpace {

    bool SpringerSmooth_Runner::addEdge_structorToPlane_valueShift(
            size_t planeIdx, size_t structorIdx, std::string xyzTag, double shiftValue, double kSpring, double weight
    ) {
        Springer_Structor *structor = structor_NodeMap[structorIdx];
        Springer_Plane *plane = plane_NodeMap[planeIdx];

        auto *edge = new SpringPoseEdge_ValueShift<SpringVertexStructor, SpringVertexPlane>(
                xyzTag, shiftValue, kSpring
        );
        edge->setVertex(0, structor->vertex);
        edge->setVertex(1, plane->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    };

    bool SpringerSmooth_Runner::addEdge_connectorToStruct_valueShift(
            size_t connectorIdx, size_t structorIdx, std::string xyzTag, double shiftValue, double kSpring,
            double weight
    ) {
        Springer_Connector *connector = connector_NodeMap[connectorIdx];
        Springer_Structor *structor = structor_NodeMap[structorIdx];

        auto *edge = new SpringPoseEdge_ValueShift<SpringVertexCell, SpringVertexStructor>(xyzTag, shiftValue, kSpring);
        edge->setVertex(0, connector->vertex);
        edge->setVertex(1, structor->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_connectorToStruct_radiusFixed(
            size_t connectorIdx, size_t structorIdx, std::string xyzTag, double radius, double kSpring, double weight
    ) {
        Springer_Connector *connector = connector_NodeMap[connectorIdx];
        Springer_Structor *structor = structor_NodeMap[structorIdx];

        auto *edge = new SpringPoseEdge_RadiusFixed<SpringVertexCell, SpringVertexStructor>(xyzTag, radius, kSpring);
        edge->setVertex(0, connector->vertex);
        edge->setVertex(1, structor->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_connectorToStruct_poseFixed(
            size_t connectorIdx, size_t structorIdx, std::string xyzTag, double shapeX, double shapeY, double shapeZ,
            double kSpring, double weight
    ) {
        Springer_Connector *connector = connector_NodeMap[connectorIdx];
        Springer_Structor *structor = structor_NodeMap[structorIdx];

        auto *edge = new SpringPoseEdge_PoseFixed(shapeX, shapeY, shapeZ, kSpring);
        edge->setVertex(0, connector->vertex);
        edge->setVertex(1, structor->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToConnector_elasticBand(
            size_t cellIdx, size_t connectorIdx, double kSpring, double weight
    ) {
        Springer_Cell *cell = cell_NodeMap[cellIdx];
        Springer_Connector *connector = connector_NodeMap[connectorIdx];

        auto *edge = new SpringConEdge_ElasticBand(kSpring);
        edge->setVertex(0, cell->vertex);
        edge->setVertex(1, connector->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToCell_elasticBand(
            size_t cellIdx0, size_t cellIdx1, double kSpring, double weight
    ) {
        Springer_Cell *cell0 = cell_NodeMap[cellIdx0];
        Springer_Cell *cell1 = cell_NodeMap[cellIdx1];

        auto *edge = new SpringConEdge_ElasticBand(kSpring);
        edge->setVertex(0, cell0->vertex);
        edge->setVertex(1, cell1->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToConnector_kinematicPoint(
            size_t cellIdx, size_t connectorIdx, double vecX, double vecY, double vecZ, double targetValue,
            bool fromCell, double kSpring, double weight
    ) {
        Springer_Cell *cell = cell_NodeMap[cellIdx];
        Springer_Connector *connector = connector_NodeMap[connectorIdx];

        Eigen::Vector3d orientation = Eigen::Vector3d(vecX, vecY, vecZ);
        auto *edge = new SpringConEdge_KinematicPoint(orientation, targetValue, kSpring);
        if (fromCell) {
            edge->setVertex(0, cell->vertex);
            edge->setVertex(1, connector->vertex);
        } else {
            edge->setVertex(0, connector->vertex);
            edge->setVertex(1, cell->vertex);
        }

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToConnector_kinematicSegment(
            size_t connectorIdx, size_t cellIdx0, size_t cellIdx1,
            double targetValue, bool fromConnector, double kSpring, double weight
    ) {
        Springer_Connector *connector = connector_NodeMap[connectorIdx];
        Springer_Cell *cell0 = cell_NodeMap[cellIdx0];
        Springer_Cell *cell1 = cell_NodeMap[cellIdx1];

        auto *edge = new SpringConEdge_KinematicSegment(targetValue, kSpring);
        if (fromConnector) {
            edge->setVertex(0, connector->vertex);
            edge->setVertex(1, cell0->vertex);
            edge->setVertex(2, cell1->vertex);
        } else {
            edge->setVertex(0, cell0->vertex);
            edge->setVertex(1, cell1->vertex);
            edge->setVertex(2, connector->vertex);
        }

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToCell_kinematicSegment(
            size_t cellIdx0, size_t cellIdx1, size_t cellIdx2, double targetValue, double kSpring, double weight
    ) {
        Springer_Cell *cell0 = cell_NodeMap[cellIdx0];
        Springer_Cell *cell1 = cell_NodeMap[cellIdx1];
        Springer_Cell *cell2 = cell_NodeMap[cellIdx2];

        auto *edge = new SpringConEdge_KinematicSegment(targetValue, kSpring);
        edge->setVertex(0, cell0->vertex);
        edge->setVertex(1, cell1->vertex);
        edge->setVertex(2, cell2->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_structorToPlane_planeRepel(
            size_t structorIdx, size_t planeIdx,
            std::string planeTag, std::string compareTag,
            std::vector<std::tuple<double, double, double>> conflict_xyzs,
            double bound_shift, double kSpring, double weight
    ) {
        Springer_Structor *structor = structor_NodeMap[structorIdx];
        Springer_Plane *plane = plane_NodeMap[planeIdx];

        auto *edge = new SpringForceEdge_structorToPlaneRepel(
                planeTag, compareTag, conflict_xyzs, bound_shift + structor->shell_radius, kSpring
        );
        edge->setVertex(0, structor->vertex);
        edge->setVertex(1, plane->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToPlane_planeRepel(
            size_t cellIdx, size_t planeIdx,
            std::string planeTag, std::string compareTag,
            double bound_shift, double kSpring, double weight
    ) {
        Springer_Cell *cell = cell_NodeMap[cellIdx];
        Springer_Plane *plane = plane_NodeMap[planeIdx];

        auto *edge = new SpringForceEdge_CellToPlaneRepel(
                planeTag, compareTag, cell->radius + bound_shift, kSpring
        );
        edge->setVertex(0, cell->vertex);
        edge->setVertex(1, plane->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToCell_shapeRepel(
            size_t cellIdx0, size_t cellIdx1, double bound_shift, double kSpring, double weight
    ) {
        Springer_Cell *cell0 = cell_NodeMap[cellIdx0];
        Springer_Cell *cell1 = cell_NodeMap[cellIdx1];

        auto *edge = new SpringForceEdge_CellToCell_ShapeRepel(
                cell0->radius + cell1->radius + bound_shift, kSpring
        );
        edge->setVertex(0, cell0->vertex);
        edge->setVertex(1, cell1->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_cellToStructor_shapeRepel(
            size_t cellIdx, size_t structorIdx,
            double shapeX, double shapeY, double shapeZ,
            double bound_shift, double kSpring, double weight
    ) {
        Springer_Cell *cell = cell_NodeMap[cellIdx];
        Springer_Structor *structor = structor_NodeMap[structorIdx];

        auto *edge = new SpringForceEdge_CellToStructor_ShapeRepel(
                shapeX, shapeY, shapeZ, cell->radius + bound_shift, kSpring
        );
        edge->setVertex(0, cell->vertex);
        edge->setVertex(1, structor->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_structorToStructor_shapeRepel(
            size_t structorIdx0, size_t structorIdx1,
            double shapeX_0, double shapeY_0, double shapeZ_0,
            double shapeX_1, double shapeY_1, double shapeZ_1,
            double bound_shift, double kSpring, double weight
    ) {
        Springer_Structor *structor0 = structor_NodeMap[structorIdx0];
        Springer_Structor *structor1 = structor_NodeMap[structorIdx1];

        auto *edge = new SpringForceEdge_StructorToStructor_ShapeRepel(
                shapeX_0, shapeY_0, shapeZ_0, shapeX_1, shapeY_1, shapeZ_1, bound_shift, kSpring
        );
        edge->setVertex(0, structor0->vertex);
        edge->setVertex(1, structor1->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_minVolume(
            size_t minPlaneIdx, size_t maxPlaneIdx, double scale, double kSpring, double weight
    ) {
        Springer_Plane *plane_min = plane_NodeMap[minPlaneIdx];
        Springer_Plane *plane_max = plane_NodeMap[maxPlaneIdx];

        auto *edge = new SpringTarEdge_MinVolume(scale, kSpring);
        edge->setVertex(0, plane_min->vertex);
        edge->setVertex(1, plane_max->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_structor_poseCluster(
            size_t structorIdx, double scale, double kSpring, double weight
    ) {
        Springer_Structor *structor = structor_NodeMap[structorIdx];
        auto *edge = new SpringTarEdge_PoseCluster<SpringVertexStructor>(scale, kSpring);
        edge->setVertex(0, structor->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_connector_poseCluster(
            size_t connectorIdx, double scale, double kSpring, double weight
    ) {
        Springer_Connector *connector = connector_NodeMap[connectorIdx];
        auto *edge = new SpringTarEdge_PoseCluster<SpringVertexCell>(scale, kSpring);
        edge->setVertex(0, connector->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

    bool SpringerSmooth_Runner::addEdge_minAxes(
            size_t minPlaneIdx, size_t maxPlaneIdx, std::string xyzTag, double scale, double kSpring, double weight
    ) {
        Springer_Plane *plane_min = plane_NodeMap[minPlaneIdx];
        Springer_Plane *plane_max = plane_NodeMap[maxPlaneIdx];

        auto *edge = new SpringTarEdge_MinAxes(xyzTag, scale, kSpring);
        edge->setVertex(0, plane_min->vertex);
        edge->setVertex(1, plane_max->vertex);

        Eigen::Matrix<double, 1, 1> information;
        information.fill(weight);
        edge->setInformation(information);
        bool success = optimizer->addEdge(edge);
        return success;
    }

}

 */