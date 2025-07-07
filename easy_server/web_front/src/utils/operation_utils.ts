import * as THREE from 'three'
import { RootObj } from '../classes/postprocess_classes/store_obj.ts'
import { PointObj, VecPointObj, VecRadiusPointObj } from '../classes/postprocess_classes/point_obj.ts'
import { LineObj, CircleObj, ArcObj, BsplineObj, TorusObj } from '../classes/postprocess_classes/line_obj.ts'
import { DiskObj } from '../classes/postprocess_classes/surface_obj.ts'
import { ElMessage } from 'element-plus'
import { SplineSkeletonObj, CircleSurfacePointsObj } from '../classes/postprocess_classes/group_obj.ts'
import { BaseNode, GroupNode } from '../classes/postprocess_classes/base_obj.ts'
import { PipeShellObj, BsplineSurfaceObj } from '../classes/postprocess_classes/surface_obj.ts'

export function generate_point(
    name: string, x: number, y: number, z: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    const point = new PointObj(name, x, y, z, new BaseNode(name), rootObj);
    rootObj.add_objs([point]);
    return {name: name, success: true};
}

export function generate_point_from_line(
    name: string, line_name: string, percentage: number, rootObj: RootObj
): {name: string, success: boolean} {
    // todo something
    throw new Error('Not implemented');
}

export function generate_vector_point(
    name: string, x: number, y: number, z: number, vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    if (vec_x == 0 && vec_y == 0 && vec_z == 0){
        ElMessage.error('Vector non exists.');
        return {name: '', success: false};
    }
    const point = new VecPointObj(name, x, y, z, vec_x, vec_y, vec_z, new BaseNode(name), rootObj);
    rootObj.add_objs([point]);
    return {name: name, success: true};
}

export function generate_vec_radius_point(
    name: string, x: number, y: number, z: number, vec_x: number, vec_y: number, vec_z: number, radius: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    if (radius <= 0){
        ElMessage.error('Radius must be greater than 0.');
        return {name: '', success: false};
    }
    const point = new VecRadiusPointObj(name, x, y, z, vec_x, vec_y, vec_z, radius, new BaseNode(name), rootObj);
    rootObj.add_objs([point]);
    return {name: name, success: true};
}

export function generate_line_from_points(
    name: string, points: PointObj[], rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    points.forEach(point => {
        if (point.style != 'point' && point.style != 'vec_point'){
            ElMessage.error('Points must be point or vec_point');
            return {name: '', success: false};
        }
    })
    const line = new LineObj(name, points.map(point => point.node_style), new BaseNode(name), rootObj);
    rootObj.add_objs([line]);
    return {name: name, success: true};
}

export function generate_line_from_input(
    name: string, points: number[][], rootObj: RootObj): {name: string, success: boolean} {
    // todo something
    throw new Error('Not implemented');
}

export function generate_circle(
    name: string, x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Circle already exists');
        return {name: '', success: false};
    }
    if (vec_x == 0 && vec_y == 0 && vec_z == 0){
        ElMessage.error('Vector non exists.');
        return {name: '', success: false};
    }
    if (radius <= 0){
        ElMessage.error('Radius must be greater than 0.');
        return {name: '', success: false};
    }
    const circle = new CircleObj(name, x, y, z, radius, vec_x, vec_y, vec_z, new BaseNode(name), rootObj);
    rootObj.add_objs([circle]);
    return {name: name, success: true};
}

export function generate_arc(
    name: string, x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, 
    start_angle: number, end_angle: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Arc already exists');
        return {name: '', success: false};
    }
    if (vec_x == 0 && vec_y == 0 && vec_z == 0){
        ElMessage.error('Vector non exists.');
        return {name: '', success: false};
    }
    if (radius <= 0){
        ElMessage.error('Radius must be greater than 0.');
        return {name: '', success: false};
    }
    if (start_angle >= end_angle){
        ElMessage.error('Start angle must be less than end angle.');
        return {name: '', success: false};
    }
    const arc = new ArcObj(name, x, y, z, radius, vec_x, vec_y, vec_z, start_angle, end_angle, new BaseNode(name), rootObj);
    rootObj.add_objs([arc]);
    return {name: name, success: true};
}

export function generate_bspline_from_points(
    name: string, control_points: PointObj[], knots: number[], degree: number, sample_num: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Bspline already exists');
        return {name: '', success: false};
    }
    control_points.forEach(point => {
        if (point.style != 'point' && point.style != 'vec_point'){
            ElMessage.error('Control points must be point or vec_point');
            return {name: '', success: false};
        }
    })

    if (knots.length == 0) {
        // Generate uniform knot vector from 0 to 1
        const numControlPoints = control_points.length;
        knots = Array.from({length: numControlPoints - degree + 1}, (_, i) => i / (numControlPoints - degree));
        
        // Add degree multiplicity at ends
        const startKnots = Array(degree).fill(0);
        const endKnots = Array(degree).fill(1);
        knots = [...startKnots, ...knots, ...endKnots];
    }
    const bspline = new BsplineObj(
        name, control_points.map(point => point.node_style), knots, degree, new BaseNode(name), rootObj, sample_num
    );
    rootObj.add_objs([bspline]);
    return {name: name, success: true};
}

export function generate_bspline_from_input(
    name: string, control_points: number[][], knots: number[], degree: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Bspline already exists');
        return {name: '', success: false};
    }
    if (knots.length == 0) {
        // Generate uniform knot vector from 0 to 1
        const numControlPoints = control_points.length;
        let knots = Array.from({length: numControlPoints - degree + 1}, (_, i) => i / (numControlPoints - degree));
        
        // Add degree multiplicity at ends
        const startKnots = Array(degree).fill(0);
        const endKnots = Array(degree).fill(1);
        knots = [...startKnots, ...knots, ...endKnots];
    }
    // todo something
    throw new Error('Not implemented');
}

export function generate_circle_skeleton(
    name: string, bspline: BsplineObj, radius: number[], ts: number[], rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Skeleton already exists');
        return {name: '', success: false};
    }
    if (bspline.style != 'bspline'){
        ElMessage.error('Skeleton must be bspline');
        return {name: '', success: false};
    }
    const skeleton = new SplineSkeletonObj(name, bspline.node_style, radius, ts, new BaseNode(name), rootObj)
    rootObj.add_objs([skeleton])
    return {name: name, success: true};
}

export function generate_disk(
    name: string, x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Disk already exists');
        return {name: '', success: false};
    }
    const disk = new DiskObj(name, x, y, z, radius, vec_x, vec_y, vec_z, new BaseNode(name), rootObj);
    rootObj.add_objs([disk]);
    return {name: name, success: true};
}

export function generate_circle_surface_points_from_skeletons(
    name: string, torus_objs: TorusObj[], sample_num: number[], rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    torus_objs.forEach(torus_obj => {
        if (torus_obj.style != 'torus'){
            ElMessage.error('Skeleton nodes must be torus');
            return {name: '', success: false};
        }
    })

    const surface_points = new CircleSurfacePointsObj(
        name, torus_objs.map(torus_obj => torus_obj.node_style), sample_num, new BaseNode(name), rootObj
    )
    rootObj.add_objs([surface_points])
    return {name: name, success: true};
}

export function generate_pipe_shell(
    name: string, url: string, torus_objs: TorusObj[], rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Pipe shell already exists');
        return {name: '', success: false};
    }
    torus_objs.forEach(torus_obj => {
        if (torus_obj.style != 'torus'){
            ElMessage.error('Skeleton nodes must be torus');
            return {name: '', success: false};
        }
    })

    const pipe_shell = new PipeShellObj(
        name, url, torus_objs.map(torus_obj => torus_obj.node_style), new BaseNode(name), rootObj
    )
    // rootObj.add_objs([pipe_shell])
    return {name: name, success: true};
}

export function generate_bspline_surface(
    name: string, url: string, 
    boundary_objs: (BsplineObj|ArcObj|LineObj)[], guide_objs: (BsplineObj|LineObj)[], 
    rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.has_obj(name)) {
        ElMessage.error('Bspline surface already exists');
        return {name: '', success: false};
    }
    boundary_objs.forEach(boundary_obj => {
        if (
            boundary_obj.style != 'bspline' 
            && boundary_obj.style != 'line' 
            && boundary_obj.style != 'arc'
        ){
            ElMessage.error('Boundary nodes must be bspline or line or arc');
            return {name: '', success: false};
        }
    })
    guide_objs.forEach(guide_obj => {
        if (
            guide_obj.style != 'bspline' 
            && guide_obj.style != 'line'
        ){
            ElMessage.error('Guide nodes must be bspline or line');
            return {name: '', success: false};
        }
    })
    
    const bspline_surface = new BsplineSurfaceObj(
        name, url, 
        boundary_objs.map(boundary_obj => boundary_obj.node_style), 
        guide_objs.map(guide_obj => guide_obj.node_style), 
        new BaseNode(name), rootObj
    )
    rootObj.add_objs([bspline_surface])
    return {name: name, success: true};
}