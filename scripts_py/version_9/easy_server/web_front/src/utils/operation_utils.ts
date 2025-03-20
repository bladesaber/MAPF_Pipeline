import * as THREE from 'three'
import { RootObj } from '../classes/store_obj.ts'
import { PointObj, VecPointObj } from '../classes/point_obj.ts'
import { LineObj, CircleObj, ArcObj, BsplineObj, TorusObj } from '../classes/line_obj.ts'
import { DiskObj } from '../classes/surface_obj.ts'
import { ElMessage } from 'element-plus'
import { sample_bspline, compute_path_tangent, sample_surface_points_from_circles } from './math_utils.ts'

export function generate_point(
    name: string, x: number, y: number, z: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.get_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    const point = new PointObj(name, x, y, z, 0.2, rootObj.get_scene());
    rootObj.add_obj(point);
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
    if (rootObj.get_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    const point = new VecPointObj(name, x, y, z, 0.2, vec_x, vec_y, vec_z, rootObj.get_scene());
    rootObj.add_obj(point);
    return {name: name, success: true};
}

export function generate_line_from_points(name: string, points: PointObj[], rootObj: RootObj): {name: string, success: boolean} {
    if (rootObj.get_obj(name)) {
        ElMessage.error('Point already exists');
        return {name: '', success: false};
    }
    const line = new LineObj(name, points, rootObj.get_scene());
    rootObj.add_obj(line);
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
    if (rootObj.get_obj(name)) {
        ElMessage.error('Circle already exists');
        return {name: '', success: false};
    }
    const circle = new CircleObj(name, x, y, z, radius, vec_x, vec_y, vec_z, rootObj.get_scene());
    rootObj.add_obj(circle);
    return {name: name, success: true};
}

export function generate_arc(
    name: string, x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, 
    start_angle: number, end_angle: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.get_obj(name)) {
        ElMessage.error('Arc already exists');
        return {name: '', success: false};
    }
    const arc = new ArcObj(name, x, y, z, radius, vec_x, vec_y, vec_z, start_angle, end_angle, rootObj.get_scene());
    rootObj.add_obj(arc);
    return {name: name, success: true};
}

export function generate_bspline_from_points(
    name: string, control_points: PointObj[], knots: number[], degree: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.get_obj(name)) {
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
    const bspline = new BsplineObj(name, control_points, knots, degree, rootObj.get_scene());
    rootObj.add_obj(bspline);
    return {name: name, success: true};
}

export function generate_bspline_from_input(
    name: string, control_points: number[][], knots: number[], degree: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.get_obj(name)) {
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

export function generate_skeleton(
    name: string, bspline: BsplineObj, percentages: number[], radius: number, rootObj: RootObj
): {names: string[], success: boolean} {
    const control_points = bspline.control_points.map(point => [point.x, point.y, point.z])
    const knots = bspline.knots
    const degree = bspline.degree
    const sample_points = sample_bspline(control_points, degree, knots, percentages)
    const tangents = compute_path_tangent(sample_points)
    const names = []
    for (let i = 0; i < sample_points.length; i++) {
        const point = sample_points[i]
        const tangent = tangents[i]
        const torus = new TorusObj(
            `${name}_skeleton_${i}`, point[0], point[1], point[2], radius, radius + 0.2, 
            tangent[0], tangent[1], tangent[2], rootObj.get_scene()
        )
        rootObj.add_obj(torus)
        names.push(torus.name)
    }
    return {names: names, success: true};
}

export function generate_disk(
    name: string, x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
): {name: string, success: boolean} {
    if (rootObj.get_obj(name)) {
        ElMessage.error('Disk already exists');
        return {name: '', success: false};
    }
    const disk = new DiskObj(name, x, y, z, radius, vec_x, vec_y, vec_z, rootObj.get_scene());
    rootObj.add_obj(disk);
    return {name: name, success: true};
}

export function generate_surface_points_from_skeletons(
    name: string, skeletons: TorusObj[], sample_num: number[], rootObj: RootObj
): {names: string[], success: boolean} {
    if (rootObj.get_obj(name)) {
        ElMessage.error('Point already exists');
        return {names: [], success: false};
    }
    const points = sample_surface_points_from_circles(
        skeletons.map(skeleton => [skeleton.x, skeleton.y, skeleton.z]),
        skeletons.map(skeleton => [skeleton.vec_x, skeleton.vec_y, skeleton.vec_z]),
        skeletons.map(skeleton => skeleton.radius),
        sample_num
    )
    const names = []
    for (let i = 0; i < points.length; i++) {
        const point = points[i]
        const point_obj = new PointObj(
            `${name}_skeleton_point_${i}`, point[0], point[1], point[2], 0.2, rootObj.get_scene()
        )
        rootObj.add_obj(point_obj)
        names.push(point_obj.name)
    }
    return {names: names, success: true};
}


