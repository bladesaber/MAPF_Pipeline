import * as THREE from 'three'
import { RootObj } from '../classes/store_obj.ts'
import { PointObj, LineObj, CircleObj, ArcObj, BsplineObj, DiskObj, TorusObj } from '../classes/geo_obj.ts'

export function generate_point(
    x: number, y: number, z: number, radius: number, rootObj: RootObj
) {
    // do something
}

export function generate_vector_point(
    x: number, y: number, z: number, radius: number, 
    vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
) {
    // do something
}

export function generate_line_from_points(points: PointObj[], rootObj: RootObj) {
    // do something
}

export function generate_line_from_input(points: number[][], rootObj: RootObj) {
    // do something
}

export function generate_circle(
    x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
) {
    // do something
}

export function generate_arc(
    x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, 
    start_angle: number, end_angle: number, rootObj: RootObj
) {
    // do something
}

export function generate_bspline_from_points(
    control_points: PointObj[], knots: number[], degree: number, rootObj: RootObj
) {
    if (knots.length == 0) {
        // Generate uniform knot vector from 0 to 1
        const numControlPoints = control_points.length;
        let knots = Array.from({length: numControlPoints - degree + 1}, (_, i) => i / (numControlPoints - degree));
        
        // Add degree multiplicity at ends
        const startKnots = Array(degree).fill(0);
        const endKnots = Array(degree).fill(1);
        knots = [...startKnots, ...knots, ...endKnots];
    }
    // do something
}

export function generate_bspline_from_input(control_points: number[][], knots: number[], degree: number, rootObj: RootObj) {
    if (knots.length == 0) {
        // Generate uniform knot vector from 0 to 1
        const numControlPoints = control_points.length;
        let knots = Array.from({length: numControlPoints - degree + 1}, (_, i) => i / (numControlPoints - degree));
        
        // Add degree multiplicity at ends
        const startKnots = Array(degree).fill(0);
        const endKnots = Array(degree).fill(1);
        knots = [...startKnots, ...knots, ...endKnots];
    }
    // do something
}

export function generate_disk(
    x: number, y: number, z: number, radius: number, vec_x: number, vec_y: number, vec_z: number, rootObj: RootObj
) {
    // do something
}
