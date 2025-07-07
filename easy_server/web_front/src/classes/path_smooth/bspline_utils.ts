export class BsplineUtils {
    /**
     * Compute B-spline basis function
     */
    static compute_Bip(t: number, i: number, degree: number, knot: number[]): number {
        if (degree === 0) {
            if (knot[i] <= t && t < knot[i + 1]) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        let c1 = 0.0;
        if (knot[i + degree] !== knot[i]) {
            c1 = (t - knot[i]) / (knot[i + degree] - knot[i]);
            c1 = c1 * BsplineUtils.compute_Bip(t, i, degree - 1, knot);
        }

        let c2 = 0.0;
        if (knot[i + degree + 1] !== knot[i + 1]) {
            c2 = (knot[i + degree + 1] - t) / (knot[i + degree + 1] - knot[i + 1]);
            c2 = c2 * BsplineUtils.compute_Bip(t, i + 1, degree - 1, knot);
        }

        return c1 + c2;
    }

    /**
     * Compute B-spline matrix
     */
    static compute_mat(ts: number[], pcd_num: number, degree: number, knot: number[], fill_last_one: boolean = true): number[][] {
        const b_mat: number[][] = [];
        
        for (let j = 0; j < ts.length; j++) {
            b_mat[j] = [];
            for (let i = 0; i < pcd_num; i++) {
                b_mat[j][i] = BsplineUtils.compute_Bip(ts[j], i, degree, knot);
            }
        }
        
        if (fill_last_one) {
            b_mat[ts.length - 1][pcd_num - 1] = 1.0;
        }
        
        return b_mat;
    }

    /**
     * Create knot configuration
     */
    static create_knot_cfg(pcd_num: number, degree: number, left_clamp: boolean = true, right_clamp: boolean = true): { knot: number[], fill_last_one: boolean } {
        let knot: number[];
        let fill_last_one: boolean;

        if (left_clamp && right_clamp) {
            const inner_knots = BsplineUtils.linspace(0.0, 1.0, pcd_num - degree + 1);
            const left_padding = new Array(degree).fill(0.0);
            const right_padding = new Array(degree).fill(1.0);
            knot = [...left_padding, ...inner_knots, ...right_padding];
            fill_last_one = true;
        } else if (left_clamp && !right_clamp) {
            const inner_knots = BsplineUtils.linspace(0.0, 1.0, pcd_num + 1);
            const left_padding = new Array(degree).fill(0.0);
            knot = [...left_padding, ...inner_knots];
            fill_last_one = false;
        } else if (!left_clamp && right_clamp) {
            const inner_knots = BsplineUtils.linspace(0.0, 1.0, pcd_num + 1);
            const right_padding = new Array(degree).fill(1.0);
            knot = [...inner_knots, ...right_padding];
            fill_last_one = true;
        } else {
            knot = BsplineUtils.linspace(0.0, 1.0, pcd_num + degree + 1);
            fill_last_one = false;
        }

        return { knot, fill_last_one };
    }

    /**
     * Create parameter values for sampling
     */
    static create_ts(
        sample_num: number, 
        degree: number, 
        knot: number[], 
        left_clamp: boolean = true, 
        right_clamp: boolean = true,
        left_split_num?: number, 
        right_split_num?: number
    ): number[] {
        if (left_split_num === undefined) {
            left_split_num = degree;
        }
        if (left_split_num < degree) {
            throw new Error('left_split_num must be >= degree');
        }

        if (right_split_num === undefined) {
            right_split_num = degree;
        }
        if (right_split_num < degree) {
            throw new Error('right_split_num must be >= degree');
        }

        let ts: number[];
        if (left_clamp && right_clamp) {
            ts = BsplineUtils.linspace(0.0, 1.0, sample_num);
        } else if (left_clamp && !right_clamp) {
            ts = BsplineUtils.linspace(0.0, knot[knot.length - right_split_num - 1], sample_num);
        } else if (!left_clamp && right_clamp) {
            ts = BsplineUtils.linspace(knot[left_split_num], 1.0, sample_num);
        } else {
            // Restrict to range where the curve is defined
            ts = BsplineUtils.linspace(knot[left_split_num], knot[knot.length - right_split_num - 1], sample_num);
        }
        
        return ts;
    }

    /**
     * Compute parameter values for given control points and target points
     */
    static compute_ts(control_pcd: number[][], pcd: number[][], degree: number, left_clamp: boolean, right_clamp: boolean): number[] {
        const knot_cfg = BsplineUtils.create_knot_cfg(control_pcd.length, degree, left_clamp, right_clamp);

        const cost_fun = (ts: number[]): number => {
            const mat = BsplineUtils.compute_mat(
                ts, control_pcd.length, degree, knot_cfg.knot, knot_cfg.fill_last_one
            );
            const pcd_tmp = BsplineUtils.matrixMultiply(mat, control_pcd);
            const cost = BsplineUtils.sumSquaredDifferences(pcd, pcd_tmp);
            return cost;
        };

        // Initialize with 0.5 for all parameters
        const initial_ts = new Array(pcd.length).fill(0.5);
        
        // For now, return initial values. In a real implementation, you would use an optimization library
        // like scipy.optimize.minimize equivalent for JavaScript
        return initial_ts;
    }

    /**
     * Helper function to create linearly spaced array (equivalent to np.linspace)
     */
    private static linspace(start: number, end: number, num: number): number[] {
        const result: number[] = [];
        const step = (end - start) / (num - 1);
        
        for (let i = 0; i < num; i++) {
            result.push(start + i * step);
        }
        
        return result;
    }

    /**
     * Helper function to multiply matrix with array of points
     */
    private static matrixMultiply(mat: number[][], points: number[][]): number[][] {
        const result: number[][] = [];
        
        for (let i = 0; i < mat.length; i++) {
            result[i] = [];
            for (let j = 0; j < points[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < mat[i].length; k++) {
                    sum += mat[i][k] * points[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        return result;
    }

    /**
     * Helper function to compute sum of squared differences
     */
    private static sumSquaredDifferences(pcd1: number[][], pcd2: number[][]): number {
        let sum = 0;
        for (let i = 0; i < pcd1.length; i++) {
            for (let j = 0; j < pcd1[i].length; j++) {
                const diff = pcd1[i][j] - pcd2[i][j];
                sum += diff * diff;
            }
        }
        return sum;
    }
}

