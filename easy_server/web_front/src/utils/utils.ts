export function setVisableDict(
  active_key: string, visable_dict: any, single_mode: boolean = false, set_value: boolean = true
) {
  if (single_mode) {
    Object.keys(visable_dict).forEach(key => {
      if (key !== active_key) {
        visable_dict[key as keyof typeof visable_dict].value = false
      }
    })
  }
  visable_dict[active_key as keyof typeof visable_dict].value = set_value
}

export function pcd2plane_dist(pcd: number[][], norm: number[], center: number[]): number[] {
    // Calculate vector from center to each point
    const vec = pcd.map(point => [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2]
    ]);
    
    // Calculate norm magnitude
    const norm_mag = Math.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
    
    // Calculate dot product and divide by norm magnitude
    return vec.map(v => (v[0] * norm[0] + v[1] * norm[1] + v[2] * norm[2]) / norm_mag);
}