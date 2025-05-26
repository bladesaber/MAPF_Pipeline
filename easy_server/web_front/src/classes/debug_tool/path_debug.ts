import { distance } from "three/tsl"

export function generatePointObstacles_by_percentage(
    x_size: number, y_size: number, z_size: number, count: number, 
    pipes: {x: number, y: number, z: number, radius: number}[], 
    rescale_factor: number
) : number[][] {
  const obstacles: number[][] = []
  const usedPositions = new Set<string>() // Track used positions

  while (obstacles.length < count) {
    const x = Math.min(x_size, Math.max(0, Math.random() * x_size))
    const y = Math.min(y_size, Math.max(0, Math.random() * y_size))
    const z = Math.min(z_size, Math.max(0, Math.random() * z_size))

    // Check conflict with pipes
    let hasConflict = false
    for (const pipe of pipes) {
      const dx = x - pipe.x
      const dy = y - pipe.y
      const dz = z - pipe.z
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (distance <= pipe.radius) {
        hasConflict = true
        break
      }
    }
    if (hasConflict) continue

    // Create a unique key for this position
    const positionKey = `${x},${y},${z}`
    
    // Skip if position is already used
    if (usedPositions.has(positionKey)) {
      continue
    }

    obstacles.push([x * rescale_factor, y * rescale_factor, z * rescale_factor])
    usedPositions.add(positionKey)
  }

  return obstacles
}

export function generateSphereObstacles_by_percentage(
    x_length: number, y_length: number, z_length: number, 
    percentage: number, radius_min: number, radius_max: number,
    pipes: {x: number, y: number, z: number, radius: number}[]
) {
  const target_volume = x_length * y_length * z_length * percentage
  let obstacle_volume = 0.0

  const obstacles = []
  const usedPositions = new Set<string>() // Track used positions

  while (obstacle_volume < target_volume) {
    const x = Math.min(x_length, Math.max(0, Math.random() * x_length))
    const y = Math.min(y_length, Math.max(0, Math.random() * y_length))
    const z = Math.min(z_length, Math.max(0, Math.random() * z_length))

    // Create a unique key for this position
    const positionKey = `${x},${y},${z}`
    
    // Skip if position is already used
    if (usedPositions.has(positionKey)) {
      continue
    }

    const obstacle_radius = Math.random() * (radius_max - radius_min) + radius_min

    // Check conflict with pipes
    let hasConflict = false
    for (const pipe of pipes) {
      const dx = x - pipe.x
      const dy = y - pipe.y
      const dz = z - pipe.z
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (distance <= (pipe.radius + obstacle_radius)) {
        hasConflict = true
        break
      }
    }
    if (hasConflict) continue

    obstacle_volume += 4.0 / 3.0 * Math.PI * obstacle_radius * obstacle_radius * obstacle_radius
    obstacles.push({x, y, z, radius: obstacle_radius})
    usedPositions.add(positionKey)
  }

  return obstacles
}

export function generateSquaresObstacles_by_percentage(
    x_length: number, y_length: number, z_length: number, 
    size_min: number, size_max: number, percentage: number,
    pipes: {x: number, y: number, z: number, radius: number}[]
) {
  const target_volume = x_length * y_length * z_length * percentage
  let obstacle_volume = 0.0

  const obstacles = []
  const usedPositions = new Set<string>() // Track used positions
  
  while (obstacle_volume < target_volume) {
    const x = Math.min(x_length, Math.max(0, Math.random() * x_length))
    const y = Math.min(y_length, Math.max(0, Math.random() * y_length))
    const z = Math.min(z_length, Math.max(0, Math.random() * z_length))

    const positionKey = `${x},${y},${z}`
    if (usedPositions.has(positionKey)) {
      continue
    }

    const obstacle_size = Math.random() * (size_max - size_min) + size_min

    // Check conflict with pipes
    let hasConflict = false
    for (const pipe of pipes) {
      const dx = Math.abs(x - pipe.x)
      const dy = Math.abs(y - pipe.y)
      const dz = Math.abs(z - pipe.z)
      if (dx <= (pipe.radius + obstacle_size) 
          && dy <= (pipe.radius + obstacle_size) 
          && dz <= (pipe.radius + obstacle_size)) {
        hasConflict = true
        break
      }
    }
    if (hasConflict) continue

    obstacle_volume += obstacle_size * obstacle_size * obstacle_size
    obstacles.push({x, y, z, size: obstacle_size})
    usedPositions.add(positionKey)
  }

  return obstacles
}

export function generateRandomPipes(
    x_size: number, y_size: number, z_size: number, radius: number,  count: number, 
    other_pipes_dict: {x: number, y: number, z: number, radius: number}[],
    obstacle_dict: {x: number, y: number, z: number, radius: number}[],
    rescale_factor: number
): {
  x: number, y: number, z: number, radius: number, face: string, 
  vector_x: number, vector_y: number, vector_z: number, pipe_mode: "inlet" | "outlet" | "intermediate"
}[] {
  const min_spacing = Math.max(x_size, y_size, z_size) * 0.25;
  const faces = ['left', 'right', 'top', 'bottom', 'front', 'back'];
  let pipes: {
    x: number, y: number, z: number, radius: number, face: string, 
    vector_x: number, vector_y: number, vector_z: number, pipe_mode: "inlet" | "outlet" | "intermediate"
  }[] = [];
  
  // Generate inlet pipe
  const inlet_face = faces[Math.floor(Math.random() * faces.length)];
  let inlet_pipe;
  
  do {
    inlet_pipe = generatePipeOnFace(inlet_face, x_size, y_size, z_size, radius, true);
  } while (
    hasConflict(inlet_pipe, other_pipes_dict, min_spacing) ||
    hasConflict(inlet_pipe, obstacle_dict, min_spacing)
  );
  
  pipes.push({
    x: inlet_pipe.x * rescale_factor, 
    y: inlet_pipe.y * rescale_factor, 
    z: inlet_pipe.z * rescale_factor, 
    radius: inlet_pipe.radius, 
    face: inlet_face, 
    vector_x: inlet_pipe.vector_x, 
    vector_y: inlet_pipe.vector_y, 
    vector_z: inlet_pipe.vector_z,
    pipe_mode: 'inlet'
  });
  
  // Generate outlet pipes  
  for (let i = 1; i < count; i++) {
    let outlet_pipe;
    let attempts = 0;
    const max_attempts = 100;
    let found = false;

    while (attempts < max_attempts) {
      const random_face = faces[Math.floor(Math.random() * faces.length)];
      outlet_pipe = generatePipeOnFace(random_face, x_size, y_size, z_size, radius, false);
      attempts++;

      if (attempts >= max_attempts) {
        break
      }

      if (hasConflict(outlet_pipe, other_pipes_dict, min_spacing) || 
          hasConflict(outlet_pipe, pipes, min_spacing) ||
          hasConflict(outlet_pipe, obstacle_dict, min_spacing)) {
        continue;
      }

      found = true;
      break;
    }

    if (found) {
      pipes.push({
        x: outlet_pipe!.x, 
        y: outlet_pipe!.y, 
        z: outlet_pipe!.z, 
        radius: outlet_pipe!.radius, 
        face: outlet_pipe!.face, 
        vector_x: outlet_pipe!.vector_x, 
        vector_y: outlet_pipe!.vector_y, 
        vector_z: outlet_pipe!.vector_z,
        pipe_mode: 'outlet'
      });
    }
  }

  pipes.forEach(pipe => {
    pipe.x *= rescale_factor
    pipe.y *= rescale_factor
    pipe.z *= rescale_factor
    pipe.radius *= rescale_factor
  })
  
  return pipes;
}

function generatePipeOnFace(
  face: string, x_size: number, y_size: number, z_size: number, radius: number, 
  inlet: boolean,
) {
  let x, y, z;
  let vector_x, vector_y, vector_z;
  
  switch (face) {
    case 'left':
      x = 0;
      y = Math.min(y_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * y_size)));
      z = Math.min(z_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * z_size)));
      if (inlet) {
        vector_x = 1;
      } else {
        vector_x = -1;
      }
      vector_y = 0;
      vector_z = 0;
      break;
    case 'right':
      x = x_size;
      y = Math.min(y_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * y_size)));
      z = Math.min(z_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * z_size)));
      if (inlet) {
        vector_x = -1;
      } else {
        vector_x = 1;
      }
      vector_y = 0;
      vector_z = 0;
      break;
    case 'top':
      x = Math.min(x_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * x_size)));
      y = y_size;
      z = Math.min(z_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * z_size)));
      if (inlet) {
        vector_y = -1;
      } else {
        vector_y = 1;
      }
      vector_x = 0;
      vector_z = 0;
      break;
    case 'bottom':
      x = Math.min(x_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * x_size)));
      y = 0;
      z = Math.min(z_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * z_size)));
      if (inlet) {
        vector_y = 1;
      } else {
        vector_y = -1;
      }
      vector_x = 0;
      vector_z = 0;
      break;
    case 'front':
      x = Math.min(x_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * x_size)));
      y = Math.min(y_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * y_size)));
      z = z_size;
      if (inlet) {
        vector_z = -1;
      } else {
        vector_z = 1;
      }
      vector_x = 0;
      vector_y = 0;
      break;
    case 'back':
      x = Math.min(x_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * x_size)));
      y = Math.min(y_size - radius - 2, Math.max(radius + 2, Math.round(Math.random() * y_size)));
      z = 0;
      if (inlet) {
        vector_z = 1;
      } else {
        vector_z = -1;
      }
      vector_x = 0;
      vector_y = 0;
      break;
    default:
      throw new Error('Invalid face');
  }
  
  return {x, y, z, radius, face, vector_x, vector_y, vector_z};
}

function hasConflict(
    pipe: {x: number, y: number, z: number, radius: number, face?: string},
    other_pipes: {x: number, y: number, z: number, radius: number, face?: string}[],
    min_spacing: number
) {
  return other_pipes.some(other => {
    const dx = pipe.x - other.x;
    const dy = pipe.y - other.y;
    const dz = pipe.z - other.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    
    // For pipes in the same group, use min_spacing
    if (pipe.face && other.face) {
      return distance <= min_spacing;
    }
    
    // For other pipes, use sum of radii
    return distance <= (pipe.radius + other.radius) + 1e-6;
  });
}

export function generateTask(pipe_names: string[], radius: number, priority_count: number) {
  // Ensure we have enough pipes for each priority
  if (pipe_names.length < priority_count * 2) {
    throw new Error('Not enough pipe names for the requested priority count');
  }

  // Split pipe_names into priority_count parts
  const parts: string[][] = [];
  const minPipesPerPart = 2;

  const random_shuffled_pipe_names = pipe_names.sort(() => Math.random() - 0.5);
  const remainingPipes = [...random_shuffled_pipe_names];
  
  // First, ensure each part gets at least 2 pipes
  for (let i = 0; i < priority_count; i++) {
    parts.push([
      remainingPipes.shift()!,
      remainingPipes.shift()!
    ]);
  }
  
  // Distribute remaining pipes randomly
  while (remainingPipes.length > 0) {
    const randomPart = Math.floor(Math.random() * priority_count);
    parts[randomPart].push(remainingPipes.shift()!);
  }

  // Generate tasks for each part
  const tasks = parts.map(partPipes => {
    const priority_tasks = [];
    for (let j = 0; j < partPipes.length - 1; j++) {
      priority_tasks.push({
        start_name: partPipes[j],
        end_name: partPipes[j + 1],
        radius: radius
      });
    }
    return priority_tasks;
  });

  return tasks;
}