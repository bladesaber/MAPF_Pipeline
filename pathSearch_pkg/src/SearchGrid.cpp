#include "SearchGrid.h"

namespace GridSpace
{
    bool SearchGrid::is_in_grid(LocXyz loc) const
    {
        return loc.x >= 0 && loc.x < m_x_size 
            && loc.y >= 0 && loc.y < m_y_size 
            && loc.z >= 0 && loc.z < m_z_size;
    }

    bool SearchGrid::is_move_valid(LocXyz loc, double threshold, int8_t dir_x, int8_t dir_y, int8_t dir_z) const
    {
        if (!is_in_grid(loc + LocXyz{dir_x, dir_y, dir_z})) return false;
        if (std::get<0>(is_obstacle(loc + LocXyz{dir_x, dir_y, dir_z}, threshold))) return false;

        // In test, it is too strict, so we remove the check
        // if (std::get<0>(is_obstacle(loc + LocXyz{dir_x, 0, 0}, threshold))) return false;
        // if (std::get<0>(is_obstacle(loc + LocXyz{0, dir_y, 0}, threshold))) return false;
        // if (std::get<0>(is_obstacle(loc + LocXyz{0, 0, dir_z}, threshold))) return false;
        return true;
    }

    bool SearchGrid::is_light_of_sight(LocXyz loc1, LocXyz loc2, double threshold) const
    {
        // First check if both points are within grid bounds
        if (!is_in_grid(loc1) || !is_in_grid(loc2)) {
            return false;
        }

        // 3D Bresenham's Line Algorithm with proper error terms
        int x0 = loc1.x, y0 = loc1.y, z0 = loc1.z;
        int x1 = loc2.x, y1 = loc2.y, z1 = loc2.z;

        int dx = abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
        int dy = abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
        int dz = abs(z1 - z0), sz = (z0 < z1) ? 1 : -1;

        // Find the dominant axis for proper stepping
        int max_delta = std::max({dx, dy, dz});
        if (max_delta == 0) return true;  // Points are the same

        double err_xy = (double)dx / max_delta - (double)dy / max_delta;
        double err_xz = (double)dx / max_delta - (double)dz / max_delta;

        LocXyz current{x0, y0, z0};
        while (true) {
            // Check if the move to the next point is valid
            if (x0 != x1 || y0 != y1 || z0 != z1) {
                int8_t next_dx = 0, next_dy = 0, next_dz = 0;
                
                // Calculate step direction
                double e2_xy = 2 * err_xy;
                double e2_xz = 2 * err_xz;
                
                if (e2_xy > -1 && e2_xz > -1 && x0 != x1) {
                    next_dx = sx;
                }
                else if (e2_xy < 1 && e2_xz > -1 && y0 != y1) {
                    next_dy = sy;
                }
                else if (z0 != z1) {
                    next_dz = sz;
                }
                
                // Check if the move is valid
                if (!is_move_valid(current, threshold, next_dx, next_dy, next_dz)) {
                    return false;
                }
                
                // Update position and errors
                if (next_dx != 0) {
                    x0 += next_dx;
                    err_xy -= (double)dy / max_delta;
                    err_xz -= (double)dz / max_delta;
                }
                else if (next_dy != 0) {
                    y0 += next_dy;
                    err_xy += (double)dx / max_delta;
                    err_xz -= (double)dz / max_delta;
                }
                else if (next_dz != 0) {
                    z0 += next_dz;
                    err_xz += (double)dx / max_delta;
                    err_xy -= (double)dy / max_delta;
                }
                else {
                    break;  // No more moves possible
                }
                
                current = LocXyz{x0, y0, z0};
            }
            else {
                break;  // Reached the target
            }
        }
        return true;
    }

    void StandardGrid::set_obstacle(LocXyz loc, double weight)
    {   
        m_grid[loc.z * m_slice_size + loc.y * m_x_size + loc.x] = true;     
    }

    void StandardGrid::clear_obstacle(LocXyz loc)
    {
        m_grid[loc.z * m_slice_size + loc.y * m_x_size + loc.x] = false;
    }

    std::tuple<bool, double> StandardGrid::is_obstacle(LocXyz loc, double threshold) const
    {
        return std::make_tuple(m_grid[loc.z * m_slice_size + loc.y * m_x_size + loc.x], 0.0);
    }
    
    void DistanceGrid::set_obstacle(LocXyz loc, double distance)
    {
        assert(distance >= 0.0 && distance <= 500.0);
        m_distances[loc.z * m_slice_size + loc.y * m_x_size + loc.x] = distance * m_scale_factor;
    }
    
    void DistanceGrid::clear_obstacle(LocXyz loc)
    {
        // m_distances[loc.z * m_slice_size + loc.y * m_x_size + loc.x] = std::numeric_limits<uint16_t>::infinity();
        throw std::runtime_error("Not implemented");
    }

    std::tuple<bool, double> DistanceGrid::is_obstacle(LocXyz loc, double threshold) const
    {
        // TODO: Very important, it should be consist with definition of ConflictBaseSearchNode::detect_conflict_between_groups
        double distance = m_distances[loc.z * m_slice_size + loc.y * m_x_size + loc.x] / m_scale_factor;
        return std::make_tuple(distance < threshold, distance);
    }
    
    void BitGrid::set_obstacle(LocXyz loc, double weight)
    {
        size_t bit_pos = get_bit_position(loc);
        size_t word_index = bit_pos / 64;
        size_t bit_offset = bit_pos % 64;
        
        // set the bit to 1
        m_bits[word_index] |= (1ULL << bit_offset);
    }

    void BitGrid::clear_obstacle(LocXyz loc)
    {
        size_t bit_pos = get_bit_position(loc);
        size_t word_index = bit_pos / 64;
        size_t bit_offset = bit_pos % 64;
        
        // clear the bit to 0
        m_bits[word_index] &= ~(1ULL << bit_offset);
    }
    
    std::tuple<bool, double> BitGrid::is_obstacle(LocXyz loc, double threshold) const
    {
        size_t bit_pos = get_bit_position(loc);
        size_t word_index = bit_pos / 64;
        size_t bit_offset = bit_pos % 64;

        // check if the bit is set to 1
        return std::make_tuple((m_bits[word_index] & (1ULL << bit_offset)) != 0, 0.0);
    }
    
    std::tuple<bool, double> SparseGrid::is_obstacle(LocXyz loc, double threshold) const
    {
        // If coordinate is not in the map, it's free space
        return std::make_tuple(m_obstacles.find(loc) != m_obstacles.end(), 0.0);
    }
    
    void SparseGrid::set_obstacle(LocXyz loc, double weight)
    {
        m_obstacles[loc] = true;
    }
    
    void SparseGrid::clear_obstacle(LocXyz loc)
    {
        auto it = m_obstacles.find(loc);
        if (it != m_obstacles.end())
        {
            m_obstacles.erase(it);
        }
    }

    void TreeGrid::set_obstacle(LocXyz loc, double radius)
    {
        m_data.emplace_back(loc, radius);
    }

    void TreeGrid::clear_obstacle(LocXyz loc)
    {
        m_data.erase(std::remove_if(m_data.begin(), m_data.end(), [loc](const auto& item) {
            return std::get<0>(item) == loc;
        }), m_data.end());
    }

    std::tuple<bool, double> TreeGrid::is_obstacle(LocXyz loc, double threshold) const
    {
        assert(threshold > 0.0 && "threshold must be non-negative");

        if (m_data.size() == 0) {
            return std::make_tuple(false, 0.0);
        }

        std::vector<int> indices;
        std::vector<double> distances;
        m_tree.search_radius(loc, m_max_radius + threshold, indices, distances);

        double min_distance = 1e6;
        bool is_obstacle = false;
        for (int i = 0; i < indices.size(); ++i)
        {

            // if (loc.x == 49 && loc.y == 62 && loc.z == 64) {
            //     std::cout << "[C++]: TreeGrid: is_obstacle: loc: " 
            //         << std::get<0>(m_data[indices[i]]) 
            //         << " distance: " << distances[i]
            //         << " radius: " << std::get<1>(m_data[indices[i]])
            //         << " threshold: " << threshold
            //         << " obs_radius: " << std::get<1>(m_data[indices[i]])
            //         << std::endl;
            // }

            // TODO: Very important, it should be consist with definition of ConflictBaseSearchNode::detect_conflict_between_groups
            if (distances[i] < threshold + std::get<1>(m_data[indices[i]]))
            {
                min_distance = std::min(min_distance, distances[i]);
                is_obstacle = true;
            }
        }
        return std::make_tuple(is_obstacle, min_distance);
    }
}