#ifndef SEARCH_GRID_H
#define SEARCH_GRID_H

#include "BaseHead.h"
#include "NeighbourTree.h"

namespace GridSpace
{
    class SearchGrid
    {
    public:
        SearchGrid(uint16_t x_size, uint16_t y_size, uint16_t z_size, double grid_resolution)
            : m_x_size(x_size), m_y_size(y_size), m_z_size(z_size), m_grid_resolution(grid_resolution)
        {
        }

        virtual ~SearchGrid() = default;

        virtual void set_obstacle(LocXyz loc, double weight) = 0;
        virtual void clear_obstacle(LocXyz loc) = 0;
        virtual std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold) const = 0;
        virtual void preprocess() = 0;
        
        bool is_in_grid(LocXyz loc) const;
        std::tuple<bool, double> is_constrained(LocXyz loc, double threshold) const
        {
            return std::make_tuple(false, 0.0);
        }

        // bool is_in_grid(LocXyz loc) const;  // Assumed boundary of grid always exists and environment is in positive space
        bool is_move_valid(LocXyz loc, double threshold, int8_t dir_x, int8_t dir_y, int8_t dir_z) const;
        bool is_light_of_sight(LocXyz loc1, LocXyz loc2, double threshold) const;

        uint16_t get_x_size() const { return m_x_size; }
        uint16_t get_y_size() const { return m_y_size; }
        uint16_t get_z_size() const { return m_z_size; }
        double get_grid_resolution() const { return m_grid_resolution; }

    protected:
        uint16_t m_x_size, m_y_size, m_z_size;
        double m_grid_resolution;
    };

    class StandardGrid : public SearchGrid
    {
    public:
        StandardGrid(uint16_t x_size, uint16_t y_size, uint16_t z_size, double grid_resolution)
            : SearchGrid(x_size, y_size, z_size, grid_resolution)
        {
            m_slice_size = x_size * y_size;
            m_grid = std::vector<bool>(x_size * y_size * z_size, false);
        }

        ~StandardGrid() = default;

        void preprocess() override {};
        void set_obstacle(LocXyz loc, double weight=0.0) override;
        void clear_obstacle(LocXyz loc) override;
        std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold=0.0) const override;

    protected:
        uint32_t m_slice_size; // Size of each XY slice
        std::vector<bool> m_grid;
    };

    class DistanceGrid : public SearchGrid
    {
        // Store distance to closest obstacle for each cell
    public:
        DistanceGrid(uint16_t x_size, uint16_t y_size, uint16_t z_size, double grid_resolution)
            : SearchGrid(x_size, y_size, z_size, grid_resolution)
        {
            m_slice_size = x_size * y_size;
            m_distances.resize(x_size * y_size * z_size, 500);
        }

        ~DistanceGrid() = default;

        void preprocess() override {};
        void set_obstacle(LocXyz loc, double distance) override;
        void clear_obstacle(LocXyz loc) override;
        std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold) const override;

        void set_obstacles(std::vector<std::vector<int>> locs, std::vector<double> distances)
        {
            for (int i = 0; i < locs.size(); ++i)
            {
                this->set_obstacle(LocXyz(locs[i][0], locs[i][1], locs[i][2]), distances[i]);
            }
        }

    protected:
        const double m_scale_factor = 100.0;
        uint32_t m_slice_size;  // Size of each XY slice
        std::vector<uint16_t> m_distances;
    };

    class BitGrid : public SearchGrid
    {
    public:
        BitGrid(uint16_t x_size, uint16_t y_size, uint16_t z_size, double grid_resolution)
            : SearchGrid(x_size, y_size, z_size, grid_resolution)
        {
            // Calculate size needed for each XY slice (in bits)
            m_slice_size = static_cast<unsigned int>(std::ceil((m_x_size * m_y_size) / 64.0)) * 64;

            // Allocate bit array
            size_t total_bits = m_slice_size * m_z_size;
            m_bits.resize((total_bits + 63) / 64, 0); // Round up to nearest multiple of 64
        }

        ~BitGrid() = default;

        void preprocess() override {};
        void set_obstacle(LocXyz loc, double weight) override;
        void clear_obstacle(LocXyz loc) override;
        std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold) const override;

    protected:
        unsigned int m_slice_size;
        std::vector<uint64_t> m_bits; // Bit array for obstacle representation

        // Helper function to calculate bit position
        inline size_t get_bit_position(LocXyz loc) const
        {
            // calculate the bit position in the bit array
            return (
                static_cast<size_t>(loc.z) * m_slice_size 
                + static_cast<size_t>(loc.y) * m_x_size 
                + static_cast<size_t>(loc.x));
        }
    };

    class SparseGrid : public SearchGrid
    {
    public:
        SparseGrid(uint16_t x_size, uint16_t y_size, uint16_t z_size, double grid_resolution)
            : SearchGrid(x_size, y_size, z_size, grid_resolution)
        {
            // Initialize with empty obstacle map
            m_obstacles.reserve(100000); // Reserve space for efficiency
        }

        ~SparseGrid() = default;

        void preprocess() override {};
        std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold) const override;
        void set_obstacle(LocXyz loc, double weight) override;
        void clear_obstacle(LocXyz loc) override;

    protected:
        std::unordered_map<LocXyz, bool, CoordHash> m_obstacles;
    };

    class TreeGrid : public SearchGrid
    {
    public:
        TreeGrid(uint16_t x_size, uint16_t y_size, uint16_t z_size, double grid_resolution)
            : SearchGrid(x_size, y_size, z_size, grid_resolution)
        {
        }

        ~TreeGrid() = default;

        void set_obstacle(LocXyz loc, double radius) override;
        void clear_obstacle(LocXyz loc) override;
        std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold) const override;

        void preprocess() override
        {
            m_tree = MathSpace::KdTree(); // reset tree
            m_max_radius = 0.0;
            if (m_data.size() == 0) {
                return;
            }

            m_tree.init_tree(m_data.size());
            for (int i = 0; i < m_data.size(); ++i)
            {
                m_tree.add_leaf(i, std::get<0>(m_data[i]));
                m_max_radius = std::max(m_max_radius, std::get<1>(m_data[i]));
            }
            m_tree.build_tree();
        }

    protected:
        std::vector<std::tuple<LocXyz, double>> m_data;
        MathSpace::KdTree m_tree;
        double m_max_radius;
    };

    class SearchGridSet : public SearchGrid
    {
    public:
        SearchGridSet(const SearchGrid *static_grid)
            : SearchGrid(static_grid->get_x_size(),
                        static_grid->get_y_size(),
                        static_grid->get_z_size(),
                        static_grid->get_grid_resolution()),
              m_dynamic_obs_grid(static_grid->get_x_size(), static_grid->get_y_size(), 
                                static_grid->get_z_size(), static_grid->get_grid_resolution()),
              m_constraint_table(static_grid->get_x_size(), static_grid->get_y_size(), 
                                static_grid->get_z_size(), static_grid->get_grid_resolution())
        {
            m_static_obs_grid = static_grid;
        }

        ~SearchGridSet() = default;

        std::tuple<bool, double> is_obstacle(LocXyz loc, double threshold) const override
        {
            bool is_obstacle0 = false;
            double min_distance0 = 0.0;
            std::tie(is_obstacle0, min_distance0) = m_static_obs_grid->is_obstacle(loc, threshold);

            bool is_obstacle1 = false;
            double min_distance1 = 0.0;
            std::tie(is_obstacle1, min_distance1) = m_dynamic_obs_grid.is_obstacle(loc, threshold);
            
            return std::make_tuple(is_obstacle0 || is_obstacle1, std::min(min_distance0, min_distance1));
        }

        std::tuple<bool, double> is_constrained(LocXyz loc, double threshold) const
        {
            return m_constraint_table.is_obstacle(loc, threshold);
        }

        void set_obstacle(LocXyz loc, double weight) override
        {
            m_dynamic_obs_grid.set_obstacle(loc, weight);
        }

        void set_constraint(LocXyz loc, double weight)
        {
            m_constraint_table.set_obstacle(loc, weight);
        }

        void clear_obstacle(LocXyz loc) override
        {
            throw std::runtime_error("Not implemented");
        }

        void preprocess() override
        {
            m_dynamic_obs_grid.preprocess();
            m_constraint_table.preprocess();
        }

    protected:
        const SearchGrid *m_static_obs_grid;
        TreeGrid m_dynamic_obs_grid;
        TreeGrid m_constraint_table;
    };
}

#endif // SEARCH_GRID_H