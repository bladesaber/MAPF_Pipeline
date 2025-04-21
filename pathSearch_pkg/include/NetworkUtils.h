#ifndef NETWORK_UTILS_H
#define NETWORK_UTILS_H

#include "BaseHead.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <unordered_map>

namespace MathSpace
{
    class NetworkGraph
    {
    public:
        typedef boost::adjacency_list<
            boost::vecS,          // Edge storage (vecS for fast traversal)
            boost::vecS,          // Vertex storage
            boost::undirectedS,   // Undirected graph
            boost::no_property,   // Vertex properties (none here)
            boost::property<boost::edge_weight_t, double>  // Edge weight property
        > Graph;
        typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
        typedef boost::graph_traits<Graph>::edge_descriptor Edge;
        typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;

    public:
        NetworkGraph();
        ~NetworkGraph();

        void add_edge(const LocXyz& start, const LocXyz& end, const double& weight);
        double get_shortest_path_length(const LocXyz& start, const LocXyz& end);
        void get_shortest_path(const LocXyz& start, const LocXyz& end, std::vector<LocXyz>& path);

        size_t get_degree(const LocXyz& loc);
        void get_neighbours(const LocXyz& loc, std::vector<LocXyz>& neighbours);
        double get_edge_weight(const LocXyz& start, const LocXyz& end);
        void get_terminal_nodes(std::vector<LocXyz>& terminals);

        size_t get_node_count()
        {
            return boost::num_vertices(m_graph);
        }

        size_t get_edge_count()
        {
            return boost::num_edges(m_graph);
        }

        void get_nodes(std::vector<LocXyz>& nodes) const
        {
            for (auto &vertex : m_reverse_map)
            {
                nodes.push_back(vertex.second);
            }
        }

    protected:
        Graph m_graph;
        WeightMap m_weight_map;
        std::unordered_map<LocXyz, Vertex, CoordHash> m_vertex_map;
        std::unordered_map<Vertex, LocXyz> m_reverse_map;

        Vertex get_or_create_vertex(const LocXyz& loc);
        Vertex& get_vertex(const LocXyz& loc);
    };

}

#endif