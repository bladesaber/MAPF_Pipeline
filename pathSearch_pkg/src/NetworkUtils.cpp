#include "NetworkUtils.h"
#include <boost/graph/dijkstra_shortest_paths.hpp>

namespace MathSpace
{
    NetworkGraph::NetworkGraph()
        : m_graph()  // Explicitly initialize the graph
        , m_weight_map(boost::get(boost::edge_weight, m_graph))
    {
    }

    NetworkGraph::~NetworkGraph()
    {
    }

    NetworkGraph::Vertex NetworkGraph::get_or_create_vertex(const LocXyz& loc)
    {
        auto it = m_vertex_map.find(loc);
        if (it != m_vertex_map.end()) {
            return it->second;
        }
        
        Vertex v = boost::add_vertex(m_graph);
        m_vertex_map[loc] = v;
        m_reverse_map[v] = loc;
        return v;
    }

    NetworkGraph::Vertex& NetworkGraph::get_vertex(const LocXyz& loc)
    {
        auto it = m_vertex_map.find(loc);
        if (it != m_vertex_map.end()) {
            return it->second;
        }
        throw std::runtime_error("Vertex not found");
    }

    void NetworkGraph::add_edge(const LocXyz& start, const LocXyz& end, const double& weight)
    {
        Vertex v1 = get_or_create_vertex(start);
        Vertex v2 = get_or_create_vertex(end);
        
        Edge e;
        bool inserted;
        boost::tie(e, inserted) = boost::add_edge(v1, v2, m_graph);
        if (inserted) {
            m_weight_map[e] = weight;
        }
    }

    double NetworkGraph::get_shortest_path_length(const LocXyz& start, const LocXyz& end)
    {
        std::vector<Vertex> predecessors(boost::num_vertices(m_graph));
        std::vector<double> distances(boost::num_vertices(m_graph));

        Vertex start_vertex = get_vertex(start);
        Vertex end_vertex = get_vertex(end);

        boost::dijkstra_shortest_paths(
            m_graph, start_vertex,
            boost::predecessor_map(
                boost::make_iterator_property_map(predecessors.begin(), boost::get(boost::vertex_index, m_graph))
            ).distance_map(
                boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, m_graph))
            ).weight_map(m_weight_map)
        );

        return distances[end_vertex];
    }

    void NetworkGraph::get_shortest_path(const LocXyz& start, const LocXyz& end, std::vector<LocXyz>& path)
    {
        std::vector<Vertex> predecessors(boost::num_vertices(m_graph));
        std::vector<double> distances(boost::num_vertices(m_graph));

        Vertex start_vertex = get_vertex(start);
        Vertex end_vertex = get_vertex(end);

        // Run Dijkstra's algorithm
        boost::dijkstra_shortest_paths(m_graph, start_vertex,
            boost::predecessor_map(
                boost::make_iterator_property_map(predecessors.begin(), boost::get(boost::vertex_index, m_graph))
            ).distance_map(
                boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, m_graph))
            ).weight_map(m_weight_map)
        );

        // Reconstruct the path
        Vertex current = end_vertex;
        while (current != start_vertex) {
            path.push_back(m_reverse_map[current]);
            current = predecessors[current];
        }
        path.push_back(m_reverse_map[start_vertex]);
        
        // Reverse the path to get start-to-end order
        std::reverse(path.begin(), path.end());
    }

    size_t NetworkGraph::get_degree(const LocXyz& loc)
    {
        Vertex v = get_vertex(loc);
        return boost::degree(v, m_graph);
    }

    void NetworkGraph::get_neighbours(const LocXyz& loc, std::vector<LocXyz>& neighbours)
    {
        Vertex v = get_vertex(loc);
        boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;
        boost::tie(ai, ai_end) = boost::adjacent_vertices(v, m_graph);
        neighbours.clear();
        neighbours.reserve(boost::num_vertices(m_graph));
        for (auto it = ai; it != ai_end; ++it) {
            neighbours.push_back(m_reverse_map[*it]);
        }
    }

    double NetworkGraph::get_edge_weight(const LocXyz& start, const LocXyz& end)
    {
        Vertex v1 = get_vertex(start);
        Vertex v2 = get_vertex(end);
        Edge e;
        bool found;
        boost::tie(e, found) = boost::edge(v1, v2, m_graph);
        if (!found) {
            throw std::runtime_error("Edge not found");
        }
        return m_weight_map[e];
    }
    
    void NetworkGraph::get_terminal_nodes(std::vector<LocXyz>& terminals)
    {
        terminals.clear();
        for (auto it = boost::vertices(m_graph); it.first != it.second; ++it.first) {
            if (boost::degree(*it.first, m_graph) == 1) {
                terminals.push_back(m_reverse_map[*it.first]);
            }
        }
    }
}