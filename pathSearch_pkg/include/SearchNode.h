#ifndef SEARCH_NODE_H
#define SEARCH_NODE_H

#include "BaseHead.h"
#include <boost/heap/pairing_heap.hpp>

namespace SearchSpace
{

    enum SearchNodeStatus
    {
        OnNone,
        OnOpen,
        OnClosed
    };

    
    class SearchNode
    {
    public:
        struct NodeHasher
        {
            size_t operator()(const SearchNode *n) const
            {
                // TODO: assume size no more than 500
                return (size_t)n->m_loc.x * 500 + n->m_loc.y * 500 + n->m_loc.z;
            }
        };

        struct NodeEqual
        {
            bool operator()(const SearchNode *lhs, const SearchNode *rhs) const
            {
                // return (lhs == rhs);
                return lhs->m_loc == rhs->m_loc;
            }
        };

        struct NodeCompare
        {
            // returns true if n1 > n2 (note -- this gives us *min*-heap).
            bool operator()(const SearchNode *lhs, const SearchNode *rhs) const
            {
                if (lhs->f_cost == rhs->f_cost)
                {
                    return (lhs->f_cost - lhs->g_cost) > (rhs->f_cost - rhs->g_cost);
                }
                return lhs->f_cost > rhs->f_cost;
            }
        };

    public:
        LocXyz m_loc;
        double g_cost;
        double f_cost;
        double conflict_distance = 0.0;

        SearchNode *m_parent;
        SearchNodeStatus m_status;

        SearchNode(LocXyz loc, SearchNode *parent = nullptr);

        ~SearchNode();

        boost::heap::pairing_heap<
            SearchNode *, 
            boost::heap::compare<SearchNode::NodeCompare>
            >::handle_type m_openHandle;
    };

}

#endif
