#include "SearchNode.h"

namespace SearchSpace
{
    SearchNode::SearchNode(LocXyz loc, SearchNode *parent)
        : m_loc(loc), m_parent(parent)
    {
        m_status = OnNone;
    }

    SearchNode::~SearchNode()
    {
    }
}