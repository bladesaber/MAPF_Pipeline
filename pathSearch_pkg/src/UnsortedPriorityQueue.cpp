#include "UnsortedPriorityQueue.h"

#define ALL_NODES_EQUAL_COST	// ~40% faster and still results in optimal paths in virtually all cases

namespace QueueSpace
{
    void UnsortedPriorityQueue::push(SearchSpace::SearchNode *node)
    {
        m_nodeArray[m_nextFreeNode++] = node;
    }

    void UnsortedPriorityQueue::remove(SearchSpace::SearchNode *node)
    {
        for (int i = 0; i < m_nextFreeNode; ++i)
        {
            if (m_nodeArray[i] == node)
            {
                // Delete off Open list (put last node where this one was)
                m_nodeArray[i] = m_nodeArray[--m_nextFreeNode];
                return;
            }
        }
    }

    SearchSpace::SearchNode *UnsortedPriorityQueue::pop()
    {
    #ifdef ALL_NODES_EQUAL_COST
        return m_nodeArray[--m_nextFreeNode];

    #else
        // Find lowest f-cost node
        double bestCost = m_nodeArray[0]->f_cost;
        unsigned int bestIndex = 0;

        for (unsigned int i = 1; i < m_nextFreeNode; ++i)
        {
            if (m_nodeArray[i]->f_cost < bestCost)
            {
                bestCost = m_nodeArray[i]->f_cost;
                bestIndex = i;
            }
        }
        SearchSpace::SearchNode *node = m_nodeArray[bestIndex];
        // Delete off Open list (put last node where this one was)
        m_nodeArray[bestIndex] = m_nodeArray[--m_nextFreeNode];
        return node;
    #endif
    }

}
