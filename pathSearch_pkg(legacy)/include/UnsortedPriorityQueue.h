#ifndef UNSORTED_PRIORITY_QUEUE_H
#define UNSORTED_PRIORITY_QUEUE_H

#include "BaseHead.h"
#include "SearchNode.h"

#define UNSORTED_ARRAY_SIZE 3000

namespace QueueSpace
{

    class UnsortedPriorityQueue
    {
    public:
        UnsortedPriorityQueue()
        {
            m_nextFreeNode = 0;
        }

        ~UnsortedPriorityQueue() = default;

        inline bool is_empty()
        {
            return m_nextFreeNode == 0;
        }
        void push(SearchSpace::SearchNode *node);
        void remove(SearchSpace::SearchNode *node);
        SearchSpace::SearchNode *pop();

    private:
        int m_nextFreeNode;
        SearchSpace::SearchNode *m_nodeArray[UNSORTED_ARRAY_SIZE];
    };
}

#endif // UNSORTED_PRIORITY_QUEUE_H