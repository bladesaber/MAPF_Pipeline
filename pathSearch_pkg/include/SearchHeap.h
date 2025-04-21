#ifndef SEARCH_HEAP_H
#define SEARCH_HEAP_H

#include "BaseHead.h"
#include "SearchNode.h"
#include "UnsortedPriorityQueue.h"

#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#define NUM_BUCKETS 10000

namespace QueueSpace
{
    class SearchHeap
    {
    public:
        SearchHeap() = default;
        virtual ~SearchHeap() = default;

        virtual void reset() = 0;
        virtual void push(SearchSpace::SearchNode *node) = 0;
        virtual SearchSpace::SearchNode *pop() = 0;
        virtual bool contains(SearchSpace::SearchNode *node) = 0;
        virtual SearchSpace::SearchNode *find_node(SearchSpace::SearchNode *node) = 0;
        virtual void update(SearchSpace::SearchNode *node, double lastCost) = 0;
        virtual bool is_empty() = 0;
        virtual void remove(SearchSpace::SearchNode *node) = 0;
    };

    class BoostHeap : public SearchHeap
    {
    public:
        BoostHeap() = default;

        ~BoostHeap() override
        {
            reset();
        }

        void reset() override;
        void push(SearchSpace::SearchNode *node) override;
        SearchSpace::SearchNode *pop() override;
        bool contains(SearchSpace::SearchNode *node) override;
        SearchSpace::SearchNode *find_node(SearchSpace::SearchNode *node) override;
        void update(SearchSpace::SearchNode *node, double lastCost) override;
        bool is_empty() override;
        void remove(SearchSpace::SearchNode *node) override;

    protected:
        boost::unordered_set<
            SearchSpace::SearchNode *,
            SearchSpace::SearchNode::NodeHasher,
            SearchSpace::SearchNode::NodeEqual>
            m_table;

        boost::heap::pairing_heap<
            SearchSpace::SearchNode *,
            boost::heap::compare<SearchSpace::SearchNode::NodeCompare>>
            m_openHeap;
    };

    class BucketPriorityQueue : public SearchHeap
    {
    public:
        BucketPriorityQueue()
        {
            reset();
        }

        ~BucketPriorityQueue() override
        {
            for (auto bin : m_bin)
            {
                delete bin;
            }
            m_bin.clear();
            m_table.clear();
        }

        void reset() override;

        bool is_empty() override
        {
            return m_numNodesTracked == 0;
        }

        void update(SearchSpace::SearchNode *node, double lastCost) override;
        void push(SearchSpace::SearchNode *node) override;
        SearchSpace::SearchNode *pop() override;
        bool contains(SearchSpace::SearchNode *node) override;
        SearchSpace::SearchNode *find_node(SearchSpace::SearchNode *node) override;

        inline void remove(SearchSpace::SearchNode *node) override
        {
            throw std::runtime_error("remove not implemented");
        }

    protected:
        int m_lowestNonEmptyBin;
        int m_numNodesTracked;
        std::vector<UnsortedPriorityQueue *> m_bin;

        inline size_t get_bin_index(double cost)
        {
            assert((cost >= 0.0 && cost <= 999.0) && "cost is out of range");
            return static_cast<size_t>(cost * 10.0);
        }

        boost::unordered_set<
            SearchSpace::SearchNode *,
            SearchSpace::SearchNode::NodeHasher,
            SearchSpace::SearchNode::NodeEqual>
            m_table;
    };

}

#endif
