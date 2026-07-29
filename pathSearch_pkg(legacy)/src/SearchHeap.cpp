#include "SearchHeap.h"

namespace QueueSpace
{
    void BoostHeap::reset()
    {
        m_openHeap.clear();
        m_table.clear();  // Just clear the table without deleting nodes
    }

    void BoostHeap::push(SearchSpace::SearchNode *node)
    {
        node->m_openHandle = m_openHeap.push(node);
        m_table.insert(node);
        node->m_status = SearchSpace::SearchNodeStatus::OnOpen;
    }

    void BoostHeap::remove(SearchSpace::SearchNode *node)
    {
        m_openHeap.erase(node->m_openHandle);
        m_table.erase(node);
    }

    SearchSpace::SearchNode *BoostHeap::pop()
    {
        SearchSpace::SearchNode *node = m_openHeap.top();
        m_openHeap.pop();
        node->m_status = SearchSpace::SearchNodeStatus::OnClosed;
        return node;
    }

    bool BoostHeap::contains(SearchSpace::SearchNode *node)
    {
        return m_table.find(node) != m_table.end();
    }

    SearchSpace::SearchNode *BoostHeap::find_node(SearchSpace::SearchNode *node)
    {
        auto it = m_table.find(node);
        if (it == m_table.end())
        {
            return nullptr;
        }
        return *it;
    }

    void BoostHeap::update(SearchSpace::SearchNode *node, double lastCost)
    {
        m_openHeap.increase(node->m_openHandle);
    }

    bool BoostHeap::is_empty()
    {
        return m_openHeap.empty();
    }

    void BucketPriorityQueue::reset()
    { 
        for (auto bin : m_bin)
            {
                delete bin;
            }
            m_bin.clear();
        m_bin.clear();
        m_table.clear();

        m_lowestNonEmptyBin = NUM_BUCKETS;
        m_numNodesTracked = 0;

        m_bin.reserve(NUM_BUCKETS);
        for (int i = 0; i < NUM_BUCKETS; ++i)
        {
            m_bin[i] = new UnsortedPriorityQueue();
        }
    }

    void BucketPriorityQueue::push(SearchSpace::SearchNode *node)
    {
        int bin_index = get_bin_index(node->f_cost);
        m_bin[bin_index]->push(node);
        m_numNodesTracked++;
        if (bin_index < m_lowestNonEmptyBin)
        {
            m_lowestNonEmptyBin = bin_index;
        }

        m_table.insert(node);
        node->m_status = SearchSpace::SearchNodeStatus::OnOpen;
    }

    SearchSpace::SearchNode *BucketPriorityQueue::pop()
    {
        SearchSpace::SearchNode *node = m_bin[m_lowestNonEmptyBin]->pop();
        m_numNodesTracked--;

        if (m_numNodesTracked > 0)
        {
            // Find the next non-empty bin
            for (m_lowestNonEmptyBin; m_lowestNonEmptyBin < NUM_BUCKETS; m_lowestNonEmptyBin++)
            {
                if (!m_bin[m_lowestNonEmptyBin]->is_empty())
                {
                    break;
                }
            }
        }
        else
        {
            m_lowestNonEmptyBin = NUM_BUCKETS;
        }

        node->m_status = SearchSpace::SearchNodeStatus::OnClosed;
        return node;
    }

    void BucketPriorityQueue::update(SearchSpace::SearchNode *node, double lastCost)
    {
        int bin_index = get_bin_index(lastCost);
        m_bin[bin_index]->remove(node);

        bin_index = get_bin_index(node->f_cost);
        m_bin[bin_index]->push(node);

        if (bin_index < m_lowestNonEmptyBin)
        {
            m_lowestNonEmptyBin = bin_index;
        }
    }

    bool BucketPriorityQueue::contains(SearchSpace::SearchNode *node)
    {
        return m_table.find(node) != m_table.end();
    }

    SearchSpace::SearchNode *BucketPriorityQueue::find_node(SearchSpace::SearchNode *node)
    {
        auto it = m_table.find(node);
        if (it == m_table.end())
        {
            return nullptr;
        }
        return *it;
    }

}