#ifndef CONSTRAINT_SELECTOR_H
#define CONSTRAINT_SELECTOR_H

#include "BaseHead.h"
#include "NetworkUtils.h"

namespace SearchSpace
{
    class ConstraintSelector
    {
    public:
        ConstraintSelector(ConflictSet &conflict_set)
            : m_conflict_set(conflict_set) {}

        ~ConstraintSelector() = default;

        virtual void select_constraint(std::map<std::string, MathSpace::NetworkGraph> &graph_map, Constraint &constraint) = 0;

    protected:
        ConflictSet &m_conflict_set;
    };

    class NearEndConstraintSelector : public ConstraintSelector
    {
    public:
        NearEndConstraintSelector(ConflictSet &conflict_set)
            : ConstraintSelector(conflict_set) {}

        ~NearEndConstraintSelector() = default;

        void select_constraint(std::map<std::string, MathSpace::NetworkGraph> &graph_map, Constraint &constraint) override;
    };
}

#endif
