#include "conflict/NearTerminalAnalyzer.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace pipe_design::conflict {

NearTerminalAnalyzer::NearTerminalAnalyzer(
    std::unordered_map<topology::GroupId, std::vector<geom::Vec3>> terminals)
    : terminals_(std::move(terminals)) {}

void NearTerminalAnalyzer::clear() {
    ConflictAnalyzer::clear();
    candidates_.clear();
}

void NearTerminalAnalyzer::score(Conflict& /*conflict*/) const {
    throw std::logic_error(
        "NearTerminalAnalyzer::score(Conflict&): absolute distance is not a valid "
        "urgency; ranks are relative within each group's conflict set — use score_all()");
}

void NearTerminalAnalyzer::score(ConflictCluster& cluster) const {
    float m = -std::numeric_limits<float>::infinity();
    for (const Conflict& c : cluster.conflicts) {
        m = std::max(m, c.score);
    }
    cluster.score = m;
}

void NearTerminalAnalyzer::score_all() {
    candidates_.clear();
    if (conflicts_.empty()) {
        clusters_.clear();
        return;
    }

    // group → indices into conflicts_ (each conflict appears once per involved group)
    std::unordered_map<topology::GroupId, std::vector<std::size_t>> by_group;
    for (std::size_t i = 0; i < conflicts_.size(); ++i) {
        by_group[conflicts_[i].group_a].push_back(i);
        by_group[conflicts_[i].group_b].push_back(i);
    }

    // conflict index → (group → rank); missing group = not involved
    std::vector<std::unordered_map<topology::GroupId, std::size_t>> ranks(conflicts_.size());

    for (auto& gkv : by_group) {
        const topology::GroupId& gid = gkv.first;
        const auto& idxs = gkv.second;

        const auto tit = terminals_.find(gid);
        const auto git = graphs_.find(gid);
        if (tit == terminals_.end() || git == graphs_.end() || tit->second.empty()) {
            throw std::runtime_error(
                "NearTerminalAnalyzer::score_all: missing PathGraph or pipe terminals "
                "for group \"" +
                gid + "\"");
        }

        const PathGraph& g = git->second;
        const std::size_t m = idxs.size();

        // Resolve witness node per conflict on this group (must already be on graph)
        std::vector<NodeId> wit(m);
        for (std::size_t k = 0; k < m; ++k) {
            const Conflict& c = conflicts_[idxs[k]];
            const SampledPoint& sp = (c.group_a == gid) ? c.sample_a : c.sample_b;
            wit[k] = g.find_node(sp);
        }

        // Min rank across pipe terminals (Python HeadConflictSelector)
        std::vector<std::size_t> best_rank(m, m);
        std::vector<float> dists(m);
        std::vector<std::size_t> order(m);

        for (const geom::Vec3& tpos : tit->second) {
            const NodeId t = g.find_node(SampledPoint{tpos, 0.f});
            for (std::size_t k = 0; k < m; ++k) {
                dists[k] = g.shortest_path_length(t, wit[k]);
            }
            std::iota(order.begin(), order.end(), 0);  /// Fills the order container with sequential values starting from 0
            /// Indirect sort (also called an argsort)
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return dists[a] < dists[b];
            });
            for (std::size_t r = 0; r < m; ++r) {
                best_rank[order[r]] = std::min(best_rank[order[r]], r);
            }
        }

        for (std::size_t k = 0; k < m; ++k) {
            ranks[idxs[k]][gid] = best_rank[k];
        }
    }

    for (std::size_t i = 0; i < conflicts_.size(); ++i) {
        std::size_t sum = 0;
        bool any_zero = false;
        for (const auto& rkv : ranks[i]) {
            sum += rkv.second;
            if (rkv.second == 0) {
                any_zero = true;
            }
        }
        conflicts_[i].score = -static_cast<float>(sum);
        if (any_zero) {
            candidates_.push_back(&conflicts_[i]);
        }
    }

    for (ConflictCluster& cl : clusters_) {
        score(cl);
    }
}

const Conflict* NearTerminalAnalyzer::select_conflict() const {
    if (candidates_.empty()) {
        throw std::logic_error(
            "NearTerminalAnalyzer::select_conflict: no candidates found; use score_all()");
    }
    const Conflict* best = candidates_[0];
    for (const Conflict* c : candidates_) {
        if (c->score > best->score) {
            best = c;
        }
    }
    return best;
}

}  // namespace pipe_design::conflict
