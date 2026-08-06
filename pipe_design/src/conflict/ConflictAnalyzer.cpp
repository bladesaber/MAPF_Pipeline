#include "conflict/ConflictAnalyzer.h"

#include <algorithm>

namespace pipe_design::conflict {

void ConflictAnalyzer::clear() {
    sampled_paths_.clear();
    graphs_.clear();
    conflicts_.clear();
    clusters_.clear();
}

void ConflictAnalyzer::analyze(const std::vector<topology::Task>& tasks) {
    clear();
    normalize(tasks);
    build_graphs();
    find_conflicts();
    find_clusters();
    score_all();
}

void ConflictAnalyzer::normalize(const std::vector<topology::Task>& /*tasks*/) {}

void ConflictAnalyzer::build_graphs() {}

void ConflictAnalyzer::find_conflicts() {}

void ConflictAnalyzer::find_clusters() {}

void ConflictAnalyzer::score(ConflictCluster& cluster) const {
    for (Conflict& c : cluster.conflicts) {
        score(c);
    }
    float m = 0.f;
    for (const Conflict& c : cluster.conflicts) {
        m = std::max(m, c.score);
    }
    cluster.score = m;
}

void ConflictAnalyzer::score_all() {
    for (Conflict& c : conflicts_) {
        score(c);
    }
    for (ConflictCluster& cl : clusters_) {
        score(cl);
    }
}

const ConflictCluster* ConflictAnalyzer::select_cluster() const {
    if (clusters_.empty()) {
        return nullptr;
    }
    const ConflictCluster* best = &clusters_.front();
    for (std::size_t i = 1; i < clusters_.size(); ++i) {
        if (clusters_[i].score > best->score) {
            best = &clusters_[i];
        }
    }
    return best;
}

}  // namespace pipe_design::conflict
