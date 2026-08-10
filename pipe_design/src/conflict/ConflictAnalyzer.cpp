#include "conflict/ConflictAnalyzer.h"

#include "conflict/SampleKdTree.h"
#include "geom/Geometry.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pipe_design::conflict {

void ConflictAnalyzer::clear() {
    sampled_paths_.clear();
    graphs_.clear();
    conflicts_.clear();
    clusters_.clear();
}

void ConflictAnalyzer::analyze(const std::vector<const topology::Task*>& tasks) {
    clear();
    normalize(tasks);
    build_graphs();
    find_conflicts();
    find_clusters();
    score_all();
}

void ConflictAnalyzer::normalize(const std::vector<const topology::Task*>& tasks) {
    sampled_paths_.clear();
    if (grid_ == nullptr || plans_ == nullptr) {
        return;
    }

    const float min_step = 2.f * PathGraph::kDefaultEps;

    for (const topology::Task* task_ptr : tasks) {
        if (task_ptr == nullptr) {
            continue;
        }
        const topology::Task& task = *task_ptr;
        if (!task.succeeded()) {
            continue;
        }
        const topology::GroupTopologyPlan* gplan = plans_->find_group(task.group_id);
        if (gplan == nullptr) {
            continue;
        }
        const topology::TopologyEdge* edge = gplan->find_edge(task.edge_id);
        if (edge == nullptr || edge->radius <= 0.f) {
            continue;
        }

        const float radius = edge->radius;
        const float step = std::max(sample_step_factor_ * radius, min_step);

        SampledPath sp{task.group_id, task.task_id};
        const auto& cells = task.path.cells;
        if (cells.empty()) {
            continue;
        }

        // NOTE: Match chain_with_path occupancy:
        //   Chebyshev(cells[i],cells[i+1])==1 → grid neighbor: record endpoint only (sphere)
        //   Chebyshev>1 → any-angle hop: line-sample the capsule on that edge only
        // Do not chain samples from the last inserted point (corners must stay on the polyline).
        // Near-duplicates: skip, never replace — except path start/end which are always kept
        // (otherwise PathGraph can miss terminals and break the group tree).
        auto push_point = [&](const geom::Vec3& p, bool force = false) {
            if (!sp.samples.empty() &&
                geom::distance(sp.samples.back().center, p) < min_step) {
                if (force) {
                    sp.samples.back() = SampledPoint{p, radius};
                }
                return;
            }
            sp.samples.push_back(SampledPoint{p, radius});
        };

        push_point(grid_->cell_center(cells[0]), /*force=*/true);
        for (std::size_t i = 0; i + 1 < cells.size(); ++i) {
            const grid::CellCoord d = cells[i + 1] - cells[i];
            const int cheb = std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)});
            if (cheb == 0) {
                continue;
            }

            const geom::Vec3 b = grid_->cell_center(cells[i + 1]);
            if (cheb == 1) {
                push_point(b);
                continue;
            }

            const geom::Vec3 a = grid_->cell_center(cells[i]);
            const float len = geom::distance(a, b);
            const int n = std::max(1, static_cast<int>(std::ceil(len / step)));
            for (int k = 1; k <= n; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(n);
                push_point(a + (b - a) * t);
            }
        }
        push_point(grid_->cell_center(cells.back()), /*force=*/true);

        if (!sp.samples.empty()) {
            sampled_paths_[task.group_id].push_back(std::move(sp));
        }
    }
}

void ConflictAnalyzer::build_graphs() {
    graphs_.clear();
    for (const auto& kv : sampled_paths_) {
        PathGraph& g = graphs_[kv.first];
        for (const SampledPath& sp : kv.second) {
            g.add_sampled_path(sp);
        }
        const std::size_t ncomp = g.component_count();
        if (ncomp != 1) {
            throw std::runtime_error(
                "ConflictAnalyzer::build_graphs: group \"" + kv.first +
                "\" PathGraph has " + std::to_string(ncomp) +
                " connected component(s); expected exactly 1 "
                "(missing shared start/end merge?)");
        }
    }
}

void ConflictAnalyzer::find_conflicts() {
    conflicts_.clear();
    if (sampled_paths_.empty()) {
        return;
    }

    // Flatten per group once: points + task_ids stay stable for KD-tree pointers.
    struct GroupCloud {
        topology::GroupId id;
        std::vector<SampledPoint> points;
        std::vector<std::string> task_ids;
        float max_radius = 0.f;
        SampleKdTree tree;
    };

    std::vector<GroupCloud> clouds;
    clouds.reserve(sampled_paths_.size());
    for (const auto& kv : sampled_paths_) {
        GroupCloud cloud;
        cloud.id = kv.first;
        for (const SampledPath& sp : kv.second) {
            cloud.points.reserve(cloud.points.size() + sp.samples.size());
            cloud.task_ids.reserve(cloud.task_ids.size() + sp.samples.size());
            for (const SampledPoint& s : sp.samples) {
                cloud.points.push_back(s);
                cloud.task_ids.push_back(sp.task_id);
                cloud.max_radius = std::max(cloud.max_radius, s.radius);
            }
        }
        if (!cloud.points.empty()) {
            clouds.push_back(std::move(cloud));
        }
    }

    // Build one KD-tree per group (after all clouds exist — no further moves of points).
    for (GroupCloud& cloud : clouds) {
        cloud.tree.build(cloud.points);
    }

#if defined(_OPENMP)
    const int nthreads = std::max(1, omp_get_max_threads());
#else
    const int nthreads = 1;
#endif
    std::vector<std::vector<Conflict>> locals(static_cast<std::size_t>(nthreads));

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<Conflict>& out = locals[static_cast<std::size_t>(tid)];
        std::vector<std::size_t> hit_idx;
        std::vector<float> hit_dist;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
        for (std::ptrdiff_t gi = 0; gi < static_cast<std::ptrdiff_t>(clouds.size()); ++gi) {
            const GroupCloud& ca = clouds[static_cast<std::size_t>(gi)];
            for (std::size_t gj = static_cast<std::size_t>(gi) + 1; gj < clouds.size(); ++gj) {
                const GroupCloud& cb = clouds[gj];

                for (std::size_t ia = 0; ia < ca.points.size(); ++ia) {
                    const SampledPoint& sa = ca.points[ia];
                    cb.tree.radius_search(sa.center, sa.radius + cb.max_radius, hit_idx, hit_dist);
                    for (std::size_t h = 0; h < hit_idx.size(); ++h) {
                        const std::size_t ib = hit_idx[h];
                        const SampledPoint& sb = cb.points[ib];
                        const float need = sa.radius + sb.radius;
                        if (hit_dist[h] >= need) {
                            continue;
                        }
                        Conflict c;
                        c.group_a = ca.id;
                        c.group_b = cb.id;
                        c.task_a = ca.task_ids[ia];
                        c.task_b = cb.task_ids[ib];
                        c.sample_a = sa;
                        c.sample_b = sb;
                        c.deficit = need - hit_dist[h];
                        out.push_back(std::move(c));
                    }
                }
            }
        }
    }

    for (std::vector<Conflict>& loc : locals) {
        conflicts_.insert(conflicts_.end(),
                          std::make_move_iterator(loc.begin()),
                          std::make_move_iterator(loc.end()));
    }
}

void ConflictAnalyzer::find_clusters() {
    clusters_.clear();
    if (conflicts_.empty()) {
        return;
    }

    const std::size_t n = conflicts_.size();
    std::vector<std::size_t> parent(n);
    for (std::size_t i = 0; i < n; ++i) {
        parent[i] = i;
    }
    auto findp = [&](std::size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](std::size_t a, std::size_t b) {
        a = findp(a);
        b = findp(b);
        if (a != b) {
            parent[b] = a;
        }
    };

    // group → (node → conflict indices that witness on that node)
    std::unordered_map<topology::GroupId, std::unordered_map<NodeId, std::vector<std::size_t>>> at;
    for (std::size_t i = 0; i < n; ++i) {
        const Conflict& c = conflicts_[i];
        const NodeId na = graphs_.at(c.group_a).find_node(c.sample_a);
        const NodeId nb = graphs_.at(c.group_b).find_node(c.sample_b);
        at[c.group_a][na].push_back(i);
        at[c.group_b][nb].push_back(i);
    }

    // Same graph node → same cluster.
    for (auto& gkv : at) {
        for (auto& nkv : gkv.second) {
            const auto& idxs = nkv.second;
            for (std::size_t k = 1; k < idxs.size(); ++k) {
                unite(idxs[0], idxs[k]);
            }
        }
    }

    // Adjacent conflict-touched nodes on a PathGraph → merge clusters (conflict band).
    for (auto& gkv : at) {
        const PathGraph& g = graphs_.at(gkv.first);
        const auto& node_map = gkv.second;
        for (NodeId u = 0; u < g.node_count(); ++u) {
            const auto itu = node_map.find(u);
            if (itu == node_map.end()) {
                continue;
            }
            for (const PathEdge& e : g.neighbors(u)) {
                if (e.to < u) {
                    continue;
                }
                const auto itv = node_map.find(e.to);
                if (itv == node_map.end()) {
                    continue;
                }
                unite(itu->second.front(), itv->second.front());
            }
        }
    }

    std::unordered_map<std::size_t, std::size_t> root_to_ci;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = findp(i);
        const auto it = root_to_ci.find(r);
        if (it == root_to_ci.end()) {
            root_to_ci.emplace(r, clusters_.size());
            ConflictCluster cl;
            cl.add(conflicts_[i]);
            clusters_.push_back(std::move(cl));
        } else {
            clusters_[it->second].add(conflicts_[i]);
        }
    }
}

void ConflictAnalyzer::score(ConflictCluster& /*cluster*/) const {
    throw std::logic_error(
        "ConflictAnalyzer::score(ConflictCluster&): override in subclass");
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
