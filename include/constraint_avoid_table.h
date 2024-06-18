//
// Created by admin123456 on 2024/6/17.
//

#ifndef MAPF_PIPELINE_CONSTRAINT_AVOID_TABLE_H
#define MAPF_PIPELINE_CONSTRAINT_AVOID_TABLE_H

#include "common.h"

using namespace std;

class ConflictAvoidTable {
public:
    ConflictAvoidTable() {}

    ~ConflictAvoidTable() {}

    void insert(size_t flag) {
        if (table.find(flag) == table.end()) {
            table[flag] = 0;
        }
        table[flag] += 1;
    }

    int get_num_of_conflict(size_t flag) const {
        const auto &iter = table.find(flag);
        if (iter == table.end()) {
            return 0;
        }
        return iter->second;
    }

    void clear() { table.clear(); }

    map<size_t, int> get_data() const {
        return table;
    }

    int get_size() const {
        return table.size();
    }

private:
    map<size_t, int> table;
};

#endif //MAPF_PIPELINE_CONSTRAINT_AVOID_TABLE_H
