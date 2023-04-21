#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "utils.h"
#include "kdtreeWrapper.h"

class AgentInfo{
public:
    size_ut agentIdx;
    double radius;

    bool isConflict = false;
    ConstrainType firstConflict;
    double firstConflictLength;

    std::set<std::tuple<int, int, int>> conflictSet;

    AgentInfo(){};
    AgentInfo(size_ut agentIdx, double radius):agentIdx(agentIdx), radius(radius){};
    AgentInfo(AgentInfo* rhs){
        copy(rhs);
    }

    std::shared_ptr<std::vector<ConstrainType>> constrains;
    std::shared_ptr<DetailPath> detailPath;
    std::shared_ptr<KDTreeWrapper> pathTree;

    void update_DetailPath_And_Tree(const DetailPath& path){
        detailPath = nullptr;
        detailPath = std::make_shared<DetailPath>(path);

        pathTree = nullptr;
        pathTree = std::make_shared<KDTreeWrapper>();
        pathTree->insertPath3D(*detailPath, radius);

        // KDTreeRes res;
        // pathTree->nearest(3.0, 2.0, 0.0, res);
        // std::cout << "x: " << res.x << " y:" << res.y << " z:" << res.z <<std::endl;
    }

    void update_Constrains(const std::vector<ConstrainType>& new_constrains){
        this->constrains = nullptr;
        this->constrains = std::make_shared<std::vector<ConstrainType>>(new_constrains);
    }

    void copy(AgentInfo* rhs){
        this->agentIdx = rhs->agentIdx;
        this->radius = rhs->radius;

        this->constrains = std::shared_ptr<std::vector<ConstrainType>>(rhs->constrains);
        this->detailPath = std::shared_ptr<DetailPath>(rhs->detailPath);
        this->pathTree = std::shared_ptr<KDTreeWrapper>(rhs->pathTree);
    }

    std::vector<ConstrainType> getConstrains(){
        // just for python test
        return *constrains;
    }

    DetailPath getDetailPath(){
        // just for python test
        return *detailPath;
    }

    void release(){
        constrains = nullptr;
        detailPath = nullptr;
        pathTree = nullptr;
        conflictSet.clear();
    }

    void info(){
        std::cout << "AgentIdx: " << agentIdx << std::endl;
        std::cout << "  Detail Path Size: " << (*detailPath).size() << std::endl;
        std::cout << "  isConflict: " << isConflict << std::endl;
        std::cout << "  firstConflictLength: " << firstConflictLength << std::endl;

        double x, y, z, radius;
        std::tie(x, y, z, radius) = firstConflict;
        std::cout << "   firstConflict x:" << x << " y:" << y << " z:" << z << " radius:" << radius << std::endl;

        // std::cout << "   Constrain Size: " << (*constrains).size() << std::endl;
    }

};

class CBSNode{
public:
    CBSNode(size_t num_of_agents):num_of_agents(num_of_agents){};
    ~CBSNode(){
        release_AgentMap();
    };

    size_t num_of_agents;

    std::map<size_ut, AgentInfo*> agentMap;

    double g_val = 0.0;
	double h_val = 0.0;
	int depth;

    void updateAgentConflict(size_t agentIdx);
    void findAllAgentConflict();
    void updateFirstConflict(
        double x, double y, double z, 
        double radius, double length, AgentInfo* agent
    );

    void copy(const CBSNode& rhs){
        release_AgentMap();

        for (auto iter : rhs.agentMap)
        {
            AgentInfo agent = AgentInfo(iter.second);
            agentMap[iter.first] = &agent;
        }
    }

    void update_Constrains(size_ut agentIdx, const std::vector<ConstrainType>& new_constrains){
        agentMap[agentIdx]->update_Constrains(new_constrains);
    }

    void update_DetailPath_And_Tree(size_ut agentIdx, const DetailPath& path){
        agentMap[agentIdx]->update_DetailPath_And_Tree(path);
    }

    struct compare_node 
	{
		bool operator()(const CBSNode* n1, const CBSNode* n2) const 
		{
			return n1->g_val + n1->h_val >= n2->g_val + n2->h_val;
		}
	};
    
    inline double getFVal() const{
        return g_val + h_val;
    }

    // typedef boost::heap::pairing_heap<CBSNode*, boost::heap::compare<CBSNode::compare_node>>::handle_type Open_handle_t;
    // Open_handle_t open_handle;

    void setAgentInfo(size_ut agentIdx, AgentInfo* agent){
        agentMap[agentIdx] = agent;
    }

    void debug(){
        // for (auto iter : agentMap){
        //     iter.second->info();
        // }
    }

private:
    void release_AgentMap(){
        for (auto iter : agentMap)
        {
            iter.second->release();
        }
        agentMap.clear();
    }

};

class CBS{
public:
    CBS(){};
    ~CBS(){};

    double heuristics_mupltier = 1.5;

    DetailPath sampleDetailPath(Path& path, Instance& instance, double stepLength);

    void compute_Heuristics(CBSNode* node);
    void compute_Gval(CBSNode* node);

    void pushNode(CBSNode* node);
    CBSNode* popNode();
    bool is_openList_empty(){
        return this->open_list.empty();
    }

private:
    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node>> open_list;

    inline void releaseNodes(){
        open_list.clear();
    }
    
};

#endif