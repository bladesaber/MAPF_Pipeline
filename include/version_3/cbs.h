#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "utils.h"
#include "conflict.h"

#include "angleAstar.h"
#include "kdtreeWrapper.h"

class AgentInfo{
public:
    size_ut agentIdx;
    double radius;
    std::tuple<int, int, int> startPos;
    std::tuple<int, int, int> endPos;

    bool isConflict = false;
    std::set<std::tuple<int, int, int>> conflictSet;
    size_t conflictNum = 0;
    Conflict firstConflict;

    AgentInfo(){};
    AgentInfo(
        size_ut agentIdx, double radius,
        std::tuple<int, int, int> startPos,
        std::tuple<int, int, int> endPos
    ):agentIdx(agentIdx), radius(radius), startPos(startPos), endPos(endPos){};
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
        this->startPos = rhs->startPos;
        this->endPos = rhs->endPos;

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
        std::cout << "   StartPos:" << " x:" << std::get<0>(startPos) << " y:" << std::get<1>(startPos) << " z:" << std::get<2>(startPos) << std::endl;
        std::cout << "   EndPos:" << " x:" << std::get<0>(endPos) << " y:" << std::get<1>(endPos) << " z:" << std::get<2>(endPos) << std::endl;
        std::cout << "   radius: " << radius << std::endl;
        std::cout << "   isConflict: " << isConflict << std::endl;
        std::cout << "   ConflictNum:" << conflictNum << std::endl;
        std::cout << "   Constrain Size: " << (*constrains).size() << std::endl;

        // std::cout << "   Share Ptr Test: constrains.use_count:" << constrains.use_count();
        // std::cout << " detailPath.use_count:" << detailPath.use_count();
        // std::cout << " pathTree.use_count: " << pathTree.use_count() << std::endl;
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

    int node_id;
    double g_val = 0.0;
	double h_val = 0.0;
	int depth = 0;

    void findAllAgentConflict();
    void updateFirstConflict(
        AgentInfo* agent1,
        
        double conflict1_x, double conflict1_y, double conflict1_z, 
        double conflict1_radius, double conflict1_length, 
        
        size_ut conflict_agentIdx,
        double conflict2_x, double conflict2_y, double conflict2_z, 
        double conflict2_radius, double conflict2_length
    );

    void copy(const CBSNode& rhs){
        release_AgentMap();

        for (auto iter : rhs.agentMap)
        {
            AgentInfo* agent = new AgentInfo(iter.second);
            agentMap[iter.first] = agent;
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

    void addAgent(
        size_ut agentIdx, double radius,
        std::tuple<int, int, int> startPos,
        std::tuple<int, int, int> endPos
    ){
        AgentInfo* agent = new AgentInfo(agentIdx, radius, startPos, endPos);
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
            delete iter.second;
        }
        agentMap.clear();
    }

};

class CBS{
public:
    CBS(){};
    ~CBS(){
        releaseNodes();
        releaseEngines();
    };

    double heuristics_mupltier = 1.5;
    double stepLength = 0.5;

    double runtime_search = 0;

    DetailPath sampleDetailPath(Path& path, Instance& instance, double stepLength);

    void compute_Heuristics(CBSNode* node);
    void compute_Gval(CBSNode* node);
    bool isGoal(CBSNode* node);

    void pushNode(CBSNode* node);
    CBSNode* popNode();
    bool is_openList_empty(){
        return this->open_list.empty();
    }

    bool update_AgentPath(Instance& instance, CBSNode* node, size_ut agentIdx);

    void addSearchEngine(size_ut agentIdx, double radius){
        search_engines[agentIdx] = new AngleAStar(radius);
    }

    void info(){
        std::cout << "CBS Info" << std::endl;
        std::cout << "   SearchEngine Size: " << search_engines.size() << std::endl;
        std::cout << "   openList Size: " << open_list.size() << std::endl;
    }

private:
    std::map<size_ut, AngleAStar*> search_engines;

    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node>> open_list;

    inline void releaseNodes(){
        open_list.clear();
    }

    inline void releaseEngines(){
        for (auto iter : search_engines){
            iter.second->releaseNodes();
            delete iter.second;
        }
        search_engines.clear();
    }

};

#endif