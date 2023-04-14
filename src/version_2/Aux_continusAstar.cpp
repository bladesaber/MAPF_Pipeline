#include "Aux_continusAstar.h"

void HybridAstar::pushNode(HybridAstarNode* node){
    num_generated++;

    node->open_handle = open_list.push(node);
	node->in_openlist = true;
}

HybridAstarNode* HybridAstar::popNode(){
    HybridAstarNode* node = open_list.top();
    open_list.pop();
    node->in_openlist = false;
    return node;
}
