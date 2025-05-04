#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>
#include <map>
#include <metis.h>

void readEdgeList(const std::string& filename, std::vector<std::set<idx_t>>& adjList, idx_t& nvtxs) {
    std::ifstream infile(filename);
    idx_t u, v;
    std::set<idx_t> nodes;

    while (infile >> u >> v) {
        size_t maxIndex = std::max(std::max(adjList.size(), size_t(u + 1)), size_t(v + 1));
        adjList.resize(maxIndex);
        adjList[u].insert(v);
        adjList[v].insert(u);  // Because the graph is undirected
        nodes.insert(u);
        nodes.insert(v);
    }

    nvtxs = nodes.size();
}


void convertToCSR(const std::vector<std::set<idx_t>>& adjList,
                  std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy) {
    idx_t edgeCounter = 0;
    xadj.push_back(0);  // xadj[0] = 0

    for (const auto& neighbors : adjList) {
        edgeCounter += neighbors.size();
        xadj.push_back(edgeCounter);
        for (idx_t neighbor : neighbors) {
            adjncy.push_back(neighbor);
        }
    }
}

int main() {
    std::string filename = "graph.txt";
    std::vector<std::set<idx_t>> adjList;
    idx_t nvtxs;

    readEdgeList(filename, adjList, nvtxs);

    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    convertToCSR(adjList, xadj, adjncy);

    idx_t ncon = 1;
    idx_t nparts = 2;
    std::vector<idx_t> part(nvtxs);
    idx_t objval;

    idx_t* vwgt = NULL;
    idx_t* vsize = NULL;
    idx_t* adjwgt = NULL;

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    real_t ubvec[1] = {1.05};

    int status = METIS_PartGraphKway(
        &nvtxs, &ncon,
        xadj.data(), adjncy.data(),
        vwgt, vsize, adjwgt,
        &nparts, NULL, ubvec,
        options, &objval, part.data()
    );

    if (status == METIS_OK) {
        std::cout << "Partitioning successful. Edge cut = " << objval << "\n";
        for (idx_t i = 0; i < nvtxs; ++i) {
            std::cout << "Vertex " << i << " -> Part " << part[i] << "\n";
        }
    } else {
        std::cerr << "METIS_PartGraphKway failed with code " << status << "\n";
    }

    return 0;
}
