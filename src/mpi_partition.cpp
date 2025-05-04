#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <metis.h>
#include <mpi.h>

void readEdgeList(const std::string& filename, std::vector<std::set<idx_t>>& adjList, idx_t& nvtxs) {
    std::ifstream infile(filename);
    idx_t u, v;
    std::set<idx_t> nodes;

    while (infile >> u >> v) {
        size_t maxIndex = std::max(std::max(adjList.size(), size_t(u + 1)), size_t(v + 1));
        adjList.resize(maxIndex);
        adjList[u].insert(v);
        adjList[v].insert(u);
        nodes.insert(u);
        nodes.insert(v);
    }

    nvtxs = nodes.size();
}

void convertToCSR(const std::vector<std::set<idx_t>>& adjList,
                  std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy) {
    idx_t edgeCounter = 0;
    xadj.push_back(0);
    for (const auto& neighbors : adjList) {
        edgeCounter += neighbors.size();
        xadj.push_back(edgeCounter);
        for (idx_t neighbor : neighbors) {
            adjncy.push_back(neighbor);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<idx_t> xadj, adjncy, part;
    idx_t nvtxs = 0, objval;

    if (rank == 0) {
        std::vector<std::set<idx_t>> adjList;
        readEdgeList("graph.txt", adjList, nvtxs);
        convertToCSR(adjList, xadj, adjncy);

        idx_t ncon = 1;
        idx_t* vwgt = NULL;
        idx_t* vsize = NULL;
        idx_t* adjwgt = NULL;
        real_t ubvec[1] = {1.05};
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);

        part.resize(nvtxs);
        METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                            vwgt, vsize, adjwgt, &nprocs, NULL,
                            ubvec, options, &objval, part.data());
    }

    // Broadcast graph size
    MPI_Bcast(&nvtxs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast xadj and adjncy sizes
    idx_t xadj_size = 0, adjncy_size = 0;
    if (rank == 0) {
        xadj_size = xadj.size();
        adjncy_size = adjncy.size();
    }
    MPI_Bcast(&xadj_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&adjncy_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        xadj.resize(xadj_size);
        adjncy.resize(adjncy_size);
        part.resize(nvtxs);
    }

    MPI_Bcast(xadj.data(), xadj_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(adjncy.data(), adjncy_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(part.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);

    // Extract local vertices
    std::vector<idx_t> local_vertices;
    for (idx_t i = 0; i < nvtxs; ++i)
        if (part[i] == rank)
            local_vertices.push_back(i);

    std::cout << "Process " << rank << " owns " << local_vertices.size() << " vertices:\n";
    for (idx_t v : local_vertices)
        std::cout << "  " << v << "\n";

    MPI_Finalize();
    return 0;
}