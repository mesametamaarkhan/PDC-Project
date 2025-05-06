#include <mpi.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <metis.h>

// Structure to represent a vertex in the graph
struct Vertex {
    int id;
    std::vector<std::pair<int, int>> neighbors; // <neighbor_id, weight>
    int distance;  // Distance from source
    int parent;    // Parent in the shortest path tree
    bool affected; // Whether the vertex is affected by recent changes
    bool boundary; // Whether the vertex is a boundary vertex
};

// Structure to represent an edge change
struct EdgeChange {
    int u, v;      // Edge endpoints
    int weight;    // New weight (use -1 for deletion)
    bool insertion; // True if insertion, false if deletion
};

// Class for Parallel SSSP
class ParallelDynamicSSSP {
private:
    // Graph data
    std::vector<Vertex> local_graph;
    std::unordered_map<int, int> global_to_local; // Map global vertex IDs to local IDs
    std::unordered_map<int, int> local_to_global; // Map local vertex IDs to global IDs
    
    // MPI-related data
    int rank;
    int size;
    int source_vertex;
    int num_vertices;
    
    // Partition information
    std::vector<int> vertex_partition; // Which partition each vertex belongs to
    std::vector<std::unordered_set<int>> boundary_vertices; // Boundary vertices for each partition
    
    // Priority queue for distance updates (min-heap based on distance)
    using PQEntry = std::pair<int, int>; // <distance, vertex_id>
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> update_queue;
    
    // Buffer for communication
    struct BoundaryUpdate {
        int global_vertex_id;
        int new_distance;
        int new_parent;
    };
    std::vector<BoundaryUpdate> send_buffer;
    std::vector<BoundaryUpdate> recv_buffer;

public:
    ParallelDynamicSSSP(int my_rank, int num_procs, int source) 
        : rank(my_rank), size(num_procs), source_vertex(source) {
        // Initialize MPI buffers
        boundary_vertices.resize(size);
    }
    
    void loadAndPartitionGraph(const std::vector<std::tuple<int, int, int>>& edges, int num_vertices) {
        std::cerr << "Rank " << rank << ": Entering loadAndPartitionGraph, num_vertices=" << num_vertices << ", edges=" << edges.size() << std::endl;
        this->num_vertices = num_vertices;
        
        // Building adjacency list
        std::vector<std::vector<std::pair<int, int>>> adj_list(num_vertices);
        for (const auto& edge : edges) {
            int u = std::get<0>(edge);
            int v = std::get<1>(edge);
            int weight = std::get<2>(edge);
            if (u >= num_vertices || v >= num_vertices || u < 0 || v < 0) {
                std::cerr << "Rank " << rank << ": Invalid edge (" << u << ", " << v << ")" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            adj_list[u].emplace_back(v, weight);
            adj_list[v].emplace_back(u, weight);
        }
        std::cerr << "Rank " << rank << ": Adjacency list created, size=" << adj_list.size() << std::endl;
        
        // Manual partitioning for small graphs to ensure balanced distribution
        vertex_partition.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++) {
            vertex_partition[i] = i % size;
            std::cerr << "Rank " << rank << ": Vertex " << i << " assigned to partition " << vertex_partition[i] << std::endl;
        }
        
        // For larger graphs, use METIS
        if (num_vertices > size * 5) {
            // Initialize METIS arrays
            idx_t nvtxs = num_vertices;
            idx_t ncon = 1;
            idx_t nparts = size;
            
            std::vector<idx_t> xadj(num_vertices + 1, 0);
            std::vector<idx_t> adjncy;
            std::vector<idx_t> adjwgt;
            
            // Convert adjacency list to CSR format for METIS
            int edge_count = 0;
            for (int i = 0; i < num_vertices; i++) {
                xadj[i] = edge_count;
                for (const auto& neighbor : adj_list[i]) {
                    adjncy.push_back(neighbor.first);
                    adjwgt.push_back(neighbor.second);
                    edge_count++;
                }
            }
            xadj[num_vertices] = edge_count;
            
            // Run METIS partitioning
            std::vector<idx_t> part(num_vertices);
            idx_t objval;
            
            int ret = METIS_PartGraphKway(
                &nvtxs, &ncon, xadj.data(), adjncy.data(),
                nullptr, nullptr, adjwgt.data(), &nparts,
                nullptr, nullptr, nullptr, &objval, part.data()
            );
            
            if (ret != METIS_OK) {
                std::cerr << "Rank " << rank << ": METIS partitioning failed with error code " << ret << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            vertex_partition = std::vector<int>(part.begin(), part.end());
            std::cerr << "Rank " << rank << ": METIS partitioning succeeded, edge-cut=" << objval << std::endl;
        }
        
        // Log vertices owned by this rank
        for (int i = 0; i < num_vertices; ++i) {
            if (vertex_partition[i] == rank) {
                std::cerr << "Rank " << rank << ": Owns vertex " << i << std::endl;
            }
        }
        
        // Build local graph
        buildLocalGraph(adj_list, num_vertices);
        std::cerr << "Rank " << rank << ": buildLocalGraph completed, local_graph.size=" << local_graph.size() << std::endl;
        
        // Identify boundary vertices
        identifyBoundaryVertices(adj_list);
        std::cerr << "Rank " << rank << ": loadAndPartitionGraph completed" << std::endl;
    }
    
    void buildLocalGraph(const std::vector<std::vector<std::pair<int, int>>>& adj_list, int num_vertices) {
        std::cerr << "Rank " << rank << ": Entering buildLocalGraph, num_vertices=" << num_vertices << std::endl;
        int local_idx = 0;
        local_graph.clear();
        global_to_local.clear();
        local_to_global.clear();
    
        for (int global_id = 0; global_id < num_vertices; global_id++) {
            if (vertex_partition[global_id] == rank) {
                global_to_local[global_id] = local_idx;
                local_to_global[local_idx] = global_id;
                
                Vertex v;
                v.id = local_idx;
                v.distance = (global_id == source_vertex) ? 0 : std::numeric_limits<int>::max();
                v.parent = -1;
                v.affected = false;
                v.boundary = false;
                
                local_graph.push_back(v);
                local_idx++;
            }
        }
        std::cerr << "Rank " << rank << ": Assigned " << local_idx << " local vertices" << std::endl;
    
        for (int global_id = 0; global_id < num_vertices; global_id++) {
            if (vertex_partition[global_id] == rank) {
                int local_u = global_to_local[global_id];
                if (local_u >= (int)local_graph.size()) {
                    std::cerr << "Rank " << rank << ": Invalid local_u=" << local_u << " for global_id=" << global_id << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                
                for (const auto& neighbor : adj_list[global_id]) {
                    int global_v = neighbor.first;
                    int weight = neighbor.second;
                    
                    if (vertex_partition[global_v] == rank) {
                        if (global_to_local.find(global_v) == global_to_local.end()) {
                            std::cerr << "Rank " << rank << ": Missing global_to_local mapping for global_v=" << global_v << std::endl;
                            MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                        int local_v = global_to_local[global_v];
                        local_graph[local_u].neighbors.emplace_back(local_v, weight);
                    }
                }
            }
        }
        std::cerr << "Rank " << rank << ": Local graph edges assigned" << std::endl;
    }
    
    void identifyBoundaryVertices(const std::vector<std::vector<std::pair<int, int>>>& adj_list) {
        std::cerr << "Rank " << rank << ": Entering identifyBoundaryVertices" << std::endl;
        
        // Initialize boundary_vertices for all ranks
        boundary_vertices.resize(size);
        for (int i = 0; i < size; i++) {
            boundary_vertices[i].clear();
        }
        
        // Identify local boundary vertices
        for (int global_id = 0; global_id < adj_list.size(); global_id++) {
            if (vertex_partition[global_id] == rank) {
                int local_id = global_to_local[global_id];
                bool is_boundary = false;
                
                for (const auto& neighbor : adj_list[global_id]) {
                    int global_v = neighbor.first;
                    int weight = neighbor.second;
                    int target_rank = vertex_partition[global_v];
                    
                    if (target_rank != rank) {
                        is_boundary = true;
                        boundary_vertices[target_rank].insert(global_id);
                        local_graph[local_id].neighbors.emplace_back(global_v, weight);
                    }
                }
                
                if (is_boundary) {
                    local_graph[local_id].boundary = true;
                }
            }
        }
        
        // Log local boundary vertices
        for (int i = 0; i < size; i++) {
            std::cerr << "Rank " << rank << ": Local boundary_vertices[" << i << "] size=" << boundary_vertices[i].size() << std::endl;
        }
        
        // Collect boundary vertex counts from all ranks
        std::vector<int> send_buffer;
        for (int i = 0; i < size; i++) {
            if (i != rank) {
                for (int v : boundary_vertices[i]) {
                    send_buffer.push_back(v);
                }
            }
        }
        int send_count = send_buffer.size();
        std::cerr << "Rank " << rank << ": Sending " << send_count << " boundary vertices" << std::endl;
        
        std::vector<int> recv_counts(size);
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::cerr << "Rank " << rank << ": Received boundary counts";
        for (int i = 0; i < size; i++) {
            std::cerr << " " << recv_counts[i];
        }
        std::cerr << std::endl;
        
        // Calculate displacements and total receive size
        std::vector<int> displacements(size, 0);
        int total_recv = 0;
        for (int i = 0; i < size; i++) {
            total_recv += recv_counts[i];
            if (i > 0) {
                displacements[i] = displacements[i-1] + recv_counts[i-1];
            }
        }
        
        // Validate buffer sizes
        if (total_recv < 0 || total_recv > 1000) { // Arbitrary upper limit for safety
            std::cerr << "Rank " << rank << ": Invalid total_recv=" << total_recv << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        std::vector<int> recv_buffer(total_recv);
        MPI_Allgatherv(
            send_buffer.data(), send_count, MPI_INT,
            recv_buffer.data(), recv_counts.data(), displacements.data(), MPI_INT,
            MPI_COMM_WORLD
        );
        
        // Update boundary_vertices with received data
        for (int i = 0; i < size; i++) {
            if (i == rank) continue;
            for (int j = displacements[i]; j < displacements[i] + recv_counts[i]; j++) {
                boundary_vertices[i].insert(recv_buffer[j]);
            }
        }
        
        // Log final boundary vertices
        for (int i = 0; i < size; i++) {
            std::cerr << "Rank " << rank << ": Final boundary_vertices[" << i << "] size=" << boundary_vertices[i].size();
            std::cerr << ", vertices:";
            for (int v : boundary_vertices[i]) {
                std::cerr << " " << v;
            }
            std::cerr << std::endl;
        }
        
        std::cerr << "Rank " << rank << ": identifyBoundaryVertices completed" << std::endl;
    }
    
    void computeInitialSSSP() {
        std::cerr << "Rank " << rank << ": Starting computeInitialSSSP, local_graph.size=" << local_graph.size() << std::endl;
        if (local_graph.empty()) {
            std::cerr << "Rank " << rank << ": Empty local graph, participating in synchronization only" << std::endl;
            synchronizeBoundaries();
            return;
        }
    
        int local_source = -1;
        for (size_t i = 0; i < local_graph.size(); i++) {
            if (local_to_global[i] == source_vertex) {
                local_source = i;
                local_graph[i].distance = 0;
                update_queue.push({0, i});
                std::cerr << "Rank " << rank << ": Source vertex found at local_id=" << i << std::endl;
            }
        }
        
        int iteration = 0;
        while (!update_queue.empty()) {
            std::cerr << "Rank " << rank << ": Dijkstra iteration " << iteration++ << ", queue size=" << update_queue.size() << std::endl;
            int dist = update_queue.top().first;
            int u = update_queue.top().second;
            update_queue.pop();
            
            if (dist > local_graph[u].distance) continue;
            
            for (const auto& edge : local_graph[u].neighbors) {
                int v = edge.first;
                int weight = edge.second;
                
                if (global_to_local.find(v) == global_to_local.end()) {
                    BoundaryUpdate update;
                    update.global_vertex_id = local_to_global[u];
                    update.new_distance = local_graph[u].distance;
                    update.new_parent = local_to_global[u];
                    send_buffer.push_back(update);
                    continue;
                }
                
                int local_v = global_to_local[v];
                int new_dist = local_graph[u].distance + weight;
                if (new_dist < local_graph[local_v].distance) {
                    local_graph[local_v].distance = new_dist;
                    local_graph[local_v].parent = u;
                    update_queue.push({new_dist, local_v});
                    
                    if (local_graph[local_v].boundary) {
                        BoundaryUpdate update;
                        update.global_vertex_id = local_to_global[local_v];
                        update.new_distance = new_dist;
                        update.new_parent = local_to_global[u];
                        send_buffer.push_back(update);
                    }
                }
            }
        }
        
        std::cerr << "Rank " << rank << ": Local Dijkstra completed, synchronizing boundaries" << std::endl;
        synchronizeBoundaries();
        std::cerr << "Rank " << rank << ": Finished synchronizeBoundaries" << std::endl;
        
        for (size_t i = 0; i < local_graph.size(); i++) {
            int distance = local_graph[i].distance;
            std::cerr << "Rank " << rank << ": Vertex " << local_to_global[i] << " distance=" 
                    << (distance == std::numeric_limits<int>::max() ? "INF" : std::to_string(distance)) << std::endl;
        }
        
        std::cerr << "Rank " << rank << ": Finished computeInitialSSSP" << std::endl;
    }
    
    // Process a batch of edge changes
    void processEdgeChanges(const std::vector<EdgeChange>& changes) {
        bool local_changes = false;
        
        // Process each edge change
        for (const auto& change : changes) {
            // Check if the edge belongs to this partition
            bool u_is_local = (vertex_partition[change.u] == rank);
            bool v_is_local = (vertex_partition[change.v] == rank);
            
            if (!u_is_local && !v_is_local) continue; // Edge not in this partition
            
            local_changes = true;
            
            // Process edge insertion/deletion
            if (change.insertion) {
                // Insert edge
                if (u_is_local) {
                    int local_u = global_to_local[change.u];
                    if (v_is_local) {
                        // Both endpoints in this partition
                        int local_v = global_to_local[change.v];
                        local_graph[local_u].neighbors.emplace_back(local_v, change.weight);
                        local_graph[local_v].neighbors.emplace_back(local_u, change.weight);
                    } else {
                        // v is in another partition
                        local_graph[local_u].neighbors.emplace_back(change.v, change.weight);
                        local_graph[local_u].boundary = true;
                        boundary_vertices[vertex_partition[change.v]].insert(change.u);
                    }
                } else {
                    // u is in another partition, v is local
                    int local_v = global_to_local[change.v];
                    local_graph[local_v].neighbors.emplace_back(change.u, change.weight);
                    local_graph[local_v].boundary = true;
                    boundary_vertices[vertex_partition[change.u]].insert(change.v);
                }
                
                // Check if the new edge provides a shorter path
                updateAfterEdgeInsertion(change.u, change.v, change.weight);
            } else {
                // Delete edge
                if (u_is_local) {
                    int local_u = global_to_local[change.u];
                    if (v_is_local) {
                        // Both endpoints in this partition
                        int local_v = global_to_local[change.v];
                        // Remove edge (u,v)
                        auto& u_neighbors = local_graph[local_u].neighbors;
                        u_neighbors.erase(std::remove_if(u_neighbors.begin(), u_neighbors.end(),
                            [local_v](const std::pair<int, int>& e) { return e.first == local_v; }),
                            u_neighbors.end());
                        
                        // Remove edge (v,u)
                        auto& v_neighbors = local_graph[local_v].neighbors;
                        v_neighbors.erase(std::remove_if(v_neighbors.begin(), v_neighbors.end(),
                            [local_u](const std::pair<int, int>& e) { return e.first == local_u; }),
                            v_neighbors.end());
                        
                        // Mark endpoints as affected
                        local_graph[local_u].affected = true;
                        local_graph[local_v].affected = true;
                    } else {
                        // v is in another partition
                        auto& neighbors = local_graph[local_u].neighbors;
                        neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                            [change](const std::pair<int, int>& e) { return e.first == change.v; }),
                            neighbors.end());
                        
                        // Mark u as affected
                        local_graph[local_u].affected = true;
                        
                        // Check if u is still a boundary vertex
                        bool still_boundary = false;
                        for (const auto& neighbor : neighbors) {
                            int neighbor_id = neighbor.first;
                            if (global_to_local.find(neighbor_id) == global_to_local.end()) {
                                still_boundary = true;
                                break;
                            }
                        }
                        local_graph[local_u].boundary = still_boundary;
                    }
                } else {
                    // u is in another partition, v is local
                    int local_v = global_to_local[change.v];
                    auto& neighbors = local_graph[local_v].neighbors;
                    neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                        [change](const std::pair<int, int>& e) { return e.first == change.u; }),
                        neighbors.end());
                    
                    // Mark v as affected
                    local_graph[local_v].affected = true;
                    
                    // Check if v is still a boundary vertex
                    bool still_boundary = false;
                    for (const auto& neighbor : neighbors) {
                        int neighbor_id = neighbor.first;
                        if (global_to_local.find(neighbor_id) == global_to_local.end()) {
                            still_boundary = true;
                            break;
                        }
                    }
                    local_graph[local_v].boundary = still_boundary;
                }
                
                // After edge deletion, need to check if distances need to be updated
                updateAfterEdgeDeletion();
            }
        }
        
        // If local changes occurred, update SSSP
        if (local_changes) {
            updateSSSP();
        }
    }
    
    // Update SSSP after edge insertion
    void updateAfterEdgeInsertion(int u, int v, int weight) {
        // Check if the new edge provides a shorter path
        if (vertex_partition[u] == rank) {
            int local_u = global_to_local[u];
            int u_dist = local_graph[local_u].distance;
            
            if (u_dist != std::numeric_limits<int>::max()) {
                if (vertex_partition[v] == rank) {
                    // Both endpoints in this partition
                    int local_v = global_to_local[v];
                    int v_dist = local_graph[local_v].distance;
                    
                    // Check if the new edge provides a shorter path to v
                    if (u_dist + weight < v_dist) {
                        local_graph[local_v].distance = u_dist + weight;
                        local_graph[local_v].parent = local_u;
                        local_graph[local_v].affected = true;
                        update_queue.push({u_dist + weight, local_v});
                    }
                    
                    // Check if the new edge provides a shorter path to u
                    if (v_dist + weight < u_dist) {
                        local_graph[local_u].distance = v_dist + weight;
                        local_graph[local_u].parent = local_v;
                        local_graph[local_u].affected = true;
                        update_queue.push({v_dist + weight, local_u});
                    }
                } else {
                    // v is in another partition
                    // We need to get v's distance from its owner
                    BoundaryUpdate query;
                    query.global_vertex_id = v;
                    query.new_distance = -1; // Query indicator
                    query.new_parent = -1;
                    send_buffer.push_back(query);
                }
            }
        } else if (vertex_partition[v] == rank) {
            // u is in another partition, v is local
            int local_v = global_to_local[v];
            int v_dist = local_graph[local_v].distance;
            
            if (v_dist != std::numeric_limits<int>::max()) {
                // We need to get u's distance from its owner
                BoundaryUpdate query;
                query.global_vertex_id = u;
                query.new_distance = -1; // Query indicator
                query.new_parent = -1;
                send_buffer.push_back(query);
            }
        }
    }
    
    // Update SSSP after edge deletion
    void updateAfterEdgeDeletion() {
        // Reset all affected nodes to infinity except the source
        for (auto& vertex : local_graph) {
            if (vertex.affected && local_to_global[vertex.id] != source_vertex) {
                vertex.distance = std::numeric_limits<int>::max();
                vertex.parent = -1;
            }
        }
        
        // Starting from the source, recompute distances for affected nodes
        for (size_t i = 0; i < local_graph.size(); i++) {
            if (local_to_global[i] == source_vertex) {
                local_graph[i].distance = 0;
                update_queue.push({0, (int)i});
                break;
            }
        }
        
        // Use the update priority queue to process vertices in order of increasing distance
        while (!update_queue.empty()) {
            int dist = update_queue.top().first;
            int u = update_queue.top().second;
            update_queue.pop();
            
            // Skip if we've found a better path already
            if (dist > local_graph[u].distance) continue;
            
            // Process neighbors
            for (const auto& edge : local_graph[u].neighbors) {
                int v = edge.first;
                int weight = edge.second;
                
                // If v is a global ID (boundary vertex)
                if (global_to_local.find(v) == global_to_local.end()) {
                    // Add to send buffer for boundary update
                    BoundaryUpdate update;
                    update.global_vertex_id = local_to_global[u];
                    update.new_distance = local_graph[u].distance;
                    update.new_parent = local_to_global[u];
                    send_buffer.push_back(update);
                    continue;
                }
                
                // Convert to local ID if needed
                int local_v = (global_to_local.find(v) != global_to_local.end()) ? 
                             global_to_local[v] : v;
                
                // Relax edge
                int new_dist = local_graph[u].distance + weight;
                if (new_dist < local_graph[local_v].distance) {
                    local_graph[local_v].distance = new_dist;
                    local_graph[local_v].parent = u;
                    local_graph[local_v].affected = true;
                    update_queue.push({new_dist, local_v});
                    
                    // If this is a boundary vertex, prepare update for other partitions
                    if (local_graph[local_v].boundary) {
                        BoundaryUpdate update;
                        update.global_vertex_id = local_to_global[local_v];
                        update.new_distance = new_dist;
                        update.new_parent = local_to_global[u];
                        send_buffer.push_back(update);
                    }
                }
            }
        }
    }
    
    // Update SSSP after edge changes
    void updateSSSP() {
        bool global_changes;
        bool local_changes;
        int iteration = 0;
        std::cerr << "Rank " << rank << ": Starting updateSSSP" << std::endl;

        do {
            local_changes = false;
            std::cerr << "Rank " << rank << ": Iteration " << iteration++ << ", local_changes=" << local_changes << std::endl;
            
            // Process local updates using OpenMP
            #pragma omp parallel
            {
                #pragma omp for
                for (size_t i = 0; i < local_graph.size(); i++) {
                    Vertex& vertex = local_graph[i];
                    if (!vertex.affected) continue;
                    
                    // Reset for this iteration
                    #pragma omp atomic write
                    vertex.affected = false;
                    
                    // Process each neighbor
                    for (const auto& edge : vertex.neighbors) {
                        int v = edge.first;
                        int weight = edge.second;
                        
                        // If v is a global ID (boundary vertex)
                        if (global_to_local.find(v) == global_to_local.end()) {
                            // Will be handled in synchronization
                            continue;
                        }
                        
                        // Convert to local ID if needed
                        int local_v = (global_to_local.find(v) != global_to_local.end()) ? 
                                     global_to_local[v] : v;
                        
                        // Critical section for updating distance
                        #pragma omp critical
                        {
                            // Relax edge
                            int new_dist = vertex.distance + weight;
                            if (new_dist < local_graph[local_v].distance) {
                                local_graph[local_v].distance = new_dist;
                                local_graph[local_v].parent = vertex.id;
                                local_graph[local_v].affected = true;
                                local_changes = true;
                                
                                // If this is a boundary vertex, prepare update for other partitions
                                if (local_graph[local_v].boundary) {
                                    BoundaryUpdate update;
                                    update.global_vertex_id = local_to_global[local_v];
                                    update.new_distance = new_dist;
                                    update.new_parent = local_to_global[vertex.id];
                                    send_buffer.push_back(update);
                                }
                            }
                        }
                    }
                }
            }
            
            // Synchronize boundary information
            synchronizeBoundaries();
            
            // Check if any process still has changes
            MPI_Allreduce(&local_changes, &global_changes, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            std::cerr << "Rank " << rank << ": After Allreduce, global_changes=" << global_changes << std::endl;  
        } while (global_changes);
        std::cerr << "Rank " << rank << ": Finished updateSSSP after " << iteration << " iterations" << std::endl;
    }
    
    void synchronizeBoundaries() {
        int send_count = send_buffer.size();
        std::vector<int> recv_counts(size);
        std::cerr << "Rank " << rank << ": Starting synchronizeBoundaries, send_count=" << send_count << std::endl;
    
        // Get counts from all processes
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::cerr << "Rank " << rank << ": Received counts";
        for (int i = 0; i < size; i++) {
            std::cerr << " " << recv_counts[i];
        }
        std::cerr << std::endl;
    
        // Calculate displacements for gatherv
        std::vector<int> displacements(size, 0);
        int total_recv = 0;
        for (int i = 0; i < size; i++) {
            // Sanity check for buffer sizes
            if (recv_counts[i] < 0 || recv_counts[i] > 1000) { // Arbitrary limit for safety
                std::cerr << "Rank " << rank << ": Invalid recv_counts[" << i << "]=" << recv_counts[i] << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            total_recv += recv_counts[i];
            if (i > 0) {
                displacements[i] = displacements[i-1] + recv_counts[i-1];
            }
        }
        if (total_recv < 0 || total_recv > 10000) { // Larger arbitrary limit for total
            std::cerr << "Rank " << rank << ": Invalid total_recv=" << total_recv << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    
        // Resize receive buffer
        recv_buffer.resize(total_recv);
        std::cerr << "Rank " << rank << ": Allocated recv_buffer, size=" << total_recv << std::endl;
    
        // If no updates to exchange, skip MPI communication
        if (total_recv == 0) {
            std::cerr << "Rank " << rank << ": No boundary updates to exchange" << std::endl;
            send_buffer.clear();
            return;
        }
    
        // Create MPI datatype for boundary updates
        MPI_Datatype mpi_boundary_update;
        int blocklengths[3] = {1, 1, 1};
        MPI_Aint disps[3];
        MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
        BoundaryUpdate dummy;
        MPI_Aint base_addr;
        MPI_Get_address(&dummy, &base_addr);
        MPI_Get_address(&dummy.global_vertex_id, &disps[0]);
        MPI_Get_address(&dummy.new_distance, &disps[1]);
        MPI_Get_address(&dummy.new_parent, &disps[2]);
        disps[0] = MPI_Aint_diff(disps[0], base_addr);
        disps[1] = MPI_Aint_diff(disps[1], base_addr);
        disps[2] = MPI_Aint_diff(disps[2], base_addr);
        MPI_Type_create_struct(3, blocklengths, disps, types, &mpi_boundary_update);
        MPI_Type_commit(&mpi_boundary_update);
    
        // Exchange boundary updates
        if (send_count > 0 || total_recv > 0) {
            MPI_Allgatherv(
                send_count > 0 ? send_buffer.data() : nullptr, send_count, mpi_boundary_update,
                total_recv > 0 ? recv_buffer.data() : nullptr, recv_counts.data(), displacements.data(), mpi_boundary_update,
                MPI_COMM_WORLD
            );
        }
    
        MPI_Type_free(&mpi_boundary_update);
    
        // Process received updates
        for (const auto& update : recv_buffer) {
            std::cerr << "Rank " << rank << ": Processing update for global_vertex_id=" << update.global_vertex_id << std::endl;
            if (vertex_partition[update.global_vertex_id] == rank) continue; // Skip our own vertices
            processReceivedUpdate(update);
        }
    
        // Clear send buffer for next round
        send_buffer.clear();
        std::cerr << "Rank " << rank << ": Finished synchronizeBoundaries, received " << recv_buffer.size() << " updates" << std::endl;
    }
    
    // Process a received boundary update
    void processReceivedUpdate(const BoundaryUpdate& update) {
        int global_v = update.global_vertex_id;
        int global_parent = update.new_parent;
        int new_dist = update.new_distance;
        
        // If this is a query response, process it
        if (new_dist == -1) {
            // This is a query response logic
            return;
        }
        
        // Check if we have this vertex locally
        if (global_to_local.find(global_v) != global_to_local.end()) {
            int local_v = global_to_local[global_v];
            
            // Update the distance if it's better
            if (new_dist < local_graph[local_v].distance) {
                local_graph[local_v].distance = new_dist;
                
                // If the parent is in this partition, update it
                if (global_to_local.find(global_parent) != global_to_local.end()) {
                    local_graph[local_v].parent = global_to_local[global_parent];
                } else {
                    // Parent is in another partition
                    local_graph[local_v].parent = -1; // Special indicator for cross-partition parent
                }
                
                local_graph[local_v].affected = true;
            }
        }
        
        // Check all edges connecting to this vertex
        for (auto& vertex : local_graph) {
            for (const auto& edge : vertex.neighbors) {
                if (edge.first == global_v) {
                    int weight = edge.second;
                    int potential_dist = new_dist + weight;
                    
                    // If this provides a better path
                    if (potential_dist < vertex.distance) {
                        vertex.distance = potential_dist;
                        vertex.parent = -1; // Special indicator for cross-partition parent
                        vertex.affected = true;
                    }
                }
            }
        }
    }
    
    // Check if load balancing is needed
    bool checkLoadBalance() {
        // Count active vertices in this partition
        int local_active = 0;
        for (const auto& vertex : local_graph) {
            if (vertex.affected) {
                local_active++;
            }
        }
        
        // Gather active vertex counts from all processes
        std::vector<int> active_counts(size);
        MPI_Allgather(&local_active, 1, MPI_INT, active_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Calculate mean and max
        int total_active = 0;
        int max_active = 0;
        for (int count : active_counts) {
            total_active += count;
            max_active = std::max(max_active, count);
        }
        
        // If there are no active vertices, no need to rebalance
        if (total_active == 0) return false;
        
        double mean_active = static_cast<double>(total_active) / size;
        double imbalance = max_active / mean_active;
        
        // If imbalance exceeds threshold, trigger rebalancing
        const double IMBALANCE_THRESHOLD = 1.5; // Adjust based on your needs
        return imbalance > IMBALANCE_THRESHOLD;
    }
    
    // Repartition the graph using ParMETIS
    void repartitionGraph() {
        // This is a simplified version; in practice, you would use ParMETIS
        // to perform distributed graph partitioning
        
        // For now, we'll print a message and skip actual repartitioning
        if (rank == 0) {
            std::cout << "Load imbalance detected. Repartitioning would be triggered here." << std::endl;
        }
        
        // In real implementation, you would:
        // 1. Convert local graph to ParMETIS format
        // 2. Call ParMETIS_V3_PartKway
        // 3. Redistribute vertices based on new partitioning
        // 4. Rebuild local graphs and boundary information
    }
    
    std::vector<std::pair<int, int>> getResult() {
        std::cerr << "Rank " << rank << ": Entering getResult, local_graph.size=" << local_graph.size() << std::endl;
    
        // Prepare local results
        std::vector<std::pair<int, int>> local_result;
        for (size_t i = 0; i < local_graph.size(); i++) {
            int distance = local_graph[i].distance;
            if (distance == std::numeric_limits<int>::max()) {
                std::cerr << "Rank " << rank << ": Vertex " << local_to_global[i] << " has unreachable distance (INT_MAX)" << std::endl;
                distance = -1;
            }
            local_result.emplace_back(local_to_global[i], distance);
        }
    
        int local_count = local_result.size();
        std::cerr << "Rank " << rank << ": local_count=" << local_count << std::endl;
    
        // Gather counts from all processes
        std::vector<int> recv_counts(size);
        MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::cerr << "Rank " << rank << ": Completed MPI_Allgather for counts" << std::endl;
    
        // Calculate displacements for gatherv
        std::vector<int> displacements(size, 0);
        int total_count = 0;
        for (int i = 0; i < size; i++) {
            if (recv_counts[i] < 0 || recv_counts[i] > num_vertices) {
                std::cerr << "Rank " << rank << ": Invalid recv_counts[" << i << "]=" << recv_counts[i] << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            total_count += recv_counts[i];
            if (i > 0) {
                displacements[i] = displacements[i-1] + recv_counts[i-1];
            }
        }
        
        std::cerr << "Rank " << rank << ": Total vertex count from all processes: " << total_count << std::endl;
        
        // Create MPI datatype for result pairs
        MPI_Datatype mpi_result_type;
        int blocklengths[2] = {1, 1};
        MPI_Aint disps[2];
        MPI_Datatype types[2] = {MPI_INT, MPI_INT};
        std::pair<int, int> dummy;
        MPI_Aint base_addr;
        MPI_Get_address(&dummy, &base_addr);
        MPI_Get_address(&dummy.first, &disps[0]);
        MPI_Get_address(&dummy.second, &disps[1]);
        disps[0] = MPI_Aint_diff(disps[0], base_addr);
        disps[1] = MPI_Aint_diff(disps[1], base_addr);
        MPI_Type_create_struct(2, blocklengths, disps, types, &mpi_result_type);
        MPI_Type_commit(&mpi_result_type);
    
        // All ranks prepare buffer for results
        std::vector<std::pair<int, int>> global_result(total_count);
        
        // Use Allgatherv instead of Gatherv to avoid deadlock
        std::cerr << "Rank " << rank << ": Starting MPI_Allgatherv, local_count=" << local_count << std::endl;
        MPI_Allgatherv(
            local_count > 0 ? local_result.data() : nullptr, local_count, mpi_result_type,
            global_result.data(), recv_counts.data(), displacements.data(), mpi_result_type,
            MPI_COMM_WORLD
        );
        std::cerr << "Rank " << rank << ": Completed MPI_Allgatherv" << std::endl;
    
        MPI_Type_free(&mpi_result_type);
    
        // Sort results by vertex ID
        std::sort(global_result.begin(), global_result.end());
    
        // Only rank 0 prints the results, but all ranks have them
        if (rank == 0) {
            std::cerr << "Rank 0: Result collection complete, global_result.size=" << global_result.size() << std::endl;
            for (const auto& entry : global_result) {
                std::cerr << "Rank 0: Result vertex " << entry.first << ": distance=" << entry.second << std::endl;
            }
        }
    
        std::cerr << "Rank " << rank << ": Exiting getResult" << std::endl;
        return global_result;
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cerr << "Rank " << rank << ": Initialized MPI, size=" << size << std::endl;

    omp_set_num_threads(4);
    int source_vertex = 0;
    ParallelDynamicSSSP sssp(rank, size, source_vertex);

    std::vector<std::tuple<int, int, int>> edges;
    int num_vertices = 0;
    if (rank == 0) {
        // Example graph: edges {src, dst, weight}
        edges = {{0, 1, 5}, {1, 2, 3}};
        num_vertices = 3;
    }
    std::cerr << "Rank " << rank << ": Before broadcast, num_vertices=" << num_vertices << std::endl;

    // Broadcast number of vertices
    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cerr << "Rank " << rank << ": After broadcast, num_vertices=" << num_vertices << std::endl;

    // Broadcast edges
    int num_edges = edges.size();
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        edges.resize(num_edges);
    }

    // Create MPI datatype for edge tuples
    MPI_Datatype mpi_edge_type;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint disps[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    std::tuple<int, int, int> dummy;
    MPI_Aint base_addr;
    MPI_Get_address(&dummy, &base_addr);
    MPI_Get_address(&std::get<0>(dummy), &disps[0]);
    MPI_Get_address(&std::get<1>(dummy), &disps[1]);
    MPI_Get_address(&std::get<2>(dummy), &disps[2]);
    disps[0] = MPI_Aint_diff(disps[0], base_addr);
    disps[1] = MPI_Aint_diff(disps[1], base_addr);
    disps[2] = MPI_Aint_diff(disps[2], base_addr);
    MPI_Type_create_struct(3, blocklengths, disps, types, &mpi_edge_type);
    MPI_Type_commit(&mpi_edge_type);

    MPI_Bcast(edges.data(), num_edges, mpi_edge_type, 0, MPI_COMM_WORLD);
    std::cerr << "Rank " << rank << ": Edges received, size=" << edges.size() << std::endl;
    MPI_Type_free(&mpi_edge_type);

    // Manual partitioning for small test case (avoids unbalanced partitioning)
    if (rank == 0) {
        std::cout << "Partitioning graph..." << std::endl;
    }
    
    sssp.loadAndPartitionGraph(edges, num_vertices);
    std::cerr << "Rank " << rank << ": Graph partitioned" << std::endl;

    // Compute initial SSSP
    sssp.computeInitialSSSP();
    
    // Ensure all processes are synchronized before getting results
    MPI_Barrier(MPI_COMM_WORLD);
    std::cerr << "Rank " << rank << ": Initial SSSP computed" << std::endl;

    // Get and display initial results
    auto result = sssp.getResult();
    if (rank == 0) {
        std::cout << "Initial SSSP result:" << std::endl;
        for (const auto& entry : result) {
            std::cout << "Vertex " << entry.first << ": Distance = " 
                      << (entry.second == -1 ? "INF" : std::to_string(entry.second)) 
                      << std::endl;
        }
    }

    // Process edge changes
    std::vector<EdgeChange> changes;
    if (rank == 0) {
        // Add a new edge from vertex 0 to vertex 2 with weight 7
        changes.push_back({0, 2, 7, true});
    }
    
    int num_changes = changes.size();
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        changes.resize(num_changes);
    }

    // Create MPI datatype for edge changes
    MPI_Datatype mpi_change_type;
    int change_blocklengths[4] = {1, 1, 1, 1};
    MPI_Aint change_disps[4];
    MPI_Datatype change_types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_C_BOOL};
    EdgeChange dummy_change;
    MPI_Get_address(&dummy_change, &base_addr);
    MPI_Get_address(&dummy_change.u, &change_disps[0]);
    MPI_Get_address(&dummy_change.v, &change_disps[1]);
    MPI_Get_address(&dummy_change.weight, &change_disps[2]);
    MPI_Get_address(&dummy_change.insertion, &change_disps[3]);
    change_disps[0] = MPI_Aint_diff(change_disps[0], base_addr);
    change_disps[1] = MPI_Aint_diff(change_disps[1], base_addr);
    change_disps[2] = MPI_Aint_diff(change_disps[2], base_addr);
    change_disps[3] = MPI_Aint_diff(change_disps[3], base_addr);
    MPI_Type_create_struct(4, change_blocklengths, change_disps, change_types, &mpi_change_type);
    MPI_Type_commit(&mpi_change_type);

    MPI_Bcast(changes.data(), num_changes, mpi_change_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_change_type);

    // Process edge changes
    sssp.processEdgeChanges(changes);
    std::cerr << "Rank " << rank << ": Edge changes processed" << std::endl;

    // Synchronize before getting updated results
    MPI_Barrier(MPI_COMM_WORLD);

    // Check for load imbalance
    bool need_rebalance = sssp.checkLoadBalance();
    if (need_rebalance) {
        sssp.repartitionGraph();
    }

    // Get and display updated results
    result = sssp.getResult();
    if (rank == 0) {
        std::cout << "Updated SSSP result after edge changes:" << std::endl;
        for (const auto& entry : result) {
            std::cout << "Vertex " << entry.first << ": Distance = " 
                      << (entry.second == -1 ? "INF" : std::to_string(entry.second)) 
                      << std::endl;
        }
    }

    MPI_Finalize();
    std::cerr << "Rank " << rank << ": Finalized MPI" << std::endl;
    return 0;
}