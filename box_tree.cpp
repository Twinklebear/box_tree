#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <embree3/rtcore.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <glm/ext.hpp>
#include <glm/glm.hpp>

struct box3f {
    glm::vec3 lower = glm::vec3(std::numeric_limits<float>::infinity());
    glm::vec3 upper = -glm::vec3(std::numeric_limits<float>::infinity());

    box3f(const glm::vec3 &l, const glm::vec3 &u) : lower(l), upper(u) {}
    box3f() = default;

    void extend(const box3f &b)
    {
        lower = glm::min(lower, b.lower);
        upper = glm::max(upper, b.upper);
    }

    float volume()
    {
        glm::vec3 diag = upper - lower;
        if (glm::any(glm::lessThan(diag, glm::vec3(0.f)))) {
            return 0.f;
        }
        return diag.x * diag.y * diag.z;
    }
};

box3f intersection(const box3f &a, const box3f &b)
{
    return box3f(glm::max(a.lower, b.lower), glm::min(a.upper, b.upper));
}

bool overlaps(const box3f &a, const box3f &b)
{
    return intersection(a, b).volume() > 0.f;
}

std::ostream &operator<<(std::ostream &os, const box3f &b)
{
    os << "{" << glm::to_string(b.lower) << ", " << glm::to_string(b.upper) << "}";
    return os;
}

struct Node {
    virtual ~Node() = default;
};

struct InnerNode : Node {
    std::array<box3f, 2> child_bounds;
    std::array<Node *, 2> children = {nullptr, nullptr};
};

struct LeafNode : Node {
    const RTCBuildPrimitive *primitives = nullptr;
    size_t n_prims = 0;

    LeafNode(const RTCBuildPrimitive *primitives, size_t n_prims)
        : primitives(primitives), n_prims(n_prims)
    {
    }

    LeafNode() = default;
};

void *create_inner_node(RTCThreadLocalAllocator allocator, uint32_t child_count, void *user)
{
    void *mem = rtcThreadLocalAlloc(allocator, sizeof(InnerNode), 16);
    return reinterpret_cast<void *>(new (mem) InnerNode);
}

void *create_leaf(RTCThreadLocalAllocator allocator,
                  const RTCBuildPrimitive *primitives,
                  size_t n_prims,
                  void *user)
{
    void *mem = rtcThreadLocalAlloc(allocator, sizeof(LeafNode), 16);
    return reinterpret_cast<void *>(new (mem) LeafNode(primitives, n_prims));
}

void set_node_children(void *node, void **children, uint32_t child_count, void *user)
{
    if (child_count > 2) {
        throw std::runtime_error("more than 2 children?");
    }
    Node **child_nodes = reinterpret_cast<Node **>(children);
    InnerNode *inner_node = reinterpret_cast<InnerNode *>(node);
    for (uint32_t i = 0; i < child_count; ++i) {
        inner_node->children[i] = child_nodes[i];
    }
}

void set_node_bounds(void *node,
                     const RTCBounds **child_bounds,
                     uint32_t child_count,
                     void *user)
{
    if (child_count > 2) {
        throw std::runtime_error("more than 2 children?");
    }
    InnerNode *inner_node = reinterpret_cast<InnerNode *>(node);
    for (uint32_t i = 0; i < child_count; ++i) {
        inner_node->child_bounds[i] = box3f(
            glm::vec3(
                child_bounds[i]->lower_x, child_bounds[i]->lower_y, child_bounds[i]->lower_z),
            glm::vec3(
                child_bounds[i]->upper_x, child_bounds[i]->upper_y, child_bounds[i]->upper_z));
    }
}

void collect_leaves(Node *node, std::vector<LeafNode *> &leaves)
{
    LeafNode *l = dynamic_cast<LeafNode *>(node);
    if (l) {
        leaves.push_back(l);
    } else {
        InnerNode *i = dynamic_cast<InnerNode *>(node);
        collect_leaves(i->children[0], leaves);
        if (i->children[1]) {
            collect_leaves(i->children[1], leaves);
        }
    }
}

bool compute_divisor(uint32_t x, uint32_t &divisor);
glm::uvec3 compute_grid(uint32_t num);

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Usage: ./box_tree <num boxes>\n";
        return 1;
    }

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    RTCDevice device = rtcNewDevice(nullptr);

    const glm::uvec3 grid_dims = compute_grid(std::stoi(argv[1]));
    std::vector<RTCBuildPrimitive> boxes;
    for (uint32_t k = 0; k < grid_dims.z; ++k) {
        for (uint32_t j = 0; j < grid_dims.y; ++j) {
            for (uint32_t i = 0; i < grid_dims.x; ++i) {
                RTCBuildPrimitive prim;
                prim.lower_x = i;
                prim.lower_y = j;
                prim.lower_z = k;
                prim.upper_x = i + 1;
                prim.upper_y = j + 1;
                prim.upper_z = k + 1;
                prim.geomID = (k * grid_dims.y + j) * grid_dims.x + i;
                prim.primID = prim.geomID;
                boxes.push_back(prim);
            }
        }
    }

    RTCBVH bvh = rtcNewBVH(device);

    // Can I use higher than max leaf prims? 32 voxels is not many
    RTCBuildArguments args = rtcDefaultBuildArguments();
    args.byteSize = sizeof(args);
    args.buildQuality = RTC_BUILD_QUALITY_HIGH;
    args.buildFlags = RTC_BUILD_FLAG_NONE;
    args.minLeafSize = 500;
    args.maxLeafSize = 5000;
    args.maxBranchingFactor = 2;
    args.intersectionCost = 0.1f;
    args.bvh = bvh;
    args.primitives = boxes.data();
    args.primitiveCount = boxes.size();
    args.primitiveArrayCapacity = boxes.size();
    args.createNode = create_inner_node;
    args.setNodeChildren = set_node_children;
    args.setNodeBounds = set_node_bounds;
    args.createLeaf = create_leaf;

    InnerNode *root = reinterpret_cast<InnerNode *>(rtcBuildBVH(&args));
    {
        auto err = rtcGetDeviceError(device);
        switch (err) {
        case RTC_ERROR_UNKNOWN:
            std::cout << "RTC_ERROR_UNKNOWN\n";
            break;
        case RTC_ERROR_INVALID_ARGUMENT:
            std::cout << "RTC_ERROR_INVALID_ARGUMENT\n";
            break;
        case RTC_ERROR_INVALID_OPERATION:
            std::cout << "RTC_ERROR_INVALID_OPERATION\n";
            break;
        case RTC_ERROR_OUT_OF_MEMORY:
            std::cout << "RTC_ERROR_OUT_OF_MEMORY\n";
            break;
        case RTC_ERROR_UNSUPPORTED_CPU:
            std::cout << "RTC_ERROR_UNSUPPORTED_CPU\n";
            break;
        case RTC_ERROR_CANCELLED:
            std::cout << "RTC_ERROR_CANCELLED\n";
            break;
        default:
            break;
        }
    }

    std::vector<box3f> leaf_bounds;
    std::vector<LeafNode *> leaves;
    collect_leaves(root, leaves);
    std::cout << "Tree has " << leaves.size() << " leaves\n";
    for (size_t i = 0; i < leaves.size(); ++i) {
        box3f bounds;
        for (size_t j = 0; j < leaves[i]->n_prims; ++j) {
            const RTCBuildPrimitive &prim = leaves[i]->primitives[j];
            box3f b(glm::vec3(prim.lower_x, prim.lower_y, prim.lower_z),
                    glm::vec3(prim.upper_x, prim.upper_y, prim.upper_z));
            bounds.extend(b);
        }
        std::cout << "Leaf[" << i << "] has " << leaves[i]->n_prims << " prims\n"
                  << "bounds: " << bounds << "\n";
        leaf_bounds.push_back(bounds);
    }

    bool had_overlap = false;
    for (size_t i = 0; i < leaf_bounds.size() && !had_overlap; ++i) {
        for (size_t j = 0; j < leaf_bounds.size() && !had_overlap; ++j) {
            if (i == j) {
                continue;
            }
            if (overlaps(leaf_bounds[i], leaf_bounds[j])) {
                box3f inters = intersection(leaf_bounds[i], leaf_bounds[j]);
                std::cout << "Leaf " << i << " overlaps " << j << "\n"
                          << "Leaf[" << i << "] bounds = " << leaf_bounds[i] << "\nLeaf[" << j
                          << "] bounds = " << leaf_bounds[j] << "\nIntersection: " << inters
                          << "\nVolume: " << inters.volume() << "\n";
                had_overlap = true;
            }
        }
    }

    rtcReleaseBVH(bvh);
    rtcReleaseDevice(device);

    return 0;
}

bool compute_divisor(uint32_t x, uint32_t &divisor)
{
    const uint32_t upper = std::sqrt(x);
    for (uint32_t i = 2; i <= upper; ++i) {
        if (x % i == 0) {
            divisor = i;
            return true;
        }
    }
    return false;
}

glm::uvec3 compute_grid(uint32_t num)
{
    glm::uvec3 grid(1);
    uint32_t axis = 0;
    uint32_t divisor = 0;
    while (compute_divisor(num, divisor)) {
        grid[axis] *= divisor;
        num /= divisor;
        axis = (axis + 1) % 3;
    }
    if (num != 1) {
        grid[axis] *= num;
    }
    return grid;
}
