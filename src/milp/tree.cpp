#include "milp/tree.hpp"

namespace sankhya {
namespace milp {

void NodeQueue::push(MILPNode node) {
    queue_.push(node);
}

MILPNode NodeQueue::pop() {
    if (queue_.empty()) {
        throw std::out_of_range("NodeQueue is empty");
    }
    MILPNode top_node = queue_.top();
    queue_.pop();
    return top_node;
}

bool NodeQueue::empty() {
    return queue_.empty();
}

GlobalIncumbent::GlobalIncumbent() 
    : incumbent_obj_(math::kInfinity) {
}

void GlobalIncumbent::update(const std::vector<Float>& x, Float obj) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Update incumbent only when strictly better (less than current)
    if (obj < incumbent_obj_) {
        incumbent_obj_ = obj;
        incumbent_x_ = x;
    }
}

Float GlobalIncumbent::get_obj() {
    std::lock_guard<std::mutex> lock(mutex_);
    return incumbent_obj_;
}

std::vector<Float> GlobalIncumbent::get_x() {
    std::lock_guard<std::mutex> lock(mutex_);
    return incumbent_x_;
}

} // namespace milp
} // namespace sankhya
