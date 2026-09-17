#include "ipm/mehrotra.hpp"

namespace sankhya {
namespace ipm {

// Forward declare the Impl class defined in mehrotra_device.cu
class MehrotraSolver::Impl {
public:
    Impl(const core::Model& model);
    ~Impl();
    MehrotraResult solve();
};

MehrotraSolver::MehrotraSolver(const core::Model& model) {
    impl_ = new Impl(model);
}

MehrotraSolver::~MehrotraSolver() {
    delete impl_;
}

MehrotraResult MehrotraSolver::solve() {
    return impl_->solve();
}

} // namespace ipm
} // namespace sankhya