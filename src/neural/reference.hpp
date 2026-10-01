#pragma once

#include "blitz/mesh.hpp"
#include <stdexcept>

namespace blitz::neural {

// The source and explicitly bound reference are borrowed immutable views.
// Their owners must retain every stream. A changed/replaced reference must be
// rebound before another audit. Arbitrary views never acquire this trust through
// a cache hit.
class AuditReferences {
    MeshView source_, reference_;

    template <class T> static bool same(Stream<T> a, Stream<T> b) {
        return a.count == b.count && (!a.count || (a.data == b.data && a.stride == b.stride));
    }
    template <class T> static bool same(std::span<T> a, std::span<T> b) {
        return a.size() == b.size() && (a.empty() || a.data() == b.data());
    }
    static bool same(MeshView a, MeshView b) {
        return a.positions.count && same(a.positions, b.positions) && same(a.normals, b.normals) &&
               same(a.uv, b.uv) && same(a.colors, b.colors) && same(a.tangents, b.tangents) &&
               same(a.indices, b.indices) && same(a.materials, b.materials) &&
               same(a.double_sided, b.double_sided) &&
               same(a.exact_position_bits, b.exact_position_bits);
    }
    static void checked(MeshView mesh) {
        if (auto error = validate(mesh); !error.empty())
            throw std::invalid_argument(error);
    }

  public:
    explicit AuditReferences(MeshView source = {}) : source_(source) {
        if (source.positions.count)
            checked(source);
    }
    void bind(MeshView reference) {
        // Validate each new binding even if its allocation address was reused.
        // A failed rebind must revoke trust in the previous (possibly changed)
        // allocation before the caller handles the validation error.
        clear();
        if (!same(reference, source_))
            checked(reference);
        reference_ = reference;
    }
    void clear() noexcept {
        reference_ = {};
    }
    void validate_view(MeshView mesh) const {
        if (!same(mesh, source_) && !same(mesh, reference_))
            checked(mesh);
    }
};

} // namespace blitz::neural
