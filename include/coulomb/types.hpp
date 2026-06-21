#pragma once

#include <cmath>
#include <cstddef>

namespace coulomb {

/// Floating-point precision used throughout the simulation.
/// Kept as a single alias so we can experiment with float vs double and
/// measure the accuracy/throughput trade-off from one place.
using Real = double;

/// Minimal 3-vector. Intentionally a plain aggregate so it stays trivially
/// copyable and the layout is obvious for later SIMD / SoA work.
struct Vec3 {
  Real x{0};
  Real y{0};
  Real z{0};

  constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  constexpr Vec3 operator*(Real s) const { return {x * s, y * s, z * s}; }

  constexpr Vec3& operator+=(const Vec3& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  constexpr Vec3& operator-=(const Vec3& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
};

constexpr Vec3 operator*(Real s, const Vec3& v) { return v * s; }

constexpr Real dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Real norm(const Vec3& v) { return std::sqrt(dot(v, v)); }

}  // namespace coulomb
