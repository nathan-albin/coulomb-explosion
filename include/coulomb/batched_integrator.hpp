#pragma once

// Batched shared-dt lockstep adaptive RK45 (DP5(4)) — 0006 Part A.
//
// Drives K independent Coulomb explosions of one fixed molecule in lockstep: one
// SIMD lane per simulation, SoA over lanes, on the shared batched force kernel
// (coulomb/batched_force.hpp). This is the *simple first cut* the 0002 dispersion
// study recommended building first:
//
//   * SHARED dt — the whole batch advances with one step size, set by the
//     tightest (worst-error) still-active lane. This is the min-envelope penalty:
//     a lane that could take a big step is dragged to the batch's step.
//   * NO REFILL — a lane that meets its pe_stop is latched and masked out of the
//     error norm (so it stops constraining dt), but its slot is not refilled; the
//     batch runs until the *last* lane converges. That is the straggler idle.
//
// It mirrors the scalar DormandPrince45 (src/integrators.cpp) lane-for-lane: same
// Butcher tableau, FSAL, RMS error norm with scale = atol + rtol*max(|y_old|,
// |y_new|), and the same step controller — the only change is reducing the
// per-lane error to the worst active lane for one shared accept/reject decision.
//
// Templated on element type T like the force kernel; T=double is the
// apples-to-apples oracle against the scalar run_to_convergence, T=float is the
// production path (its accuracy story is 0004/0005). Highway STATIC dispatch:
// include only in -march=native TUs, never in the generic engine library.

#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "coulomb/batched_force.hpp"
#include "coulomb/driver.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

namespace coulomb {
namespace hn = hwy::HWY_NAMESPACE;

// Three lane-strided coordinate buffers (length n*k), the SoA unit the kernels
// and stages operate on.
template <class T>
struct Vec3Soa {
  hwy::AlignedFreeUniquePtr<T[]> x, y, z;
  void alloc(std::size_t m) {
    x = hwy::AllocateAligned<T>(m);
    y = hwy::AllocateAligned<T>(m);
    z = hwy::AllocateAligned<T>(m);
  }
  void swap(Vec3Soa& o) {
    x.swap(o.x);
    y.swap(o.y);
    z.swap(o.z);
  }
};

template <class T>
class BatchedRK45 {
 public:
  struct Result {
    std::vector<double> momenta;     ///< [(lane*n + i)*3 + c] = m_i * v_i (post-redistribution).
    std::vector<std::size_t> steps;  ///< per-lane accepted-step index at convergence.
    std::vector<char> converged;     ///< per-lane: did it meet pe_stop before max_steps?
    std::size_t batch_steps{0};      ///< accepted SIMD iterations (the batch ran this many).
  };

  BatchedRK45(const Molecule& mol, Real coulomb_k, std::size_t lanes, T rtol, T atol)
      : n_(mol.size()), k_(lanes), rtol_(rtol), atol_(atol) {
    if (k_ != hn::Lanes(hn::ScalableTag<T>())) {
      throw std::invalid_argument("BatchedRK45: lanes must equal the SIMD width for T");
    }
    pair_const_.assign(n_ * n_, T(0));
    inv_mass_.assign(n_, T(0));
    mass_d_.assign(n_, 0.0);
    for (std::size_t i = 0; i < n_; ++i) {
      mass_d_[i] = static_cast<double>(mol.atoms[i].mass);
      inv_mass_[i] = static_cast<T>(1.0 / mol.atoms[i].mass);
      for (std::size_t j = 0; j < n_; ++j) {
        pair_const_[i * n_ + j] = static_cast<T>(static_cast<double>(coulomb_k) *
                                                 static_cast<double>(mol.atoms[i].charge) *
                                                 static_cast<double>(mol.atoms[j].charge));
      }
    }

    pos_.alloc(n_ * k_);
    vel_.alloc(n_ * k_);
    stage_.alloc(n_ * k_);
    probe_.alloc(n_ * k_);
    final_vel_.alloc(n_ * k_);
    for (int s = 0; s < 7; ++s) {
      kpos_[s].alloc(n_ * k_);
      kvel_[s].alloc(n_ * k_);
    }
    scratch_.assign(k_, T(0));
    d1_.assign(k_, T(0));
    d2_.assign(k_, T(0));
    h0_.assign(k_, T(0));
    pe_lane_.assign(k_, T(0));
    pe0_.assign(k_, 0.0);
    pe_stop_.assign(k_, 0.0);
    e0_.assign(k_, 0.0);
  }

  std::size_t lanes() const { return k_; }
  std::size_t atoms() const { return n_; }

  // Set lane `lane`'s initial positions (velocities start at rest). `pos` has n entries.
  void set_lane_geometry(std::size_t lane, const std::vector<Vec3>& pos) {
    for (std::size_t i = 0; i < n_; ++i) {
      pos_.x[i * k_ + lane] = static_cast<T>(pos[i].x);
      pos_.y[i * k_ + lane] = static_cast<T>(pos[i].y);
      pos_.z[i * k_ + lane] = static_cast<T>(pos[i].z);
      vel_.x[i * k_ + lane] = T(0);
      vel_.y[i * k_ + lane] = T(0);
      vel_.z[i * k_ + lane] = T(0);
    }
  }

  Result run(const RunConfig& cfg) {
    Result r;
    r.momenta.assign(k_ * n_ * 3, 0.0);
    r.steps.assign(k_, 0);
    r.converged.assign(k_, 0);

    // Per-lane PE_0, stop threshold, and total energy (KE_0 = 0, started at rest).
    batched::potential_energy<T>(n_, k_, pos_.x.get(), pos_.y.get(), pos_.z.get(),
                                 pair_const_.data(), pe_lane_.data());
    std::vector<char> done(k_, 0);
    std::size_t n_done = 0;
    for (std::size_t lane = 0; lane < k_; ++lane) {
      pe0_[lane] = static_cast<double>(pe_lane_[lane]);
      pe_stop_[lane] = static_cast<double>(cfg.pe_stop_fraction) * pe0_[lane];
      e0_[lane] = pe0_[lane];              // KE_0 = 0.
      if (pe0_[lane] <= pe_stop_[lane]) {  // already converged (pe_stop_fraction >= 1).
        done[lane] = 1;
        r.converged[lane] = 1;
        latch_lane(lane);
        ++n_done;
      }
    }

    h_ = T(0);
    fsal_valid_ = false;
    std::size_t iter = 0;
    while (n_done < k_ && iter < cfg.max_steps) {
      step(done.data(), static_cast<T>(cfg.max_dt));
      ++iter;
      batched::potential_energy<T>(n_, k_, pos_.x.get(), pos_.y.get(), pos_.z.get(),
                                   pair_const_.data(), pe_lane_.data());
      for (std::size_t lane = 0; lane < k_; ++lane) {
        if (!done[lane] && static_cast<double>(pe_lane_[lane]) <= pe_stop_[lane]) {
          done[lane] = 1;
          r.converged[lane] = 1;
          r.steps[lane] = iter;
          latch_lane(lane);
          ++n_done;
        }
      }
    }
    for (std::size_t lane = 0; lane < k_; ++lane) {  // never converged: latch where they are.
      if (!done[lane]) {
        r.steps[lane] = iter;
        latch_lane(lane);
      }
    }
    r.batch_steps = iter;

    // Per-lane finalize: energy redistribution (s = sqrt(E_0/KE)) + momenta.
    for (std::size_t lane = 0; lane < k_; ++lane) {
      double ke = 0.0;
      for (std::size_t i = 0; i < n_; ++i) {
        const double vx = static_cast<double>(final_vel_.x[i * k_ + lane]);
        const double vy = static_cast<double>(final_vel_.y[i * k_ + lane]);
        const double vz = static_cast<double>(final_vel_.z[i * k_ + lane]);
        ke += 0.5 * mass_d_[i] * (vx * vx + vy * vy + vz * vz);
      }
      double s = 1.0;
      if (cfg.redistribute_energy && ke > 0.0) s = std::sqrt(e0_[lane] / ke);
      for (std::size_t i = 0; i < n_; ++i) {
        const double m = mass_d_[i] * s;
        r.momenta[(lane * n_ + i) * 3 + 0] = m * static_cast<double>(final_vel_.x[i * k_ + lane]);
        r.momenta[(lane * n_ + i) * 3 + 1] = m * static_cast<double>(final_vel_.y[i * k_ + lane]);
        r.momenta[(lane * n_ + i) * 3 + 2] = m * static_cast<double>(final_vel_.z[i * k_ + lane]);
      }
    }
    return r;
  }

 private:
  // Butcher tableau (DP5(4)), identical to src/integrators.cpp. Stored as double,
  // narrowed to T per use. Stage-7 a-row == 5th-order weights b (FSAL).
  static constexpr double kA2_[1] = {1.0 / 5};
  static constexpr double kA3_[2] = {3.0 / 40, 9.0 / 40};
  static constexpr double kA4_[3] = {44.0 / 45, -56.0 / 15, 32.0 / 9};
  static constexpr double kA5_[4] = {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729};
  static constexpr double kA6_[5] = {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176,
                                     -5103.0 / 18656};
  static constexpr double kA7_[6] = {35.0 / 384,     0.0,      500.0 / 1113, 125.0 / 192,
                                     -2187.0 / 6784, 11.0 / 84};
  static constexpr const double* kA_[6] = {kA2_, kA3_, kA4_, kA5_, kA6_, kA7_};
  static constexpr double kE_[7] = {71.0 / 57600,      0.0,        -71.0 / 16695, 71.0 / 1920,
                                    -17253.0 / 339200, 22.0 / 525, -1.0 / 40};
  static constexpr double kSafety_ = 0.9;
  static constexpr double kMinFactor_ = 0.2;
  static constexpr double kMaxFactor_ = 10.0;
  static constexpr double kExponent_ = -1.0 / 5.0;
  static constexpr int kMaxRejections_ = 100;

  void accel(const Vec3Soa<T>& src, Vec3Soa<T>& dst) {
    batched::accelerations<T>(n_, k_, src.x.get(), src.y.get(), src.z.get(), pair_const_.data(),
                              inv_mass_.data(), dst.x.get(), dst.y.get(), dst.z.get());
  }

  void latch_lane(std::size_t lane) {
    for (std::size_t i = 0; i < n_; ++i) {
      final_vel_.x[i * k_ + lane] = vel_.x[i * k_ + lane];
      final_vel_.y[i * k_ + lane] = vel_.y[i * k_ + lane];
      final_vel_.z[i * k_ + lane] = vel_.z[i * k_ + lane];
    }
  }

  // One accepted shared-dt step over all lanes; done lanes are excluded from the
  // error reduction so they neither block acceptance nor shrink dt.
  void step(const char* done, T max_dt) {
    const hn::ScalableTag<T> d;
    const std::size_t n = n_, k = k_;

    if (!fsal_valid_) {  // k1 = f(y) = (v, a(x)).
      for (std::size_t idx = 0; idx < n * k; ++idx) {
        kpos_[0].x[idx] = vel_.x[idx];
        kpos_[0].y[idx] = vel_.y[idx];
        kpos_[0].z[idx] = vel_.z[idx];
      }
      accel(pos_, kvel_[0]);
      fsal_valid_ = true;
    }
    if (h_ <= T(0)) h_ = select_initial_step(done, max_dt);

    // Component pointer tables (rebuilt each call; buffers only swap on accept,
    // after which we return immediately, so these stay valid across rejections).
    T* P[3] = {pos_.x.get(), pos_.y.get(), pos_.z.get()};
    T* V[3] = {vel_.x.get(), vel_.y.get(), vel_.z.get()};
    T* ST[3] = {stage_.x.get(), stage_.y.get(), stage_.z.get()};
    T* KP[7][3];
    T* KV[7][3];
    for (int s = 0; s < 7; ++s) {
      KP[s][0] = kpos_[s].x.get();
      KP[s][1] = kpos_[s].y.get();
      KP[s][2] = kpos_[s].z.get();
      KV[s][0] = kvel_[s].x.get();
      KV[s][1] = kvel_[s].y.get();
      KV[s][2] = kvel_[s].z.get();
    }

    const auto rtolv = hn::Set(d, rtol_);
    const auto atolv = hn::Set(d, atol_);
    auto errc = [&](auto verr, auto yold, auto ynew) {
      const auto scale = hn::Add(atolv, hn::Mul(rtolv, hn::Max(hn::Abs(yold), hn::Abs(ynew))));
      const auto rr = hn::Div(verr, scale);
      return hn::Mul(rr, rr);
    };

    bool rejected_once = false;
    for (int attempt = 0;; ++attempt) {
      if (attempt > kMaxRejections_) {
        throw std::runtime_error("BatchedRK45: step rejected too many times");
      }
      const T h = std::min(h_, max_dt);
      const auto hv = hn::Set(d, h);

      // Stages 2..7 -> stage_ positions and kpos_[s] stage velocities; kvel_[s] = a(stage_).
      for (int s = 1; s < 7; ++s) {
        const double* a = kA_[s - 1];
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t o = i * k;
          for (int c = 0; c < 3; ++c) {
            auto dx = hn::Zero(d);
            auto dv = hn::Zero(d);
            for (int j = 0; j < s; ++j) {
              const auto aj = hn::Set(d, static_cast<T>(a[j]));
              dx = hn::MulAdd(hn::Load(d, KP[j][c] + o), aj, dx);
              dv = hn::MulAdd(hn::Load(d, KV[j][c] + o), aj, dv);
            }
            hn::Store(hn::MulAdd(dx, hv, hn::Load(d, P[c] + o)), d, ST[c] + o);
            hn::Store(hn::MulAdd(dv, hv, hn::Load(d, V[c] + o)), d, KP[s][c] + o);
          }
        }
        accel(stage_, kvel_[s]);
      }

      // Per-lane embedded error in the RMS norm. y_new = (stage_ positions, kpos_[6] velocities).
      auto err_sq = hn::Zero(d);
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t o = i * k;
        for (int c = 0; c < 3; ++c) {
          auto ep = hn::Zero(d);
          auto ev = hn::Zero(d);
          for (int s = 0; s < 7; ++s) {
            const auto es = hn::Set(d, static_cast<T>(kE_[s]));
            ep = hn::MulAdd(hn::Load(d, KP[s][c] + o), es, ep);
            ev = hn::MulAdd(hn::Load(d, KV[s][c] + o), es, ev);
          }
          err_sq =
              hn::Add(err_sq, errc(hn::Mul(ep, hv), hn::Load(d, P[c] + o), hn::Load(d, ST[c] + o)));
          err_sq = hn::Add(err_sq,
                           errc(hn::Mul(ev, hv), hn::Load(d, V[c] + o), hn::Load(d, KP[6][c] + o)));
        }
      }
      const auto enorm = hn::Sqrt(hn::Mul(err_sq, hn::Set(d, T(1) / static_cast<T>(6 * n))));
      hn::StoreU(enorm, d, scratch_.data());
      T max_err = T(0);
      for (std::size_t lane = 0; lane < k; ++lane) {
        if (!done[lane]) max_err = std::max(max_err, scratch_[lane]);
      }

      if (max_err <= T(1)) {
        pos_.swap(stage_);                               // pos_ = x_new
        vel_.swap(kpos_[6]);                             // vel_ = v_new
        kvel_[0].swap(kvel_[6]);                         // kvel_[0] = a(x_new) (next k1)
        for (std::size_t idx = 0; idx < n * k; ++idx) {  // kpos_[0] = v_new (next k1)
          kpos_[0].x[idx] = vel_.x[idx];
          kpos_[0].y[idx] = vel_.y[idx];
          kpos_[0].z[idx] = vel_.z[idx];
        }
        const double e = static_cast<double>(max_err);
        T factor = (max_err == T(0))
                       ? static_cast<T>(kMaxFactor_)
                       : static_cast<T>(std::min(kMaxFactor_, kSafety_ * std::pow(e, kExponent_)));
        if (rejected_once) factor = std::min(factor, T(1));
        h_ = std::min(h * factor, max_dt);
        return;
      }

      const double e = static_cast<double>(max_err);
      const T factor = static_cast<T>(std::max(kMinFactor_, kSafety_ * std::pow(e, kExponent_)));
      h_ = h * factor;
      rejected_once = true;
    }
  }

  // Hairer's initial-step heuristic (scipy's select_initial_step), per lane in the
  // RMS norm, reduced by min over active lanes (the min-envelope from step 1).
  // Requires k1 already in kpos_[0] (= v0) / kvel_[0] (= a0).
  T select_initial_step(const char* done, T max_dt) {
    const hn::ScalableTag<T> d;
    const std::size_t n = n_, k = k_;
    const T* P[3] = {pos_.x.get(), pos_.y.get(), pos_.z.get()};
    const T* Vv[3] = {vel_.x.get(), vel_.y.get(), vel_.z.get()};
    const T* F0[3] = {kpos_[0].x.get(), kpos_[0].y.get(), kpos_[0].z.get()};  // = v0
    const T* A0[3] = {kvel_[0].x.get(), kvel_[0].y.get(), kvel_[0].z.get()};  // = a0
    T* ST[3] = {stage_.x.get(), stage_.y.get(), stage_.z.get()};
    const T* PA[3] = {probe_.x.get(), probe_.y.get(), probe_.z.get()};

    const auto rtolv = hn::Set(d, rtol_);
    const auto atolv = hn::Set(d, atol_);
    const auto invden = hn::Set(d, T(1) / static_cast<T>(6 * n));

    auto d0sq = hn::Zero(d);
    auto d1sq = hn::Zero(d);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t o = i * k;
      for (int c = 0; c < 3; ++c) {
        const auto x = hn::Load(d, P[c] + o);
        const auto v = hn::Load(d, Vv[c] + o);
        const auto sx = hn::Add(atolv, hn::Mul(rtolv, hn::Abs(x)));
        const auto sv = hn::Add(atolv, hn::Mul(rtolv, hn::Abs(v)));
        const auto rx = hn::Div(x, sx), rv = hn::Div(v, sv);
        d0sq = hn::Add(d0sq, hn::Add(hn::Mul(rx, rx), hn::Mul(rv, rv)));
        const auto gp = hn::Div(hn::Load(d, F0[c] + o), sx);
        const auto gv = hn::Div(hn::Load(d, A0[c] + o), sv);
        d1sq = hn::Add(d1sq, hn::Add(hn::Mul(gp, gp), hn::Mul(gv, gv)));
      }
    }
    hn::StoreU(hn::Sqrt(hn::Mul(d0sq, invden)), d, scratch_.data());  // d0 per lane
    hn::StoreU(hn::Sqrt(hn::Mul(d1sq, invden)), d, d1_.data());       // d1 per lane
    for (std::size_t lane = 0; lane < k; ++lane) {
      h0_[lane] = (scratch_[lane] < T(1e-5) || d1_[lane] < T(1e-5))
                      ? T(1e-6)
                      : T(0.01) * scratch_[lane] / d1_[lane];
    }

    const auto h0v = hn::LoadU(d, h0_.data());
    for (std::size_t i = 0; i < n; ++i) {  // probe: y1 = y0 + h0*f0, positions only.
      const std::size_t o = i * k;
      for (int c = 0; c < 3; ++c) {
        hn::Store(hn::MulAdd(h0v, hn::Load(d, F0[c] + o), hn::Load(d, P[c] + o)), d, ST[c] + o);
      }
    }
    accel(stage_, probe_);  // a1 -> probe_

    auto d2sq = hn::Zero(d);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t o = i * k;
      for (int c = 0; c < 3; ++c) {
        const auto sx = hn::Add(atolv, hn::Mul(rtolv, hn::Abs(hn::Load(d, P[c] + o))));
        const auto sv = hn::Add(atolv, hn::Mul(rtolv, hn::Abs(hn::Load(d, Vv[c] + o))));
        const auto dvp = hn::Mul(hn::Load(d, A0[c] + o), h0v);  // v1 - v0 = a0*h0
        const auto dap = hn::Sub(hn::Load(d, PA[c] + o), hn::Load(d, A0[c] + o));  // a1 - a0
        const auto rp = hn::Div(dvp, sx), ra = hn::Div(dap, sv);
        d2sq = hn::Add(d2sq, hn::Add(hn::Mul(rp, rp), hn::Mul(ra, ra)));
      }
    }
    hn::StoreU(hn::Div(hn::Sqrt(hn::Mul(d2sq, invden)), h0v), d, d2_.data());  // d2 per lane

    T hmin = max_dt;
    for (std::size_t lane = 0; lane < k; ++lane) {
      const double d1l = static_cast<double>(d1_[lane]);
      const double d2l = static_cast<double>(d2_[lane]);
      const T h1 = (d1l <= 1e-15 && d2l <= 1e-15)
                       ? std::max(T(1e-6), h0_[lane] * T(1e-3))
                       : static_cast<T>(std::pow(0.01 / std::max(d1l, d2l), 1.0 / 5.0));
      const T hlane = std::min({T(100) * h0_[lane], h1, max_dt});
      if (!done[lane]) hmin = std::min(hmin, hlane);
    }
    return hmin;
  }

  std::size_t n_, k_;
  T rtol_, atol_;
  T h_{0};
  bool fsal_valid_{false};

  std::vector<T> pair_const_, inv_mass_;
  std::vector<double> mass_d_;
  Vec3Soa<T> pos_, vel_, stage_, probe_, final_vel_;
  Vec3Soa<T> kpos_[7], kvel_[7];
  std::vector<T> scratch_, d1_, d2_, h0_, pe_lane_;
  std::vector<double> pe0_, pe_stop_, e0_;
};

}  // namespace coulomb
