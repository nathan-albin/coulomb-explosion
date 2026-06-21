#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "coulomb/integrator.hpp"

namespace coulomb {
namespace {

/// Velocity Verlet: symplectic, time-reversible, fixed-step. Good baseline for
/// the "symplectic solver" goal and cheap (two force evaluations amortized to
/// one per step by reusing the previous acceleration).
class VelocityVerlet final : public Integrator {
 public:
  Real step(const Molecule& molecule, const CoulombForce& force, State& state, Real dt) override {
    const std::size_t n = state.size();

    if (!have_accel_) {
      force.accelerations(molecule, state, accel_);
      have_accel_ = true;
    }

    // x(t+dt) = x + v*dt + 0.5*a*dt^2
    for (std::size_t i = 0; i < n; ++i) {
      state.positions[i] += state.velocities[i] * dt + accel_[i] * (Real{0.5} * dt * dt);
    }

    // a(t+dt) from the new positions.
    force.accelerations(molecule, state, next_accel_);

    // v(t+dt) = v + 0.5*(a(t) + a(t+dt))*dt
    for (std::size_t i = 0; i < n; ++i) {
      state.velocities[i] += (accel_[i] + next_accel_[i]) * (Real{0.5} * dt);
    }

    accel_.swap(next_accel_);
    return dt;
  }

  std::string_view name() const override { return "velocity-verlet"; }

 private:
  bool have_accel_{false};
  std::vector<Vec3> accel_;
  std::vector<Vec3> next_accel_;
};

/// Adaptive Dormand-Prince DP5(4) (the method scipy exposes as `RK45`). This is
/// the accuracy reference: it ports the embedded 4th/5th-order pair, the RMS
/// error norm, and the step-size controller from scipy so the C++ engine
/// reproduces the Python oracle's adaptivity.
///
/// The dynamics are autonomous (forces depend only on positions), so the stage
/// abscissae c_i are never needed. The phase vector is y = (x, v); its
/// derivative is f(y) = (v, a(x)). FSAL (First Same As Last) reuses the final
/// stage of an accepted step as the first stage of the next, costing one force
/// evaluation per accepted step beyond the initial one.
class DormandPrince45 final : public Integrator {
 public:
  DormandPrince45(Real rtol, Real atol) : rtol_(rtol), atol_(atol) {}

  Real step(const Molecule& molecule, const CoulombForce& force, State& state,
            Real max_dt) override {
    const std::size_t n = state.size();
    ensure_size(n);

    // k1 = f(y). Reused from the previous step's last stage via FSAL.
    if (!fsal_valid_) {
      kpos_[0] = state.velocities;
      force.accelerations(molecule, state, kvel_[0]);
      fsal_valid_ = true;
    }
    if (h_ <= Real{0}) h_ = select_initial_step(molecule, force, state, max_dt);

    bool rejected_once = false;
    for (int attempt = 0;; ++attempt) {
      if (attempt > kMaxRejections) {
        throw std::runtime_error(
            "Dormand-Prince: step rejected too many times (singular collision "
            "or tolerance too tight?)");
      }
      const Real h = std::min(h_, max_dt);

      // Stages 2..7. Stage 7 (a-row == 5th-order weights b) builds y_new
      // directly: its position state is x_new and kpos_[6] is v_new, while
      // kvel_[6] = a(x_new) becomes the next step's k1 (FSAL).
      for (int s = 1; s < 7; ++s) {
        const Real* a = kA[s - 1];
        for (std::size_t i = 0; i < n; ++i) {
          Vec3 dx{};
          Vec3 dv{};
          for (int j = 0; j < s; ++j) {
            const Real c = a[j];
            dx += kpos_[j][i] * c;
            dv += kvel_[j][i] * c;
          }
          stage_.positions[i] = state.positions[i] + dx * h;
          kpos_[s][i] = state.velocities[i] + dv * h;
        }
        force.accelerations(molecule, stage_, kvel_[s]);
      }

      // Embedded error estimate e = h * sum_s E_s k_s, with the scipy scaling
      // scale = atol + rtol*max(|y_old|, |y_new|), measured in the RMS norm.
      Real err_sq = Real{0};
      for (std::size_t i = 0; i < n; ++i) {
        Vec3 ep{};
        Vec3 ev{};
        for (int s = 0; s < 7; ++s) {
          const Real c = kE[s];
          ep += kpos_[s][i] * c;
          ev += kvel_[s][i] * c;
        }
        err_sq += err_component(ep.x * h, state.positions[i].x, stage_.positions[i].x);
        err_sq += err_component(ep.y * h, state.positions[i].y, stage_.positions[i].y);
        err_sq += err_component(ep.z * h, state.positions[i].z, stage_.positions[i].z);
        err_sq += err_component(ev.x * h, state.velocities[i].x, kpos_[6][i].x);
        err_sq += err_component(ev.y * h, state.velocities[i].y, kpos_[6][i].y);
        err_sq += err_component(ev.z * h, state.velocities[i].z, kpos_[6][i].z);
      }
      const Real error_norm = std::sqrt(err_sq / static_cast<Real>(6 * n));

      if (error_norm <= Real{1}) {
        // Accept: commit y_new and roll the last stage forward as next k1.
        // After these swaps state holds (x_new, v_new); the discarded buffers
        // (stage_.positions, kpos_[6], kvel_[6]) hold stale data overwritten on
        // the next step. k1's position-derivative is v_new, copied from state.
        state.positions.swap(stage_.positions);  // state.positions = x_new
        state.velocities.swap(kpos_[6]);         // state.velocities = v_new
        kvel_[0].swap(kvel_[6]);                 // kvel_[0] = a(x_new) (next k1)
        kpos_[0] = state.velocities;             // kpos_[0] = v_new (next k1)

        Real factor = (error_norm == Real{0})
                          ? kMaxFactor
                          : std::min(kMaxFactor, kSafety * std::pow(error_norm, kExponent));
        if (rejected_once) factor = std::min(factor, Real{1});
        h_ = std::min(h * factor, max_dt);
        return h;
      }

      // Reject: shrink and retry from the same y0 (k1 still valid).
      const Real factor = std::max(kMinFactor, kSafety * std::pow(error_norm, kExponent));
      h_ = h * factor;
      rejected_once = true;
    }
  }

  std::string_view name() const override { return "dormand-prince-45"; }

 private:
  // Butcher tableau rows a_{s,j} for stages 2..7 (the c_i are unused). The
  // stage-7 row equals the 5th-order solution weights b_i, so building stage 7
  // yields y_new for free (FSAL).
  static constexpr Real kA2[1] = {1.0 / 5};
  static constexpr Real kA3[2] = {3.0 / 40, 9.0 / 40};
  static constexpr Real kA4[3] = {44.0 / 45, -56.0 / 15, 32.0 / 9};
  static constexpr Real kA5[4] = {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729};
  static constexpr Real kA6[5] = {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176,
                                  -5103.0 / 18656};
  static constexpr Real kA7[6] = {35.0 / 384,     0.0,      500.0 / 1113, 125.0 / 192,
                                  -2187.0 / 6784, 11.0 / 84};
  static constexpr const Real* kA[6] = {kA2, kA3, kA4, kA5, kA6, kA7};

  // Error weights E_i = b_i - b*_i (5th minus 4th order).
  static constexpr Real kE[7] = {71.0 / 57600,      0.0,        -71.0 / 16695, 71.0 / 1920,
                                 -17253.0 / 339200, 22.0 / 525, -1.0 / 40};

  static constexpr Real kSafety = 0.9;
  static constexpr Real kMinFactor = 0.2;
  static constexpr Real kMaxFactor = 10.0;
  // Local error is order 5; the controller exponent is -1/(error_order+1) with
  // error_order = 4, matching scipy's RungeKutta.
  static constexpr Real kExponent = -1.0 / 5.0;
  static constexpr int kMaxRejections = 100;

  void ensure_size(std::size_t n) {
    if (kpos_[0].size() == n) return;
    for (int s = 0; s < 7; ++s) {
      kpos_[s].assign(n, Vec3{});
      kvel_[s].assign(n, Vec3{});
    }
    stage_.positions.assign(n, Vec3{});
    fsal_valid_ = false;
    h_ = Real{0};
  }

  // One component's contribution to the squared RMS error norm.
  Real err_component(Real err, Real y_old, Real y_new) const {
    const Real scale = atol_ + rtol_ * std::max(std::abs(y_old), std::abs(y_new));
    const Real r = err / scale;
    return r * r;
  }

  // Hairer's initial-step heuristic (scipy's select_initial_step), measured in
  // the same RMS norm. Requires k1 (= f(y0)) already in kpos_[0]/kvel_[0].
  Real select_initial_step(const Molecule& molecule, const CoulombForce& force, const State& state,
                           Real max_dt) {
    const std::size_t n = state.size();
    Real d0_sq = Real{0};
    Real d1_sq = Real{0};
    auto accum = [](Real& acc, Real value, Real scale) {
      const Real r = value / scale;
      acc += r * r;
    };
    for (std::size_t i = 0; i < n; ++i) {
      const Vec3& x = state.positions[i];
      const Vec3& v = state.velocities[i];
      const Vec3& fp = kpos_[0][i];  // = v
      const Vec3& fv = kvel_[0][i];  // = a
      for (int c = 0; c < 3; ++c) {
        const Real xc = (&x.x)[c], vc = (&v.x)[c];
        const Real sx = atol_ + rtol_ * std::abs(xc);
        const Real sv = atol_ + rtol_ * std::abs(vc);
        accum(d0_sq, xc, sx);
        accum(d0_sq, vc, sv);
        accum(d1_sq, (&fp.x)[c], sx);
        accum(d1_sq, (&fv.x)[c], sv);
      }
    }
    const Real denom = static_cast<Real>(6 * n);
    const Real d0 = std::sqrt(d0_sq / denom);
    const Real d1 = std::sqrt(d1_sq / denom);
    const Real h0 = (d0 < 1e-5 || d1 < 1e-5) ? Real{1e-6} : Real{0.01} * d0 / d1;

    // One Euler probe to estimate the second derivative: y1 = y0 + h0*f0.
    Real d2_sq = Real{0};
    for (std::size_t i = 0; i < n; ++i) {
      stage_.positions[i] = state.positions[i] + kpos_[0][i] * h0;
    }
    force.accelerations(molecule, stage_, probe_accel_);
    for (std::size_t i = 0; i < n; ++i) {
      // f1 - f0 = (v1 - v0, a1 - a0); v1 - v0 = h0 * a0.
      const Vec3& v = state.velocities[i];
      const Vec3 dv_part = kvel_[0][i] * h0;               // v1 - v0
      const Vec3 da_part = probe_accel_[i] - kvel_[0][i];  // a1 - a0
      for (int c = 0; c < 3; ++c) {
        const Real sx = atol_ + rtol_ * std::abs((&state.positions[i].x)[c]);
        const Real sv = atol_ + rtol_ * std::abs((&v.x)[c]);
        accum(d2_sq, (&dv_part.x)[c], sx);
        accum(d2_sq, (&da_part.x)[c], sv);
      }
    }
    const Real d2 = std::sqrt(d2_sq / denom) / h0;

    const Real h1 = (d1 <= 1e-15 && d2 <= 1e-15)
                        ? std::max(Real{1e-6}, h0 * Real{1e-3})
                        : std::pow(Real{0.01} / std::max(d1, d2), Real{1.0 / 5.0});
    return std::min({100 * h0, h1, max_dt});
  }

  Real rtol_;
  Real atol_;
  Real h_{0};  ///< Internally controlled step size (0 = uninitialized).
  bool fsal_valid_{false};
  std::vector<Vec3> kpos_[7];  ///< Position-derivative stages (stage velocities).
  std::vector<Vec3> kvel_[7];  ///< Velocity-derivative stages (stage accelerations).
  std::vector<Vec3> probe_accel_;
  State stage_;  ///< Scratch positions for force evaluations (velocities unused).
};

constexpr Real DormandPrince45::kA2[1];
constexpr Real DormandPrince45::kA3[2];
constexpr Real DormandPrince45::kA4[3];
constexpr Real DormandPrince45::kA5[4];
constexpr Real DormandPrince45::kA6[5];
constexpr Real DormandPrince45::kA7[6];
constexpr const Real* DormandPrince45::kA[6];
constexpr Real DormandPrince45::kE[7];

}  // namespace

std::unique_ptr<Integrator> make_integrator(IntegratorKind kind, const IntegratorOptions& options) {
  switch (kind) {
    case IntegratorKind::VelocityVerlet:
      return std::make_unique<VelocityVerlet>();
    case IntegratorKind::RK45:
      return std::make_unique<DormandPrince45>(options.rtol, options.atol);
  }
  throw std::invalid_argument("unknown integrator kind");
}

std::unique_ptr<Integrator> make_integrator(IntegratorKind kind) {
  return make_integrator(kind, IntegratorOptions{});
}

}  // namespace coulomb
