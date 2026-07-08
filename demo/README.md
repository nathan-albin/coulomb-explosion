# Demo

A single, laptop-sized command that exercises the whole engine end to end and
produces a figure. Distinct from `examples/` (production-scale, multi-GB, needs
a scratch mount) — this is the "point at it and explain it" demo.

```sh
demo/run_demo.sh          # 1,000,000 CH4 explosions; sim itself is ~3 s,
                          # building the engine and the plotting venv add more
demo/run_demo.sh 100000   # smaller/faster
```

It builds the batched f32 SIMD generator, runs `N` independent CH4 (methane)
Coulomb explosions from synthetically sampled initial geometries, and plots the
physics to [ch4_demo.png](ch4_demo.png):

![CH4 demo figure](ch4_demo.png)

- **Left — kinetic-energy release (KER):** total final kinetic energy per
  explosion, the headline observable of a real Coulomb-explosion measurement.
- **Right — where the momentum goes:** the four light H atoms carry ~98% of the
  released energy; the heavy C carries ~2% — even though, by momentum
  conservation, the fragments' momentum magnitudes are comparable.

The subtitle reports the generation throughput recorded in the run's sidecar
(`build/demo-scratch/ch4.bin.meta.json`).

## On the initial geometries

Each simulation starts from a geometry drawn by `UniformSphereSampler`: atom
positions uniform inside a fixed sphere (radius 4 Bohr) with a minimum-separation
cutoff (0.25 Bohr), all atoms at rest. This is a **deliberately synthetic**
sampling of configuration space — chosen to give a downstream network broad
coverage of the geometry→momentum map — **not** a physically motivated
distribution (equilibrium, thermal/Boltzmann, or Wigner). So the KER *shape* and
its absolute scale (~81 eV mean) are properties of this sampler, not of real
methane; substituting real CH4 equilibrium geometry would give ~102 eV. What is
physical and sampler-independent is the structure the checks below confirm:
energy is conserved (KER equals the initial Coulomb potential energy), momentum
stays zero, and the light H atoms carry ~98% of the energy purely from the mass
ratio.

## Checking the physics

`demo/sanity_check.py` runs those internal-consistency checks (energy
conservation, momentum conservation, per-species energy partition vs. the
masses-only prediction, and the absolute-scale anchor):

```sh
python demo/sanity_check.py --bin build/demo-scratch/ch4.bin
```

Everything lands in `build/demo-scratch/` (raw dataset, gitignored) and
`demo/ch4_demo.png`. To check the fast f32 numbers against the fp64
`scipy.solve_ivp` oracle:

```sh
python examples/verify_subset.py --bin build/demo-scratch/ch4.bin --n 500
```

This needs its own environment (`examples/requirements.txt`: scipy, numba,
pyarrow) — `demo/.venv`, set up by `run_demo.sh` for plotting, only has numpy
and matplotlib and won't satisfy it.

The generator, record layout, and Parquet conversion are documented in
[examples/README.md](../examples/README.md) and
[docs/benchmarks/0008-ch4-dataset-gen.md](../docs/benchmarks/0008-ch4-dataset-gen.md).
