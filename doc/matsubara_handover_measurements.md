# Matsubara handover: what was measured, and what shipped

Everything below is measured in-tree. Reproduce with

```
tests/common/quadrature/matsubara_convergence          "[.study]"    # Parts 1-13
tests/physics/integration/finiteT/matsubara_cost       "[.study]"    # cost table
ctest -LE slow -j1 -R 'atsubara|quadrature|finite temperature|p0 sums|finite T|Split'
```

---

## 1. The bug: the handover puts a step into the RHS

`MatsubaraQuadrature::predict_size` handed over to the vacuum rule at `typical_E/T = 30`, justified
by "the thermal correction is `4exp(-E/T)` = 1.9e-13 there". That law is the aliasing law of a
**pole**, and a 4D-regulated summand has none: `q^2 + R_B >= k^2` exactly. What it has instead is a
near-rectangular profile cut at `R = sqrt(x_extent) k`, whose aliasing is edge ringing.

Measured sum-minus-integral gap on the assembled 3+1D integrand (Part 13, `PolynomialExp<8>`):

| k/T | 25 | **30** | 40 | 60 | 100 | 140 |
|---|---|---|---|---|---|---|
| m² = 0 | 8.2e-4 | **1.3e-4** | 3.6e-5 | 3.5e-7 | 3.9e-11 | 6.3e-13 |
| m² = −0.9k² | 2.6e-4 | **3.1e-4** | 3.6e-5 | 4.5e-7 | 1.4e-10 | 9.7e-14 |

Non-monotone, as ringing is — and nine orders above the claim. Confirmed end-to-end three ways:

* `matsubara_continuity` on HEAD: **3.09e-4** at k/T = 30.05, a clean step (0 below, 3.09e-4 above).
* Real `QCD_Nf2/avatars_3Dquark` kernels driven statically: the shipped and fixed libraries agree
  bit-for-bit below k/T = 30 and diverge **exactly at k/T = 30.1** by 3.4e-5 (ZA), 3.8e-4 (ZA4),
  **5.8e-4 (ZA3)**. `ZAcbc` and `Zc` — pure `matsubara_finite_extent` — are identical throughout,
  which is the control that the harness is deterministic.

## 2. The decay law (what every extrapolation rests on)

| regulator | decades per mode_below | slope in log10(modes) |
|---|---|---|
| PolynomialExp<2> | −0.57 | −8.5 |
| PolynomialExp<8> | −0.45 | −7.2 |
| PolynomialExp<16> | −0.25 | −5.0 |
| Exponential<b=2> | −0.43 | −7.2 |
| **Litim** | **−0.064** | **−2.07** |

Smooth regulators decay **exponentially in `modes_below`**. Litim is **algebraic, ~modes⁻²** — still
1.2e-5 at 47 modes. No switch to an integral is safe for a hard cutoff at any affordable node count.

## 3. REFUTED: the plan's compact reach (`E_max = sqrt(x_extent)·k`, multiplier 1)

Monien rule sized by reach `C` (`E_max = C·k`) against the exact sum, `PolynomialExp<8>`,
m² = −0.9k². Cell = nodes : relative error.

| k/T | C=1 (compact) | C=20 | C=100 | C=400 |
|---|---|---|---|---|
| 30 | 8 : 4.1e-05 | 20 : 1.8e-16 | 40 : 1.8e-16 | 74 : 1.8e-16 |
| 100 | 12 : 3.7e-03 | 34 : 2.1e-15 | 68 : 4.4e-15 | 128 : 1.8e-15 |
| 300 | 18 : 2.5e-02 | 54 : 1.5e-08 | 116 : 4.4e-15 | 128 : 1.4e-15 |

**Wrong by 4–7 orders, and worse with k/T.** The sizing law encodes the *span*; it does not put
nodes where the structure is. At T = k/300 the summand carries 58 significant Matsubara modes and
an 18-node rule cannot reproduce them. `monien_reach` buys node **density**, so it applies to a
compact summand exactly as to an algebraic tail — it is not "partly compensation for a wrong scale".

Consequences: one reach policy for both paths; §4.1's provider cache re-key is then unnecessary for
correctness (with one policy `E_max` is a bijection of `typical_E`, so the shipped `(T, typical_E)`
key cannot collide). The compact path gets its accuracy from the **exact sum** instead, which is the
cheaper rule up to k/T ≈ 650 and is exact.

## 4. `monien_reach`: 400 → 200

Real `avatars_3Dquark` kernels, worst relative error against a converged reference, over k/T < 46
(above that the reference itself hands over):

| flow | reach 100 | **reach 200** | reach 400 | HEAD (400 + old switch) |
|---|---|---|---|---|
| ZA | 3.9e-15 | **3.5e-15** | 5.0e-15 | 3.4e-05 |
| ZA3 | 2.6e-15 | **1.2e-15** | 2.1e-15 | 5.8e-04 |
| ZA4 | 2.1e-15 | **1.1e-15** | 1.8e-15 | 3.8e-04 |
| ZAcbc, Zc | 0 | **0** | 0 | 0 |
| ZAqbq1 | 7.8e-05 | **1.1e-05** | 4.5e-06 | 2.6e-05 |
| Zq | 3.6e-06 | **4.6e-07** | 1.3e-07 | 7.2e-06 |
| max p0 nodes | 59 | **77** | 103 | 82 |

200 is the only column that is both **cheaper than HEAD and more accurate than HEAD on every flow**.

The two quark flows are the only ones not at machine precision: the 3D quark regulator leaves a
genuine algebraic tail in p0 with a mass scale above k that the model does not report through
`set_typical_E`. That knob now survives `set_k`, so a model can fix this itself.

Part 8b (the plan's Step 2 re-measurement) confirms the calibration basis: with the heaviest scale
reported honestly, every reach from 20 to 400 is indistinguishable on the stressor 400 was fitted to.

## 5. What still limits the handover — and it is not the reach

The residual step at the handover is **2.6e-8 for every reach**. It is the 64-node tangent map's own
quadrature error on a compact summand, not aliasing: Part 5 measures the assembled 3+1D integral
through that rule at 2.588e-08 for N = 64, which is the number `matsubara_continuity` reports to
three more digits. Raise `/integration/vacuum_quad_size` to improve it — 96 nodes gives 4.4e-10.

## 5b. How smooth is the handover on a real QCD flow?

Static k-sweep over `avatars_3Dquark`'s seven flows, k/T = 15 → 400 at ratio 1.005, dressings held
at the model's own (smooth, analytic) initial condition. The shipped rule is compared against a
reference provider that **never hands over** (reach 800, ceiling 1024) at identical k, so the
dressings' spline structure cancels exactly — without that the raw value carries a 4e-4 knot noise
floor that hides everything. Residuals are normalised by field scale, not pointwise: several of
these flows cross zero inside the sweep.

| flow | rule changes | worst step | at k/T | background |
|---|---|---|---|---|
| ZA | 104 | 6.3e-06 | 369 | 1.1e-16 |
| **ZA3** | 104 | **2.2e-06** | **190** | 2.2e-16 |
| ZA4 | 104 | 3.4e-07 | 363 | 1.9e-16 |
| ZAcbc | 60 | 4.6e-07 | 363 | 0 |
| ZAqbq1 | 104 | 1.1e-05 | 365 | 1.7e-09 |
| Zc | 60 | 2.1e-06 | 318 | 0 |
| Zq | 104 | 1.1e-05 | 349 | 4.6e-10 |

Three things follow.

1. **Every rule change below the handover is invisible.** 104 re-selections per flow (the Monien
   rule growing two nodes at a time); the median step between them is 1e-16. The design's central
   claim — that a flow no longer feels the frequency rule being re-chosen — holds.
2. **The handover itself is 2.2e-6 (ZA3, at k/T = 190), worst 1.1e-5**, against **5.8e-4** for the
   pre-2026-08 handover at k/T = 30. That is 50–250×, not the four orders the compact-kernel gate
   suggests: these are *split* kernels whose tail half is 3D-quark-regulated, and what the integral
   discards there is thermal content (`~4exp(-E_min/T)`), not aliasing.
3. **It is bounded by the budget, and demonstrably so.** Raising `max_matsubara_size` from 128 to
   192 pushes the handover past the swept band and every flow drops to **3e-14 … 1e-12**:

   | | ZA | ZA3 | ZA4 | ZAcbc | ZAqbq1 | Zc | Zq |
   |---|---|---|---|---|---|---|---|
   | ceiling 128 | 6.3e-6 | 2.2e-6 | 3.4e-7 | 4.6e-7 | 1.1e-5 | 2.1e-6 | 1.1e-5 |
   | ceiling 192 | 1.1e-12 | 3.3e-14 | 2.9e-14 | 0 | 1.2e-6 | 0 | 2.4e-7 |

   `vacuum_quad_size` is a *different* lever and only helps where the error is quadrature rather
   than thermal: at 128 it makes the two pure-compact flows (`ZAcbc`, `Zc`) exactly smooth — it
   keeps their exact sum the cheaper rule — and improves `ZA` 6.3e-6 → 8.9e-8, but leaves `ZA3`'s
   2.17e-6 unchanged to three digits, because that step is discarded thermal content.

`matsubara_continuity` now covers both structures: the compact case (step 2.6e-8) and an algebraic
tail with a light scale (1.9e-7 at the shipped ceiling, 9.5e-15 with it raised).

**Not done: a time-integrated flow.** It would be a poor discriminator — a whole-flow diff cannot
judge a quadrature change here (m2A is fine-tuned and amplifies everything), and `avatars_3Dquark`
aborts at t = 1.9 with its current UV tuning. The static sweep is also the *conservative*
measurement: feedback from evolving dressings can only smooth the trajectory, and a step in the
RHS is a step regardless of how the dressings got there.

## 6. x_extent truncation is not the dominant error

The plan warns that `x_extent_tolerance` (1e-3/1e-4 in the shipped examples) carries a systematic
1e-3…1e-5 truncation that dominates. Measured, `PolynomialExp<8>`:

| x_extent_tolerance | 1e-2 | 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|---|
| x_extent | 1.3225 | 1.5209 | 1.5209 | 1.5209 |
| truncation of the p0 integral at R | 2.1e-6 | 3.5e-10 | 3.5e-10 | 3.5e-10 |

The dial is a convergence test on a doubled interval, not the residual itself. At its shipped
settings the actual truncation is 3.5e-10, well below the aliasing error the switch is about.
(`Exponential` and `PolynomialExp<2>`, with softer edges, do track the dial: 1e-5 … 2.3e-6.)

## 7. Cost

Frequency-axis nodes and wall time per `get()`, `Integrator_fT_p2_1ang<4>`, x_order 96, cos 16,
T = 0.05, GPU (`matsubara_cost`). Untraited kernel — the Monien/vacuum path:

| k/T | HEAD nodes | HEAD µs | new nodes | new µs | ratio |
|---|---|---|---|---|---|
| 5 | 35 | 126 | 27 | 86 | **0.68×** |
| 20 | 63 | 183 | 47 | 135 | **0.74×** |
| 30 | 75 | 190 | 55 | 146 | **0.77×** |
| 50 | 64 | 117 | 69 | 150 | 1.28× |
| 93 | 64 | 117 | 93 | 175 | 1.50× |
| 140 | 64 | 117 | 113 | 208 | **1.78×** |
| ≥200 | 64 | 146 | 64 | 118 | 1.00× |

Cheaper below k/T ≈ 35, up to 1.78× in 40 < k/T < 190, unchanged above. On the real kernels the
peak is milder: 77 nodes against HEAD's 82, i.e. **cheaper than HEAD at every k/T < 46**.

A kernel that declares `matsubara_finite_extent` is **unchanged**: the exact sum was already the
cheaper rule and still is (2…41 nodes across the same sweep).

## 8. Is this accuracy necessary?

Yes for the fix, no for going further.

* The defect is a **step**, not a bias. 3–6e-4 discontinuous in k, at fixed T, in the RHS of a stiff
  ODE integrated by IDA/CVODE at rel-tol 1e-6…1e-10. A step three to six orders above the solver's
  tolerance is not a small error, it is a discontinuity: the controller rejects steps and reduces
  order at the crossing, and every finite-T flow crosses it.
* After the fix the residual is **2.6e-8** and, more importantly, it is *smooth* — a rule change no
  longer registers as an event. That is already below the tolerances the steppers run at.
* Pushing below 2.6e-8 means raising `vacuum_quad_size`, which costs nodes on the whole vacuum
  branch — most of a large-Λ flow. Not worth it: the error is smooth and below tolerance.
* The one place more accuracy is cheap: a model with a heavy scale should call `set_typical_E`.
  On `ZAqbq1` that is the difference between 1.1e-5 and machine precision, at no cost.
