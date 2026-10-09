#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA4_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &cos2, const double &phi, const double &p,
           const double &k, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      // Precompute basic quantities
      const double l1_sq = powr<2>(l1);
      const double p_sq = powr<2>(p);
      const double l1p = l1 * p;
      const double k_sq = powr<2>(k);
      const double k6 = powr<6>(k);
      const double one_plus_k6 = 1. + k6;
      const double inv_1pk6 = 1. / one_plus_k6;
      const double sin1 = sqrt(1. - powr<2>(cos1));
      const double sin2_factor = sqrt(2. - 2. * powr<2>(cos2));
      const double cos_phi = cos(phi);
      const double sin_phi = sin(phi);
      const double k_scale = pow(one_plus_k6, 0.16666666666666666667);

      const double cosl1p1 = sin1 * cos2;
      const double cosl1p2 = (-0.3333333333333333) * (cos2 + (-2.) * sin2_factor * cos_phi) * sin1;
      const double cosl1p3 = (-0.3333333333333333) * (cos2 + (cos_phi + (-1.732050807568877) * sin_phi) * sin2_factor) * sin1;
      const double cosl1p4 = (-0.3333333333333333) * (cos2 + (cos_phi + (1.732050807568877) * sin_phi) * sin2_factor) * sin1;

      // Precompute cosine powers
      const double c1_2 = cosl1p1 * cosl1p1, c1_3 = c1_2 * cosl1p1, c1_4 = c1_2 * c1_2, c1_5 = c1_4 * cosl1p1;
      const double c1_6 = c1_3 * c1_3, c1_7 = c1_6 * cosl1p1, c1_8 = c1_4 * c1_4;
      const double c2_2 = cosl1p2 * cosl1p2, c2_3 = c2_2 * cosl1p2, c2_4 = c2_2 * c2_2, c2_5 = c2_4 * cosl1p2;
      const double c2_6 = c2_3 * c2_3;
      const double c3_2 = cosl1p3 * cosl1p3, c3_3 = c3_2 * cosl1p3, c3_4 = c3_2 * c3_2, c3_5 = c3_4 * cosl1p3;
      const double c3_6 = c3_3 * c3_3;

      // Precompute momentum sums and arguments
      const double sum_c123 = cosl1p1 + cosl1p2 + cosl1p3;
      const double sum_c12 = cosl1p1 + cosl1p2;
      const double q2_123 = l1_sq - 2. * sum_c123 * l1p + p_sq;
      const double q2_12s = l1_sq - 2. * sum_c12 * l1p + 1.333333333333333 * p_sq;

      const auto _repl1 = dtZA(k_scale);
      const auto _repl2 = RB(k_sq, l1_sq);
      const auto _repl3 = RBdot(k_sq, l1_sq);
      const auto _repl4 = ZA(k_scale);
      const auto _repl5 = ZA(1.02 * k_scale);
      const auto _repl6 = ZA(l1);
      const auto _repl7 = RB(k_sq, q2_123);
      const auto _repl8 = ZA(sqrt(q2_123));
      const auto _repl9 = RB(k_sq, q2_12s);
      const auto _repl10 = ZA(sqrt(q2_12s));
      const auto _repl11 = ZA3(0.816496580927726 * sqrt(l1_sq - sum_c123 * l1p + p_sq));
      const auto _repl12 = ZA3(0.4714045207910317 * sqrt(3. * l1_sq - 3. * (sum_c12 * 2. + cosl1p3) * l1p + 5. * p_sq));
      const auto _repl13 = Zc(k);

      // Precompute regulator prefactor (reused 4+ times) and propagator denominators
      const double reg_prefactor = (-50.) * (_repl4 + (-1.) * _repl5) * _repl2 * k6 + _repl1 * one_plus_k6 * _repl2 + _repl3 * one_plus_k6 * _repl4;
      const double prop_gl_l1 = _repl2 * _repl4 + _repl6 * l1_sq;
      const double inv_prop_gl_l1_sq = 1. / (prop_gl_l1 * prop_gl_l1);

      // Precompute higher l1 powers
      const double l1_3 = l1_sq * l1, l1_4 = l1_sq * l1_sq, l1_5 = l1_4 * l1;
      const double l1_6 = l1_3 * l1_3, l1_7 = l1_6 * l1, l1_8 = l1_4 * l1_4;
      const double l1_9 = l1_8 * l1, l1_10 = l1_5 * l1_5;

      // Precompute higher p powers
      const double p_3 = p_sq * p, p_4 = p_sq * p_sq, p_5 = p_4 * p;
      const double p_6 = p_3 * p_3, p_7 = p_6 * p;

      // Precompute single-cosine momentum arguments (used many times in propagators)
      const double q2_c1 = l1_sq - 2. * cosl1p1 * l1p + p_sq;
      const double q2_c2 = l1_sq - 2. * cosl1p2 * l1p + p_sq;
      const double qa_c1_half = l1_sq - cosl1p1 * l1p + p_sq;
      const double qa_c2_half = l1_sq - cosl1p2 * l1p + p_sq;

      // Precompute remaining spline lookups for single-cosine channels
      const auto RB_q2_c1 = RB(k_sq, q2_c1);
      const auto ZA_q2_c1 = ZA(sqrt(q2_c1));
      const auto RB_q2_c2 = RB(k_sq, q2_c2);
      const auto ZA_q2_c2 = ZA(sqrt(q2_c2));
      const auto ZA3_c1_half = ZA3(0.816496580927726 * sqrt(qa_c1_half));
      const auto ZA3_c2_half = ZA3(0.816496580927726 * sqrt(qa_c2_half));

      // ZA3 for diagram 1: sqrt(3*l1^2 - 3*(2c1+c2)*l1p + 5*p^2)
      const auto ZA3_2c1c2 = ZA3(0.4714045207910317 * sqrt(3. * l1_sq - 3. * (2. * cosl1p1 + cosl1p2) * l1p + 5. * p_sq));

      // Precompute dressed gluon propagator at q_c1 and q_c2
      const double inv_prop_gl_c1 = 1. / (_repl4 * RB_q2_c1 + q2_c1 * ZA_q2_c1);
      const double inv_prop_gl_c2 = 1. / (_repl4 * RB_q2_c2 + q2_c2 * ZA_q2_c2);

      // Precompute shared propagator denominators (appear 2-3 times each)
      const double inv_q2_123 = 1. / q2_123;
      const double three_q2_12 = 3. * l1_sq - 6. * sum_c12 * l1p + 4. * p_sq;
      const double inv_three_q2_12 = 1. / three_q2_12;
      const double inv_q2_c1 = 1. / q2_c1;
      const double inv_q2_c2 = 1. / q2_c2;
      const double inv_prop_gl_123 = 1. / (_repl4 * _repl7 + q2_123 * _repl8);
      const double inv_prop_gl_12s_3 = 1. / (3. * _repl4 * _repl9 + three_q2_12 * _repl10);

      // ZA4 lookups for diagrams 2, 3, 4
      const auto ZA4_c12_half = ZA4(0.408248290463863 * sqrt(3. * l1_sq - 3. * sum_c12 * l1p + 5. * p_sq));
      const auto ZA4_c12p = ZA4(0.7071067811865475 * sqrt(l1_sq - (cosl1p1 + 2. * cosl1p2 + cosl1p3) * l1p + 2. * p_sq));

      // Ghost channel precomputations
      const double qc2_c1 = l1_sq + 2. * cosl1p1 * l1p + p_sq;
      const double qc2_123 = l1_sq + 2. * sum_c123 * l1p + p_sq;
      const double qc2_12s = l1_sq + 2. * sum_c12 * l1p + 1.333333333333333 * p_sq;
      const auto RB_qc2_c1 = RB(k_sq, qc2_c1);
      const auto Zc_qc2_c1 = Zc(sqrt(qc2_c1));
      const auto RB_qc2_123 = RB(k_sq, qc2_123);
      const auto Zc_qc2_123 = Zc(sqrt(qc2_123));
      const auto RB_qc2_12s = RB(k_sq, qc2_12s);
      const auto Zc_qc2_12s = Zc(sqrt(qc2_12s));
      const auto Zc_l1 = Zc(l1);
      const auto dtZc_k = dtZc(k);
      const auto Zc_102k = Zc(1.02 * k);

      const double inv_prop_gc_c1 = 1. / (_repl13 * RB_qc2_c1 + qc2_c1 * Zc_qc2_c1);
      const double inv_prop_gc_123 = 1. / (_repl13 * RB_qc2_123 + qc2_123 * Zc_qc2_123);
      const double inv_prop_gc_12s_3 = 1. / (3. * _repl13 * RB_qc2_12s + (3. * l1_sq + 6. * sum_c12 * l1p + 4. * p_sq) * Zc_qc2_12s);
      const double inv_prop_gc_l1_sq = 1. / ((_repl13 * _repl2 + l1_sq * Zc_l1) * (_repl13 * _repl2 + l1_sq * Zc_l1));

      // Ghost regulator prefactor
      const double ghost_reg_prefactor = _repl13 * _repl3 + _repl2 * dtZc_k + (-50.) * (_repl13 - Zc_102k) * _repl2;

      // ZAcbc lookups
      const double qacbc_c1 = l1_sq + cosl1p1 * l1p + p_sq;
      const double qacbc_123 = l1_sq + l1 * sum_c123 * p + p_sq;
      const double qacbc_2c1c2 = l1_sq + l1 * (2. * cosl1p1 + cosl1p2) * p + 1.666666666666667 * p_sq;
      const double qacbc_2c12c3 = l1_sq + l1 * (2. * cosl1p1 + 2. * cosl1p2 + cosl1p3) * p + 1.666666666666667 * p_sq;
      const auto ZAcbc_c1 = ZAcbc(0.816496580927726 * sqrt(qacbc_c1));
      const auto ZAcbc_123 = ZAcbc(0.816496580927726 * sqrt(qacbc_123));
      const auto ZAcbc_2c1c2 = ZAcbc(0.816496580927726 * sqrt(qacbc_2c1c2));
      const auto ZAcbc_2c12c3 = ZAcbc(0.816496580927726 * sqrt(qacbc_2c12c3));

      // Diagram 4: (c2+c3) channel
      const double sum_c23 = cosl1p2 + cosl1p3;
      const double q2_23s = l1_sq - 2. * sum_c23 * l1p + 1.333333333333333 * p_sq;
      const double three_q2_23 = 3. * l1_sq - 6. * sum_c23 * l1p + 4. * p_sq;
      const double inv_three_q2_23 = 1. / three_q2_23;
      const auto RB_q2_23s = RB(k_sq, q2_23s);
      const auto ZA_q2_23s = ZA(sqrt(q2_23s));
      const double inv_prop_gl_23s_3 = 1. / (3. * _repl4 * RB_q2_23s + three_q2_23 * ZA_q2_23s);
      const auto ZA4_c23_half = ZA4(0.408248290463863 * sqrt(3. * l1_sq - 3. * sum_c23 * l1p + 5. * p_sq));

      // Accumulate diagram contributions to reduce peak register pressure
      double result = 0.;

      // Diagram 1: gluon triangle ZA3*ZA3
      result += (0.163265306122449) *
                 (reg_prefactor) *
                 ((_repl11) *
                  ((-243.) *
                       ((3.) * (c1_4) + (-6.) * (c2_4) +
                        (-12.) * ((c2_3) * (cosl1p3)) +
                        (3.) * (cosl1p2 + (3.) * (cosl1p3)) * (c1_3) +
                        (2. + (-3.) * (c3_2)) * (c2_2) +
                        (-1. + (3.) * (c3_2)) * (c3_2) +
                        (cosl1p2) * (2. + (3.) * (c3_2)) * (cosl1p3) +
                        (-1. + (-3.) * (c2_2) + (5.) * ((cosl1p2) * (cosl1p3)) +
                         (6.) * (c3_2)) *
                            (c1_2) +
                        ((2.) * (cosl1p2) + (-12.) * (c2_3) + (-4.) * (cosl1p3) +
                         (-10.) * ((c2_2) * (cosl1p3)) + (5.) * ((cosl1p2) * (c3_2)) +
                         (9.) * (c3_3)) *
                            (cosl1p1)) *
                       (l1_10) +
                   (81.) *
                       ((54.) * (c1_5) + (-72.) * (c2_5) +
                        (-180.) * ((c2_4) * (cosl1p3)) +
                        (90.) * (cosl1p2 + (2.) * (cosl1p3)) * (c1_4) +
                        (6. + (-108.) * (c3_2)) * (c2_3) +
                        (-6.) *
                            (1. + (3.) * (c2_2) + (-36.) * ((cosl1p2) * (cosl1p3)) +
                             (-27.) * (c3_2)) *
                            (c1_3) +
                        (3.) * (5. + (6.) * (c3_2)) * ((c2_2) * (cosl1p3)) +
                        ((41.) * (cosl1p2) + (-252.) * (c2_3) + (-59.) * (cosl1p3) +
                         (-138.) * ((c2_2) * (cosl1p3)) + (192.) * ((cosl1p2) * (c3_2)) +
                         (198.) * (c3_3)) *
                            (c1_2) +
                        (2.) * (-2. + (9.) * (c3_4)) * (cosl1p3) +
                        (4. + (-3.) * (c3_2) + (54.) * (c3_4)) * (cosl1p2) +
                        ((-252.) * (c2_4) + (-408.) * ((c2_3) * (cosl1p3)) +
                         (192.) * ((cosl1p2) * (c3_3)) +
                         (53. + (-54.) * (c3_2)) * (c2_2) +
                         (-47. + (108.) * (c3_2)) * (c3_2)) *
                            (cosl1p1)) *
                       ((l1_9) * (p)) +
                   (27.) *
                       (16. + (-39.) * (c1_2) + (-207.) * (c1_4) +
                        (-333.) * (c1_6) +
                        (-1.) *
                            ((1224.) * (c1_5) + (1350.) * ((c1_4) * (cosl1p3)) +
                             (9.) * (15. + (164.) * (c3_2)) * (c1_3) +
                             (6.) * (-53. + (171.) * (c3_2)) * ((c1_2) * (cosl1p3)) +
                             (-28. + (135.) * (c3_2) + (9.) * (c3_4)) * (cosl1p3) +
                             (3.) * (-18. + (77.) * (c3_2) + (84.) * (c3_4)) * (cosl1p1)) *
                            (cosl1p3) +
                        ((-3.) * (44. + (231.) * (c1_2) + (258.) * (c1_4)) * (cosl1p1) +
                         (-1.) *
                             (-10. + (2220.) * (c1_4) + (2172.) * ((c1_3) * (cosl1p3)) +
                              (189.) * (c3_2) + (126.) * (c3_4) +
                              (3.) * (-23. + (344.) * (c3_2)) * ((cosl1p1) * (cosl1p3)) +
                              (3.) * (89. + (760.) * (c3_2)) * (c1_2)) *
                             (cosl1p3) +
                         (-65. + (-225.) * (c1_4) + (234.) * (c2_4) +
                          (702.) * ((c2_3) * (cosl1p3)) + (153.) * (c3_2) +
                          (-225.) * (c3_4) +
                          (12.) * ((138.) * (cosl1p2) + (11.) * (cosl1p3)) * (c1_3) +
                          (3.) *
                              (-173. + (849.) * (c2_2) + (1208.) * ((cosl1p2) * (cosl1p3)) +
                               (-24.) * (c3_2)) *
                              (c1_2) +
                          (36.) * (17. + c3_2) * ((cosl1p2) * (cosl1p3)) +
                          (9.) * (38. + (67.) * (c3_2)) * (c2_2) +
                          (6.) *
                              ((225.) * (c2_3) + (73.) * (cosl1p3) +
                               (512.) * ((c2_2) * (cosl1p3)) + (-122.) * (c3_3) +
                               (74. + (262.) * (c3_2)) * (cosl1p2)) *
                              (cosl1p1)) *
                             (cosl1p2)) *
                            (cosl1p2)) *
                       ((l1_8) * (p_sq)) +
                   (9.) *
                       ((6.) *
                            (-48. + (253.) * (c1_2) + (399.) * (c1_4) +
                             (126.) * (c1_6)) *
                            (cosl1p1) +
                        (-112. + (3051.) * (c1_6) + (3942.) * ((c1_5) * (cosl1p3)) +
                         (6.) * (c3_2) + (414.) * (c3_4) +
                         (3.) * (467. + (1089.) * (c3_2)) * ((c1_3) * (cosl1p3)) +
                         (5177.999999999999 + (4077.) * (c3_2)) * (c1_4) +
                         (3.) * (-152. + (874.) * (c3_2) + (27.) * (c3_4)) *
                             ((cosl1p1) * (cosl1p3)) +
                         (3.) * (524. + (1049.) * (c3_2) + (342.) * (c3_4)) *
                             (c1_2)) *
                            (cosl1p3) +
                        (-176. + (2982.) * (c1_2) + (6792.) * (c1_4) +
                         (2241.) * (c1_6) +
                         (3.) *
                             ((2436.) * (c1_5) + (2691.) * ((c1_4) * (cosl1p3)) +
                              (9.) * (90. + (181.) * (c3_2)) * ((c1_2) * (cosl1p3)) +
                              (12.) * (237. + (232.) * (c3_2)) * (c1_3) +
                              (-166. + (546.) * (c3_2) + (9.) * (c3_4)) * (cosl1p3) +
                              (6.) * (32. + (273.) * (c3_2) + (54.) * (c3_4)) * (cosl1p1)) *
                             (cosl1p3) +
                         (3.) *
                             ((318. + (1543.) * (c1_2) + (504.) * (c1_4)) * (cosl1p1) +
                              (3.) *
                                  (-106. + (268.) * (c1_4) + (246.) * ((c1_3) * (cosl1p3)) +
                                   (129.) * (c3_2) + (12.) * (c3_4) +
                                   (cosl1p1) * (-291. + (200.) * (c3_2)) * (cosl1p3) +
                                   (-411. + (482.) * (c3_2)) * (c1_2)) *
                                  (cosl1p3) +
                              (-1.) *
                                  (172. + (1224.) * (c1_4) + (36.) * (c2_4) +
                                   (126.) * ((c2_3) * (cosl1p3)) + (1014.) * (c3_2) +
                                   (-36.) * (c3_4) +
                                   (27.) * ((101.) * (cosl1p2) + (128.) * (cosl1p3)) * (c1_3) +
                                   (15.) * (125. + (3.) * (c3_2)) * ((cosl1p2) * (cosl1p3)) +
                                   (18.) * (43. + (8.) * (c3_2)) * (c2_2) +
                                   (2024. + (2052.) * (c2_2) + (4449.) * ((cosl1p2) * (cosl1p3)) +
                                    (2040.) * (c3_2)) *
                                       (c1_2) +
                                   ((594.) * (c2_3) + (1752.) * ((c2_2) * (cosl1p3)) +
                                    (6.) * (759.0000000000001 + (8.) * (c3_2)) * (cosl1p3) +
                                    (7.) * (391. + (213.) * (c3_2)) * (cosl1p2)) *
                                       (cosl1p1)) *
                                  (cosl1p2)) *
                             (cosl1p2)) *
                            (cosl1p2)) *
                       ((l1_7) * (p_3)) +
                   (3.) *
                       (1252. + (-1548.) * (c1_2) + (-16527.) * (c1_4) +
                        (-7776.000000000001) * (c1_6) + (-324.) * (c1_8) +
                        (-3.) *
                            ((486.) * (c1_7) + (756.) * ((c1_6) * (cosl1p3)) +
                             (117.) * (35. + (6.) * (c3_2)) * ((c1_4) * (cosl1p3)) +
                             (6996. + (810.) * (c3_2)) * (c1_5) +
                             (2.) * (98. + (225.) * (c3_2) + (9.) * (c3_4)) * (cosl1p3) +
                             (3.) * (402.9999999999999 + (1323.) * (c3_2) + (18.) * (c3_4)) *
                                 ((c1_2) * (cosl1p3)) +
                             (9617. + (3660.) * (c3_2) + (324.) * (c3_4)) * (c1_3) +
                             (2.) * (552. + (184.) * (c3_2) + (513.) * (c3_4)) * (cosl1p1)) *
                            (cosl1p3) +
                        (3.) *
                            ((-378.) * (c1_7) + (232.) * (cosl1p3) +
                             (-1395.) * ((c1_6) * (cosl1p3)) + (-456.) * (c3_3) +
                             (-414.) * (c3_5) +
                             (-15.) * (1045. + (123.) * (c3_2)) * ((c1_4) * (cosl1p3)) +
                             (-12.) * (713. + (150.) * (c3_2)) * (c1_5) +
                             (-1.) * (10039. + (8565.) * (c3_2) + (378.) * (c3_4)) *
                                 ((c1_2) * (cosl1p3)) +
                             (-1.) *
                                 (12419. + (7281.000000000001) * (c3_2) + (1305.) * (c3_4)) *
                                 (c1_3) +
                             (-1.) *
                                 (-72. + (-1130.) * (c3_2) + (4725.) * (c3_4) +
                                  (27.) * (c3_6)) *
                                 (cosl1p1) +
                             (392. + (-5412.000000000001) * (c1_2) + (-7995.) * (c1_4) +
                              (-378.) * (c1_6) +
                              (-1.) *
                                  ((900.) * (c1_5) + (855.) * ((c1_4) * (cosl1p3)) +
                                   (9.) * (-275. + (113.) * (c3_2)) * (cosl1p3) +
                                   (3.) * (-808. + (231.) * (c3_2)) * ((c1_2) * (cosl1p3)) +
                                   (3.) * (613. + (396.) * (c3_2)) * (c1_3) +
                                   (-5225. + (4167.) * (c3_2) + (108.) * (c3_4)) * (cosl1p1)) *
                                  (cosl1p3) +
                              ((432.) * (c1_5) + (5558.999999999999) * (cosl1p3) +
                               (1620.) * ((c1_4) * (cosl1p3)) + (-180.) * (c3_3) +
                               (36.) * (583. + c3_2) * ((c1_2) * (cosl1p3)) +
                               (72.) * (75. + (19.) * (c3_2)) * (c1_3) +
                               (4765. + (8328.) * (c3_2) + (-108.) * (c3_4)) * (cosl1p1) +
                               (3.) *
                                   (939. + (486.) * (c1_4) + (336.) * (c2_2) +
                                    (1008.) * ((cosl1p2) * (cosl1p3)) + (819.) * (c3_2) +
                                    (3.) * ((162.) * (cosl1p2) + (343.) * (cosl1p3)) * (c1_3) +
                                    (4507. + (216.) * (c2_2) + (618.) * ((cosl1p2) * (cosl1p3)) +
                                     (527.9999999999999) * (c3_2)) *
                                        (c1_2) +
                                    ((36.) * (c2_3) + (126.) * ((c2_2) * (cosl1p3)) +
                                     (15.) * (368. + (3.) * (c3_2)) * (cosl1p3) +
                                     (4.) * (593. + (36.) * (c3_2)) * (cosl1p2)) *
                                        (cosl1p1)) *
                                   (cosl1p2)) *
                                  (cosl1p2)) *
                                 (cosl1p2)) *
                            (cosl1p2)) *
                       ((l1_6) * (p_4)) +
                   (3.) *
                       ((1728.) * (c1_7) + (4734.) * ((c1_6) * (cosl1p3)) +
                        (9.) * (5033. + (86.) * (c3_2)) * ((c1_4) * (cosl1p3)) +
                        (6.) * (3548. + (447.0000000000001) * (c3_2)) * (c1_5) +
                        (2.) * (-604. + (201.) * (c3_2) + (342.) * (c3_4)) * (cosl1p3) +
                        (2.) * (15017. + (579.) * (c3_2) + (540.) * (c3_4)) *
                            ((c1_2) * (cosl1p3)) +
                        (3.) * (5930. + (5097.) * (c3_2) + (708.) * (c3_4)) *
                            (c1_3) +
                        (-5656.000000000001 + (7803.999999999999) * (c3_2) + (3918.) * (c3_4) +
                         (54.) * (c3_6)) *
                            (cosl1p1) +
                        (-4448. + (23336.) * (c1_2) + (61143.00000000001) * (c1_4) +
                         (7362.) * (c1_6) +
                         (3.) *
                             ((4998.) * (c1_5) + (2283.) * ((c1_4) * (cosl1p3)) +
                              (2.) * (-85. + (444.) * (c3_2)) * (cosl1p3) +
                              (3.) * (1067. + (475.) * (c3_2)) * ((c1_2) * (cosl1p3)) +
                              (2.) * (13202. + (770.9999999999999) * (c3_2)) * (c1_3) +
                              (2.) * (2636. + (808.) * (c3_2) + (99.) * (c3_4)) * (cosl1p1)) *
                             (cosl1p3) +
                         ((1106. + (46983.00000000001) * (c1_2) + (10566.) * (c1_4)) *
                              (cosl1p1) +
                          (9.) *
                              (-656. + (1169.) * (c1_4) + (14.) * ((c1_3) * (cosl1p3)) +
                               (52.) * (c3_2) +
                               (4.) * (59. + (111.) * (c3_2)) * (c1_2) +
                               (cosl1p1) * (-2231. + (195.) * (c3_2)) * (cosl1p3)) *
                              (cosl1p3) +
                          (-3.) *
                              ((-702.) * (c1_4) +
                               (6.) * ((464.) * (cosl1p2) + (647.0000000000001) * (cosl1p3)) * (c1_3) +
                               (538. + (738.) * (c2_2) + (1847.) * ((cosl1p2) * (cosl1p3)) +
                                (1241.) * (c3_2)) *
                                   (3.) +
                               (4729. + (2418.) * (c2_2) + (5679.) * ((cosl1p2) * (cosl1p3)) +
                                (2886.) * (c3_2)) *
                                   (c1_2) +
                               ((576.) * (c2_3) + (1728.) * ((c2_2) * (cosl1p3)) +
                                (4.) * (4139. + (-45.) * (c3_2)) * (cosl1p3) +
                                (9091. + (1377.) * (c3_2)) * (cosl1p2)) *
                                   (cosl1p1)) *
                              (cosl1p2)) *
                             (cosl1p2)) *
                            (cosl1p2)) *
                       ((l1_5) * (p_5)) +
                   (-1.) *
                       ((-3580. + (10980.) * (c1_6) + (24561.) * ((c1_5) * (cosl1p3)) +
                         (2064.) * (c3_2) + (774.) * (c3_4) +
                         (9.) * (5561. + (1108.) * (c3_2)) * (c1_4) +
                         ((93246.) * (cosl1p3) + (-4338.) * (c3_3)) * (c1_3) +
                         (-3.) * (904. + (-11640.) * (c3_2) + (117.) * (c3_4)) *
                             (c1_2) +
                         (3.) * (2741. + (466.) * (c3_2) + (126.) * (c3_4)) *
                             ((cosl1p1) * (cosl1p3))) *
                            (2.) +
                        (-3.) *
                            ((-27546.) * (c1_5) + (5933.999999999999) * (c2_3) +
                             (2018.) * (cosl1p3) + (11340.) * ((c2_2) * (cosl1p3)) +
                             (-576.) * (c3_3) +
                             (-3.) * ((11526.) * (cosl1p2) + (14695.) * (cosl1p3)) * (c1_4) +
                             (-23.) *
                                 (3100. + (527.9999999999999) * (c2_2) + (1245.) * ((cosl1p2) * (cosl1p3)) +
                                  (459.) * (c3_2)) *
                                 (c1_3) +
                             (5914. + (5004.) * (c3_2)) * (cosl1p2) +
                             (3.) *
                                 ((1870.) * (c2_3) + (-25052.) * (cosl1p3) +
                                  (2725.) * ((c2_2) * (cosl1p3)) + (662.) * (c3_3) +
                                  (8.) * (-1541. + (163.) * (c3_2)) * (cosl1p2)) *
                                 (c1_2) +
                             (9098. + (3402.) * (c2_4) + (8847.) * ((c2_3) * (cosl1p3)) +
                              (-8372.) * (c3_2) + (-594.) * (c3_4) +
                              (37.) * (200. + (189.) * (c3_2)) * (c2_2) +
                              (2.) * (-1004. + (602.9999999999999) * (c3_2)) * ((cosl1p2) * (cosl1p3))) *
                                 (cosl1p1)) *
                            (cosl1p2)) *
                       ((l1_4) * (p_6)) +
                   (2.) *
                       (((25038.) * (c1_5) +
                         (9.) * ((8431.) * (cosl1p2) + (5479.) * (cosl1p3)) * (c1_4) +
                         (3.) *
                             (8854. + (25497.) * (c2_2) + (33738.) * ((cosl1p2) * (cosl1p3)) +
                              (7784.999999999999) * (c3_2)) *
                             (c1_3) +
                         ((3120.) * (c2_3) + (928.) * (cosl1p3) +
                          (2706.) * ((c2_2) * (cosl1p3)) + (-312.) * (c3_3) +
                          (4357. + (-591.) * (c3_2)) * (cosl1p2)) *
                             (-2.) +
                         ((25929.) * (c2_3) + (47060.) * (cosl1p3) +
                          (53433.00000000001) * ((c2_2) * (cosl1p3)) + (-1350.) * (c3_3) +
                          (32626. + (24408.) * (c3_2)) * (cosl1p2)) *
                             (c1_2) +
                         ((279.) * (c2_4) + (1962.) * ((c2_3) * (cosl1p3)) +
                          (-6.) * (-4658. + (201.) * (c3_2)) * ((cosl1p2) * (cosl1p3)) +
                          (448. + (909.) * (c3_2)) * (c2_2) +
                          (5285. + (-7441.) * (c3_2) + (216.) * (c3_4)) * (-2.)) *
                             (cosl1p1)) *
                            (l1_3) +
                        (-1.) *
                            ((-2417. + (24486.) * (c1_4) + (-5683.999999999999) * (c2_2) +
                              (-1599.) * ((cosl1p2) * (cosl1p3)) + (1088.) * (c3_2) +
                              (42.) * ((1361.) * (cosl1p2) + (971.) * (cosl1p3)) * (c1_3) +
                              (-797.0000000000001 + (41226.) * (c2_2) + (57621.) * ((cosl1p2) * (cosl1p3)) +
                               (16656.) * (c3_2)) *
                                  (c1_2) +
                              ((-7569.) * (cosl1p2) + (8550.) * (c2_3) + (5975.) * (cosl1p3) +
                               (17199.) * ((c2_2) * (cosl1p3)) +
                               (9009.) * ((cosl1p2) * (c3_2)) + (360.) * (c3_3)) *
                                  (cosl1p1)) *
                                 (l1_sq) +
                             (-2.) *
                                 ((5139.) * (c1_3) + (-1504.) * (cosl1p2) + (-209.) * (cosl1p3) +
                                  ((8547.) * (cosl1p2) + (6870.) * (cosl1p3)) * (c1_2) +
                                  (3.) *
                                      (-571. + (1136.) * (c2_2) + (1713.) * ((cosl1p2) * (cosl1p3)) +
                                       (577.) * (c3_2)) *
                                      (cosl1p1)) *
                                 (l1p) +
                             (512.) * (-1. + (3.) * (c1_2) + (3.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
                                 (p_sq)) *
                            (p)) *
                       (p_7)) *
                  ((_repl12) *
                   ((inv_1pk6) *
                    ((inv_prop_gl_l1_sq) *
                     ((inv_q2_c1) *
                      ((inv_q2_123) *
                       ((inv_three_q2_12) *
                        ((inv_prop_gl_123) *
                         ((inv_prop_gl_12s_3) *
                          ((inv_prop_gl_c1) *
                           ((ZA3_c1_half) *
                            (ZA3_2c1c2)))))))))));

      // Diagram 2: gluon box ZA3*ZA4
      result += ((0.02040816326530612) *
                  (reg_prefactor) *
                  ((inv_1pk6) *
                   ((-3.) *
                        (40. + (9.) * (c1_4) + (-18.) * (c2_4) +
                         (-36.) * ((c2_3) * (cosl1p3)) + (-21.) * (c3_2) +
                         (9.) * (c3_4) + (9.) * (cosl1p2 + (3.) * (cosl1p3)) * (c1_3) +
                         (-3.) *
                             (7. + (3.) * (c2_2) + (-5.) * ((cosl1p2) * (cosl1p3)) +
                              (-6.) * (c3_2)) *
                             (c1_2) +
                         (cosl1p2) * (-70. + (9.) * (c3_2)) * (cosl1p3) +
                         (-1.) * (70. + (9.) * (c3_2)) * (c2_2) +
                         (-1.) *
                             ((36.) * (c2_3) + (30.) * ((c2_2) * (cosl1p3)) +
                              (70. + (-15.) * (c3_2)) * (cosl1p2) +
                              (-27.) * (-4. + c3_2) * (cosl1p3)) *
                             (cosl1p1)) *
                        (l1_6) +
                    (3.) *
                        ((9.) * (c1_5) + (-36.) * (c2_5) +
                         (-90.) * ((c2_4) * (cosl1p3)) +
                         (9.) * ((3.) * (cosl1p2) + (4.) * (cosl1p3)) * (c1_4) +
                         (-2.) * (296. + (27.) * (c3_2)) * (c2_3) +
                         (-6. + (9.) * (c2_2) + (78.) * ((cosl1p2) * (cosl1p3)) +
                          (45.) * (c3_2)) *
                             (c1_3) +
                         ((-888.) * (cosl1p3) + (9.) * (c3_3)) * (c2_2) +
                         (104. + (-6.) * (c3_2) + (9.) * (c3_4)) * (cosl1p3) +
                         (208. + (-308.) * (c3_2) + (27.) * (c3_4)) * (cosl1p2) +
                         (-1.) *
                             ((54.) * (c2_3) + (42.) * (cosl1p3) + (9.) * ((c2_2) * (cosl1p3)) +
                              (-45.) * (c3_3) + (308. + (-66.) * (c3_2)) * (cosl1p2)) *
                             (c1_2) +
                         (-1.) *
                             (-104. + (90.) * (c2_4) + (132.) * ((c2_3) * (cosl1p3)) +
                              (42.) * (c3_2) + (-36.) * (c3_4) +
                              (888. + (9.) * (c3_2)) * (c2_2) +
                              ((664.0000000000001) * (cosl1p3) + (-78.) * (c3_3)) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((l1_5) * (p)) +
                    (l1_4) *
                        (-1072. + (-27.) * ((c1_5) * (cosl1p2)) + (54.) * (c2_6) +
                         (6372.000000000001) * ((c2_3) * (cosl1p3)) +
                         (162.) * ((c2_5) * (cosl1p3)) + (402.) * (c3_2) +
                         (-99.) * (c3_4) +
                         (-9.) * (11. + (6.) * (c2_2) + (12.) * ((cosl1p2) * (cosl1p3))) *
                             (c1_4) +
                         (27.) * (118. + (5.) * (c3_2)) * (c2_4) +
                         (1576. + (3579.) * (c3_2) + (-54.) * (c3_4)) * (c2_2) +
                         ((1576.) * (cosl1p3) + (393.) * (c3_3) + (-27.) * (c3_5)) * (cosl1p2) +
                         (-3.) *
                             ((210.) * (cosl1p3) + (51.) * ((c2_2) * (cosl1p3)) +
                              (-131. + (45.) * (c3_2)) * (cosl1p2)) *
                             (c1_3) +
                         (3.) *
                             (134. + (45.) * (c2_4) + (24.) * ((c2_3) * (cosl1p3)) +
                              (-354.) * (c3_2) +
                              (1193. + (-48.) * (c3_2)) * (c2_2) +
                              ((203.) * (cosl1p3) + (-45.) * (c3_3)) * (cosl1p2)) *
                             (c1_2) +
                         ((162.) * (c2_5) + (876.) * (cosl1p3) + (306.) * ((c2_4) * (cosl1p3)) +
                          (-630.) * (c3_3) +
                          (36.) * (177. + (2.) * (c3_2)) * (c2_3) +
                          (-9.) * (-732. + (17.) * (c3_2)) * ((c2_2) * (cosl1p3)) +
                          (1576. + (609.) * (c3_2) + (-108.) * (c3_4)) * (cosl1p2)) *
                             (cosl1p1)) *
                        (p_sq) +
                    (l1_3) *
                        ((99.) * ((c1_4) * (cosl1p2)) + (-1164.) * (c2_5) +
                         (-2910.) * ((c2_4) * (cosl1p3)) +
                         (-3.) * (98. + (95.) * (c2_2) + (-210.) * ((cosl1p2) * (cosl1p3))) *
                             (c1_3) +
                         (6.) * (234. + (-49.) * (c3_2)) * (cosl1p3) +
                         (-57.) * (196. + (5.) * (c3_2)) * ((c2_2) * (cosl1p3)) +
                         (-2.) * (3723.999999999999 + (1065.) * (c3_2)) * (c2_3) +
                         (2808. + (-4311.999999999999) * (c3_2) + (99.) * (c3_4)) * (cosl1p2) +
                         (-1.) *
                             ((2130.) * (c2_3) + (798.) * (cosl1p3) +
                              (153.) * ((c2_2) * (cosl1p3)) +
                              (4311.999999999999 + (-1062.) * (c3_2)) * (cosl1p2)) *
                             (c1_2) +
                         (-1.) *
                             (-1404. + (2910.) * (c2_4) + (3792.) * ((c2_3) * (cosl1p3)) +
                              (798.) * (c3_2) +
                              (3.) * (3723.999999999999 + (51.) * (c3_2)) * (c2_2) +
                              ((8456.) * (cosl1p3) + (-630.) * (c3_3)) * (cosl1p2)) *
                             (cosl1p1)) *
                        (p_3) +
                    (2.) *
                        (-682. + (102.) * ((c1_3) * (cosl1p2)) + (2514.) * (c2_4) +
                         (5028.) * ((c2_3) * (cosl1p3)) + (203.) * (c3_2) +
                         (203. + (2616.) * (c2_2) + (234.) * ((cosl1p2) * (cosl1p3))) * (c1_2) +
                         (2.) * (454. + (51.) * (c3_2)) * ((cosl1p2) * (cosl1p3)) +
                         (4.) * (227. + (654.0000000000001) * (c3_2)) * (c2_2) +
                         ((5028.) * (c2_3) + (382.) * (cosl1p3) +
                          (5160.) * ((c2_2) * (cosl1p3)) +
                          (908. + (234.) * (c3_2)) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((l1_sq) * (p_4)) +
                    (-772.0000000000001) *
                        ((3.) * ((c1_2) * (cosl1p2)) + (6.) * (c2_3) + (-1.) * (cosl1p3) +
                         (9.) * ((c2_2) * (cosl1p3)) +
                         (-1. + (9.) * (c2_2) + (6.) * ((cosl1p2) * (cosl1p3))) * (cosl1p1) +
                         (-2. + (3.) * (c3_2)) * (cosl1p2)) *
                        ((l1) * (p_5)) +
                    (392.) *
                        (-1. + (3.) * ((cosl1p1) * (cosl1p2)) + (3.) * (c2_2) +
                         (3.) * ((cosl1p2) * (cosl1p3))) *
                        (p_6)) *
                   ((inv_prop_gl_l1_sq) *
                    ((inv_q2_c2) *
                     ((inv_q2_123) *
                      ((inv_prop_gl_123) *
                       ((inv_prop_gl_c2) *
                        ((ZA3_c2_half) *
                         (ZA4_c12p)))))));

      // Diagram 3: gluon box ZA4*ZA3
      result += (-0.04081632653061224) *
                  (reg_prefactor) *
                  ((_repl12) *
                   ((27.) *
                        (40. + (9.) * (c1_4) + (9.) * (c2_4) +
                         (9.) * ((c2_3) * (cosl1p3)) + (-70.) * (c3_2) +
                         (-18.) * (c3_4) + (9.) * ((3.) * (cosl1p2) + cosl1p3) * (c1_3) +
                         (3.) *
                             (-7. + (6.) * (c2_2) + (5.) * ((cosl1p2) * (cosl1p3)) +
                              (-3.) * (c3_2)) *
                             (c1_2) +
                         (-3.) * (7. + (3.) * (c3_2)) * (c2_2) +
                         (-2.) * (35. + (18.) * (c3_2)) * ((cosl1p2) * (cosl1p3)) +
                         ((27.) * (c2_3) + (15.) * ((c2_2) * (cosl1p3)) +
                          (-6.) * (18. + (5.) * (c3_2)) * (cosl1p2) +
                          (-2.) * (35. + (18.) * (c3_2)) * (cosl1p3)) *
                             (cosl1p1)) *
                        (l1_6) +
                    (-9.) *
                        ((81.) * (c1_5) + (81.) * (c2_5) +
                         (135.) * ((c2_4) * (cosl1p3)) +
                         (27.) * ((12.) * (cosl1p2) + (5.) * (cosl1p3)) * (c1_4) +
                         (9.) *
                             (-32. + (45.) * (c2_2) + (42.) * ((cosl1p2) * (cosl1p3)) +
                              (-3.) * (c3_2)) *
                             (c1_3) +
                         (-9.) * (32. + (3.) * (c3_2)) * (c2_3) +
                         (-6.) * (115. + (63.) * (c3_2)) * ((c2_2) * (cosl1p3)) +
                         (4.) * (14. + (93.) * (c3_2) + (-27.) * (c3_4)) * (cosl1p3) +
                         (-2.) * (-214. + (6.) * (c3_2) + (189.) * (c3_4)) * (cosl1p2) +
                         (3.) *
                             ((135.) * (c2_3) + (126.) * ((c2_2) * (cosl1p3)) +
                              (-2.) * (115. + (63.) * (c3_2)) * (cosl1p3) +
                              (-1.) * (688. + (87.) * (c3_2)) * (cosl1p2)) *
                             (c1_2) +
                         (427.9999999999999 + (324.) * (c2_4) + (378.) * ((c2_3) * (cosl1p3)) +
                          (-12.) * (c3_2) + (-378.) * (c3_4) +
                          (-12.) * (206. + (69.) * (c3_2)) * ((cosl1p2) * (cosl1p3)) +
                          (-3.) * (688. + (87.) * (c3_2)) * (c2_2)) *
                             (cosl1p1)) *
                        ((l1_5) * (p)) +
                    (9.) *
                        ((54.) * (c1_6) + (54.) * (c2_6) +
                         (135.) * ((c2_5) * (cosl1p3)) +
                         (135.) * ((2.) * (cosl1p2) + cosl1p3) * (c1_5) +
                         (27.) * (-13. + (2.) * (c3_2)) * (c2_4) +
                         (9.) *
                             (-39. + (54.) * (c2_2) + (58.) * ((cosl1p2) * (cosl1p3)) +
                              (6.) * (c3_2)) *
                             (c1_4) +
                         (-9.) * (47. + (30.) * (c3_2)) * ((c2_3) * (cosl1p3)) +
                         (-473. + (2019.) * (c3_2) + (-459.) * (c3_4)) * (c2_2) +
                         ((-2050.) * (cosl1p3) + (3012.) * (c3_3) + (-270.) * (c3_5)) *
                             (cosl1p2) +
                         (508. + (-660.9999999999999) * (c3_2) + (537.) * (c3_4) +
                          (-27.) * (c3_6)) *
                             (2.) +
                         (9.) *
                             ((60.) * (c2_3) + (81.) * ((c2_2) * (cosl1p3)) +
                              (-392. + c3_2) * (cosl1p2) +
                              (-1.) * (47. + (30.) * (c3_2)) * (cosl1p3)) *
                             (c1_3) +
                         (-473. + (486.) * (c2_4) + (729.0000000000001) * ((c2_3) * (cosl1p3)) +
                          (2019.) * (c3_2) + (-459.) * (c3_4) +
                          (-18.) * (355. + (8.) * (c3_2)) * (c2_2) +
                          (-9.) * (547.0000000000001 + (106.) * (c3_2)) * ((cosl1p2) * (cosl1p3))) *
                             (c1_2) +
                         ((270.) * (c2_5) + (-2050.) * (cosl1p3) +
                          (522.0000000000001) * ((c2_4) * (cosl1p3)) + (3012.) * (c3_3) +
                          (-270.) * (c3_5) + (9.) * (-392. + c3_2) * (c2_3) +
                          (-9.) * (547.0000000000001 + (106.) * (c3_2)) * ((c2_2) * (cosl1p3)) +
                          (-2.) * (776. + (-1344.) * (c3_2) + (477.) * (c3_4)) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((l1_4) * (p_sq)) +
                    (9.) *
                        ((72.) * (c1_5) + (72.) * (c2_5) +
                         (3.) * ((458.) * (cosl1p2) + (-87.) * (cosl1p3)) * (c1_4) +
                         (-261.) * ((c2_4) * (cosl1p3)) +
                         (3.) *
                             (887. + (1266.) * (c2_2) + (524.) * ((cosl1p2) * (cosl1p3)) +
                              (-687.) * (c3_2)) *
                             (c1_3) +
                         (-3.) * (-887. + (687.) * (c3_2)) * (c2_3) +
                         ((6402.000000000001) * (cosl1p3) + (-2934.) * (c3_3)) * (c2_2) +
                         (-4.) * (160. + (134.) * (c3_2) + (9.) * (c3_4)) * (cosl1p3) +
                         (-2.) * (1400. + (-1397.) * (c3_2) + (621.) * (c3_4)) * (cosl1p2) +
                         ((3798.) * (c2_3) + (6402.000000000001) * (cosl1p3) +
                          (3702.) * ((c2_2) * (cosl1p3)) + (-2934.) * (c3_3) +
                          (9865. + (-4173.) * (c3_2)) * (cosl1p2)) *
                             (c1_2) +
                         (-2800. + (1374.) * (c2_4) + (1572.) * ((c2_3) * (cosl1p3)) +
                          (2794.) * (c3_2) + (-1242.) * (c3_4) +
                          (9865. + (-4173.) * (c3_2)) * (c2_2) +
                          (-28.) * (-511. + (195.) * (c3_2)) * ((cosl1p2) * (cosl1p3))) *
                             (cosl1p1)) *
                        ((l1_3) * (p_3)) +
                    (-3.) *
                        (-4572. + (5723.999999999999) * (c1_4) + (5723.999999999999) * (c2_4) +
                         (11559.) * ((c2_3) * (cosl1p3)) + (3412.) * (c3_2) +
                         (-1578.) * (c3_4) +
                         (3.) * ((8760.) * (cosl1p2) + (3853.) * (cosl1p3)) * (c1_3) +
                         (-488. + (2865.) * (c3_2)) * (c2_2) +
                         (-488. + (41112.) * (c2_2) + (40491.) * ((cosl1p2) * (cosl1p3)) +
                          (2865.) * (c3_2)) *
                             (c1_2) +
                         ((7532.000000000001) * (cosl1p3) + (-4548.) * (c3_3)) * (cosl1p2) +
                         ((26280.) * (c2_3) + (7532.000000000001) * (cosl1p3) +
                          (40491.) * ((c2_2) * (cosl1p3)) + (-4548.) * (c3_3) +
                          (44. + (8160.) * (c3_2)) * (cosl1p2)) *
                             (cosl1p1)) *
                        ((l1_sq) * (p_4)) +
                    (2.) *
                        ((10869.) * (c1_3) + (10869.) * (c2_3) +
                         (18015.) * ((c2_2) * (cosl1p3)) +
                         (3.) * ((11319.) * (cosl1p2) + (6005.) * (cosl1p3)) * (c1_2) +
                         (-2.) * (777.9999999999999 + (147.) * (c3_2)) * (cosl1p3) +
                         (-9026. + (6852.000000000001) * (c3_2)) * (cosl1p2) +
                         (-9026. + (33957.) * (c2_2) + (37380.) * ((cosl1p2) * (cosl1p3)) +
                          (6852.000000000001) * (c3_2)) *
                             (cosl1p1)) *
                        ((l1) * (p_5)) +
                    (-8.) *
                        (-584. + (843.) * (c1_2) + (1686.) * ((cosl1p1) * (cosl1p2)) +
                         (843.) * (c2_2) + (909.) * ((cosl1p1) * (cosl1p3)) +
                         (909.) * ((cosl1p2) * (cosl1p3)) + (66.) * (c3_2)) *
                        (p_6)) *
                   ((inv_1pk6) *
                    ((inv_prop_gl_l1_sq) *
                     ((inv_q2_123) *
                      ((inv_three_q2_12) *
                       ((inv_prop_gl_123) *
                        ((inv_prop_gl_12s_3) *
                         (ZA4_c12_half))))))))) *
                 (_repl11);

      // Diagram 4: ZA4^2 term
      result += (0.01530612244897959) *
                 (reg_prefactor) *
                 ((inv_1pk6) *
                  ((-3.) *
                       (-1304. + (54.) * (c1_4) + (-27.) * (c2_4) +
                        (-81.) * ((c2_3) * (cosl1p3)) + (-399.) * (c3_2) +
                        (-27.) * (c3_4) + (108.) * (cosl1p2 + cosl1p3) * (c1_3) +
                        (3.) *
                            (-262. + (9.) * (c2_2) + (30.) * ((cosl1p2) * (cosl1p3)) +
                             (9.) * (c3_2)) *
                            (c1_2) +
                        (-3.) * (133. + (18.) * (c3_2)) * (c2_2) +
                        (-3.) * (4. + (27.) * (c3_2)) * ((cosl1p2) * (cosl1p3)) +
                        (-3.) *
                            ((262.) * (cosl1p2) + (9.) * (c2_3) + (262.) * (cosl1p3) +
                             (15.) * ((c2_2) * (cosl1p3)) + (15.) * ((cosl1p2) * (c3_2)) +
                             (9.) * (c3_3)) *
                            (cosl1p1)) *
                       (l1_sq) +
                   (-9.) * (cosl1p2 + cosl1p3) *
                       ((l1) *
                        (872. + (282.) * (c1_2) + (151.) * (c2_2) +
                         (20.) * ((cosl1p2) * (cosl1p3)) + (151.) * (c3_2) +
                         (282.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
                        (p)) +
                   (16.) *
                       (382. + (57.) * (c1_2) + (-9.) * (c2_2) +
                        (-75.) * ((cosl1p2) * (cosl1p3)) + (-9.) * (c3_2) +
                        (57.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
                       (p_sq)) *
                  ((inv_prop_gl_l1_sq) *
                   ((inv_three_q2_23) *
                    ((inv_prop_gl_23s_3) *
                     (ZA4_c23_half * ZA4_c23_half))));

      // Diagram 5: ghost box
      result += (0.06122448979591837) *
                 ((9.) *
                      ((3.) * (c1_4) + (-6.) * (c2_4) +
                       (-12.) * ((c2_3) * (cosl1p3)) +
                       (3.) * (cosl1p2 + (3.) * (cosl1p3)) * (c1_3) +
                       (2. + (-3.) * (c3_2)) * (c2_2) +
                       (-1. + (3.) * (c3_2)) * (c3_2) +
                       (cosl1p2) * (2. + (3.) * (c3_2)) * (cosl1p3) +
                       (-1. + (-3.) * (c2_2) + (5.) * ((cosl1p2) * (cosl1p3)) + (6.) * (c3_2)) *
                           (c1_2) +
                       ((2.) * (cosl1p2) + (-12.) * (c2_3) + (-4.) * (cosl1p3) +
                        (-10.) * ((c2_2) * (cosl1p3)) + (5.) * ((cosl1p2) * (c3_2)) +
                        (9.) * (c3_3)) *
                           (cosl1p1)) *
                      (l1_sq) +
                  (3.) *
                      ((12.) * (c1_3) + (-18.) * (c2_3) + (-4.) * (cosl1p3) +
                       (-21.) * ((c2_2) * (cosl1p3)) + (6.) * (c3_3) +
                       ((17.) * (cosl1p2) + (19.) * (cosl1p3)) * (c1_2) +
                       (4. + (-3.) * (c3_2)) * (cosl1p2) +
                       (-1.) * ((7.) * (c2_2) + (5.) * (c3_2)) * (cosl1p1)) *
                      (l1p) +
                  (-16. + (43.) * (c1_2) + (52.) * ((cosl1p1) * (cosl1p2)) + (9.) * (c2_2) +
                   (34.) * ((cosl1p1) * (cosl1p3)) + (6.) * ((cosl1p2) * (cosl1p3))) *
                      (p_sq)) *
                 ((l1_sq) *
                  (ghost_reg_prefactor) *
                  ((ZAcbc_c1) *
                   ((ZAcbc_123) *
                    ((ZAcbc_2c1c2) *
                     ((ZAcbc_2c12c3) *
                      ((inv_prop_gc_l1_sq) *
                       ((inv_prop_gc_c1) *
                        ((inv_prop_gc_123) *
                         (inv_prop_gc_12s_3))))))));

      return result;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto
    constant(const double &p, const double &k,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_FORCEINLINE_FUNCTION auto RB(const double &k2, const double &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RF(const double &k2, const double &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const double &k2, const double &p2)
    {
      return Regulator::RBdot(k2, p2);
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const double &k2, const double &p2)
    {
      return Regulator::RFdot(k2, p2);
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const double &k2, const double &p2)
    {
      return Regulator::dq2RB(k2, p2);
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const double &k2, const double &p2)
    {
      return Regulator::dq2RF(k2, p2);
    }
  };
} // namespace DiFfRG
using DiFfRG::ZA4_kernel;