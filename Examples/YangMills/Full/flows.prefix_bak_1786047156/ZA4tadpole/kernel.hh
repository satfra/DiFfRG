#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA4tadpole_kernel
  {
  public:
    using Regulator = _Regulator; // clang-format off


static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const double &cos1, const double &cos2, const double &S0, const double &S1,
                                               const double &SPhi, const double &k,
                                               const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
                                               const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
                                               const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
  using namespace DiFfRG;
                                  // clang-format on

      using namespace DiFfRG::compute;
      const double p1 = sqrt(powr<2>(S0) * (1. - S1 * sin(SPhi)));
      const double p2 = 0.866025403784439 * sqrt(powr<2>(S0) * (1. + S1 * sin(SPhi)));
      const double cosp1p2 = powr<2>(S0) * S1 * cos(SPhi) * sqrt(powr<-1>(powr<4>(S0) * (1. - powr<2>(S1) * powr<2>(sin(SPhi)))));
      const double cosl1p1 = cos1;
      const double cosl1p2 = 1.414213562373095 * cos2 *
          sqrt((-1. + powr<2>(cos1)) * (-1. + powr<2>(S1)) * powr<-1>(2. - powr<2>(S1) + powr<2>(S1) * cos(2. * SPhi))) +
        cos1 * powr<2>(S0) * S1 * cos(SPhi) * sqrt(powr<-1>(powr<4>(S0) * (1. - powr<2>(S1) * powr<2>(sin(SPhi)))));
      // clang-format off
using _T = decltype(48. * powr<-1>(11. + 6. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
        (-8. * powr<8>(l1) * powr<2>(p1) - 12. * powr<6>(l1) * powr<4>(p1) -
         4. * powr<4>(l1) * powr<6>(p1) - 3. * cosp1p2 * powr<8>(l1) * p1 * p2 +
         3. * powr<3>(cosp1p2) * powr<8>(l1) * p1 * p2 -
         17. * cosp1p2 * powr<6>(l1) * powr<3>(p1) * p2 +
         5. * powr<3>(cosp1p2) * powr<6>(l1) * powr<3>(p1) * p2 -
         10. * cosp1p2 * powr<4>(l1) * powr<5>(p1) * p2 +
         2. * powr<3>(cosp1p2) * powr<4>(l1) * powr<5>(p1) * p2 - 8. * powr<8>(l1) * powr<2>(p2) -
         34. * powr<6>(l1) * powr<2>(p1) * powr<2>(p2) -
         6. * powr<2>(cosp1p2) * powr<6>(l1) * powr<2>(p1) * powr<2>(p2) +
         12. * powr<4>(cosp1p2) * powr<6>(l1) * powr<2>(p1) * powr<2>(p2) -
         30. * powr<4>(l1) * powr<4>(p1) * powr<2>(p2) +
         powr<2>(cosp1p2) * powr<4>(l1) * powr<4>(p1) * powr<2>(p2) +
         13. * powr<4>(cosp1p2) * powr<4>(l1) * powr<4>(p1) * powr<2>(p2) -
         4. * powr<2>(l1) * powr<6>(p1) * powr<2>(p2) -
         4. * powr<7>(cosl1p2) * powr<7>(l1) * powr<3>(p2) -
         17. * cosp1p2 * powr<6>(l1) * p1 * powr<3>(p2) +
         5. * powr<3>(cosp1p2) * powr<6>(l1) * p1 * powr<3>(p2) -
         21. * cosp1p2 * powr<4>(l1) * powr<3>(p1) * powr<3>(p2) -
         8. * powr<3>(cosp1p2) * powr<4>(l1) * powr<3>(p1) * powr<3>(p2) +
         13. * powr<5>(cosp1p2) * powr<4>(l1) * powr<3>(p1) * powr<3>(p2) +
         6. * cosp1p2 * powr<2>(l1) * powr<5>(p1) * powr<3>(p2) -
         22. * powr<3>(cosp1p2) * powr<2>(l1) * powr<5>(p1) * powr<3>(p2) +
         8. * powr<5>(cosp1p2) * powr<2>(l1) * powr<5>(p1) * powr<3>(p2) +
         6. * cosp1p2 * powr<7>(p1) * powr<3>(p2) -
         8. * powr<3>(cosp1p2) * powr<7>(p1) * powr<3>(p2) +
         2. * powr<5>(cosp1p2) * powr<7>(p1) * powr<3>(p2) - 12. * powr<6>(l1) * powr<4>(p2) -
         30. * powr<4>(l1) * powr<2>(p1) * powr<4>(p2) +
         powr<2>(cosp1p2) * powr<4>(l1) * powr<2>(p1) * powr<4>(p2) +
         13. * powr<4>(cosp1p2) * powr<4>(l1) * powr<2>(p1) * powr<4>(p2) -
         18. * powr<2>(l1) * powr<4>(p1) * powr<4>(p2) +
         30. * powr<2>(cosp1p2) * powr<2>(l1) * powr<4>(p1) * powr<4>(p2) -
         20. * powr<4>(cosp1p2) * powr<2>(l1) * powr<4>(p1) * powr<4>(p2) +
         6. * powr<6>(cosp1p2) * powr<2>(l1) * powr<4>(p1) * powr<4>(p2) +
         12. * powr<2>(cosp1p2) * powr<6>(p1) * powr<4>(p2) -
         16. * powr<4>(cosp1p2) * powr<6>(p1) * powr<4>(p2) +
         4. * powr<6>(cosp1p2) * powr<6>(p1) * powr<4>(p2) -
         10. * cosp1p2 * powr<4>(l1) * p1 * powr<5>(p2) +
         2. * powr<3>(cosp1p2) * powr<4>(l1) * p1 * powr<5>(p2) +
         6. * cosp1p2 * powr<2>(l1) * powr<3>(p1) * powr<5>(p2) -
         22. * powr<3>(cosp1p2) * powr<2>(l1) * powr<3>(p1) * powr<5>(p2) +
         8. * powr<5>(cosp1p2) * powr<2>(l1) * powr<3>(p1) * powr<5>(p2) +
         10. * cosp1p2 * powr<5>(p1) * powr<5>(p2) -
         16. * powr<3>(cosp1p2) * powr<5>(p1) * powr<5>(p2) +
         4. * powr<5>(cosp1p2) * powr<5>(p1) * powr<5>(p2) +
         2. * powr<7>(cosp1p2) * powr<5>(p1) * powr<5>(p2) - 4. * powr<4>(l1) * powr<6>(p2) -
         4. * powr<2>(l1) * powr<2>(p1) * powr<6>(p2) +
         12. * powr<2>(cosp1p2) * powr<4>(p1) * powr<6>(p2) -
         16. * powr<4>(cosp1p2) * powr<4>(p1) * powr<6>(p2) +
         4. * powr<6>(cosp1p2) * powr<4>(p1) * powr<6>(p2) +
         6. * cosp1p2 * powr<3>(p1) * powr<7>(p2) -
         8. * powr<3>(cosp1p2) * powr<3>(p1) * powr<7>(p2) +
         2. * powr<5>(cosp1p2) * powr<3>(p1) * powr<7>(p2) +
         powr<7>(cosl1p1) * powr<6>(l1) * powr<3>(p1) * (-4. * l1 + 4. * cosl1p2 * p2) +
         powr<6>(cosl1p2) * powr<6>(l1) * powr<2>(p2) *
             (26. * powr<2>(l1) + 20. * powr<2>(p1) + 10. * cosp1p2 * p1 * p2 +
              (14. - 4. * powr<2>(cosp1p2)) * powr<2>(p2)) +
         powr<6>(cosl1p1) * powr<5>(l1) * powr<2>(p1) *
             (26. * powr<3>(l1) + cosl1p2 * powr<2>(l1) * (8. * cosp1p2 * p1 - 52. * p2) +
              cosl1p2 * p2 *
                  ((-14. + 4. * powr<2>(cosp1p2)) * powr<2>(p1) - 10. * cosp1p2 * p1 * p2 -
                   4. * powr<2>(p2)) +
              l1 * ((14. - 4. * powr<2>(cosp1p2)) * powr<2>(p1) +
                    (10. - 8. * powr<2>(cosl1p2)) * cosp1p2 * p1 * p2 +
                    (20. + 10. * powr<2>(cosl1p2)) * powr<2>(p2))) +
         powr<5>(cosl1p2) * powr<5>(l1) * p2 *
             (-24. * powr<4>(l1) - 8. * powr<4>(p1) - 38. * cosp1p2 * powr<3>(p1) * p2 +
              (-42. + 20. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) +
              cosp1p2 * (-32. + 10. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) +
              (-34. + 10. * powr<2>(cosp1p2)) * powr<4>(p2) +
              powr<2>(l1) * (-30. * powr<2>(p1) - 52. * cosp1p2 * p1 * p2 +
                             (-54. + 26. * powr<2>(cosp1p2)) * powr<2>(p2))) +
         powr<5>(cosl1p1) * powr<4>(l1) * p1 *
             (-24. * powr<5>(l1) + cosl1p2 * powr<4>(l1) * (-52. * cosp1p2 * p1 + 72. * p2) +
              powr<3>(l1) * ((-54. + 26. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (-4. - 4. * powr<2>(cosp1p2))) *
                                 powr<2>(p1) +
                             (-52. + 104. * powr<2>(cosl1p2)) * cosp1p2 * p1 * p2 +
                             (-30. - 52. * powr<2>(cosl1p2)) * powr<2>(p2)) +
              cosl1p2 * powr<2>(l1) *
                  (-20. * cosp1p2 * powr<3>(p1) + 4. * powr<3>(cosp1p2) * powr<3>(p1) +
                   (112. + 4. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                   (-72. + 4. * powr<2>(cosl1p2)) * powr<2>(cosp1p2) * powr<2>(p1) * p2 +
                   (60. - 20. * powr<2>(cosl1p2)) * cosp1p2 * p1 * powr<2>(p2) +
                   (40. + 4. * powr<2>(cosl1p2)) * powr<3>(p2)) +
              cosl1p2 * p1 * p2 *
                  ((34. - 10. * powr<2>(cosp1p2)) * powr<3>(p1) +
                   cosp1p2 * (32. - 10. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                   (10. + 4. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + 6. * cosp1p2 * powr<3>(p2)) +
              l1 * ((-34. + 10. * powr<2>(cosp1p2)) * powr<4>(p1) +
                    cosp1p2 *
                        (-32. + 10. * powr<2>(cosp1p2) +
                         powr<2>(cosl1p2) * (20. - 4. * powr<2>(cosp1p2))) *
                        powr<3>(p1) * p2 +
                    (-42. + 20. * powr<2>(cosp1p2) +
                     powr<2>(cosl1p2) * (-26. + 22. * powr<2>(cosp1p2))) *
                        powr<2>(p1) * powr<2>(p2) +
                    (-38. - 8. * powr<2>(cosl1p2)) * cosp1p2 * p1 * powr<3>(p2) +
                    (-8. - 2. * powr<2>(cosl1p2)) * powr<4>(p2))) +
         powr<4>(cosl1p1) * powr<3>(l1) *
             (6. * powr<7>(l1) + cosl1p2 * powr<6>(l1) * (48. * cosp1p2 * p1 - 24. * p2) +
              powr<5>(l1) * ((12. - 24. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (26. + 26. * powr<2>(cosp1p2))) *
                                 powr<2>(p1) +
                             (36. - 144. * powr<2>(cosl1p2)) * cosp1p2 * p1 * p2 +
                             (10. + 26. * powr<2>(cosl1p2)) * powr<2>(p2)) +
              cosl1p2 * powr<4>(l1) *
                  ((56. + 8. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p1) -
                   26. * powr<3>(cosp1p2) * powr<3>(p1) +
                   (-62. - 52. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                   (176. - 52. * powr<2>(cosl1p2)) * powr<2>(cosp1p2) * powr<2>(p1) * p2 +
                   (-38. + 104. * powr<2>(cosl1p2)) * cosp1p2 * p1 * powr<2>(p2) +
                   (-20. - 4. * powr<2>(cosl1p2)) * powr<3>(p2)) +
              powr<3>(l1) * ((68. - 28. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (-14. + 2. * powr<2>(cosp1p2))) *
                                 powr<4>(p1) +
                             cosp1p2 *
                                 (107. - 8. * powr<4>(cosl1p2) - 52. * powr<2>(cosp1p2) +
                                  powr<2>(cosl1p2) * (-114. + 62. * powr<2>(cosp1p2))) *
                                 powr<3>(p1) * p2 +
                             (58. - 22. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (78. - 216. * powr<2>(cosp1p2)) +
                              powr<4>(cosl1p2) * (10. + 10. * powr<2>(cosp1p2))) *
                                 powr<2>(p1) * powr<2>(p2) +
                             (40. - 22. * powr<2>(cosl1p2) - 8. * powr<4>(cosl1p2)) * cosp1p2 * p1 *
                                 powr<3>(p2) +
                             (4. + 2. * powr<2>(cosl1p2)) * powr<4>(p2)) +
              cosl1p2 * powr<2>(p1) * p2 *
                  ((-26. + 8. * powr<2>(cosp1p2)) * powr<4>(p1) +
                   cosp1p2 * (-65. + 20. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 +
                   (8. - 25. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (-28. + 8. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) -
                   2. * powr<2>(cosp1p2) * powr<4>(p2)) +
              l1 * p1 *
                  ((26. - 8. * powr<2>(cosp1p2)) * powr<5>(p1) +
                   cosp1p2 *
                       (65. - 20. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (-40. + 10. * powr<2>(cosp1p2))) *
                       powr<4>(p1) * p2 +
                   (64. - 15. * powr<2>(cosp1p2) - 4. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (-18. - 36. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2))) *
                       powr<3>(p1) * powr<2>(p2) +
                   cosp1p2 *
                       (92. - 56. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (55. - 20. * powr<2>(cosp1p2))) *
                       powr<2>(p1) * powr<3>(p2) +
                   (36. - 4. * powr<2>(cosl1p2) - 2. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) +
                   (8. + 2. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) +
              cosl1p2 * powr<2>(l1) * p1 *
                  ((-122. + 14. * powr<2>(cosl1p2)) * powr<3>(p1) * p2 -
                   10. * powr<4>(cosp1p2) * powr<3>(p1) * p2 +
                   (-106. + 4. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) +
                   powr<3>(cosp1p2) * (-10. * powr<4>(p1) +
                                       (96. - 12. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2)) +
                   powr<2>(cosp1p2) * ((104. - 2. * powr<2>(cosl1p2)) * powr<3>(p1) * p2 +
                                       (90. + 16. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) +
                   cosp1p2 * (40. * powr<4>(p1) +
                              (-194. + 26. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) -
                              26. * powr<4>(p2)))) +
         powr<3>(cosl1p2) * powr<3>(l1) *
             (-8. * powr<5>(cosp1p2) * powr<3>(p1) * powr<4>(p2) +
              powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) *
                  (20. * powr<2>(l1) + 28. * powr<2>(p1) + 2. * powr<2>(p2)) +
              powr<3>(cosp1p2) * p1 * powr<2>(p2) *
                  (36. * powr<4>(l1) + 28. * powr<4>(p1) + 65. * powr<2>(p1) * powr<2>(p2) +
                   12. * powr<4>(p2) + powr<2>(l1) * (76. * powr<2>(p1) + 40. * powr<2>(p2))) +
              powr<2>(cosp1p2) * p2 *
                  (6. * powr<6>(l1) + 4. * powr<6>(p1) - 21. * powr<4>(p1) * powr<2>(p2) +
                   9. * powr<2>(p1) * powr<4>(p2) + 2. * powr<6>(p2) +
                   powr<4>(l1) * (14. * powr<2>(p1) - 12. * powr<2>(p2)) +
                   powr<2>(l1) * (12. * powr<4>(p1) - 21. * powr<2>(p1) * powr<2>(p2))) +
              cosp1p2 * p1 *
                  (-6. * powr<6>(l1) - 68. * powr<4>(p1) * powr<2>(p2) -
                   125. * powr<2>(p1) * powr<4>(p2) - 38. * powr<6>(p2) +
                   powr<4>(l1) * (-10. * powr<2>(p1) - 36. * powr<2>(p2)) +
                   powr<2>(l1) * (-4. * powr<4>(p1) - 131. * powr<2>(p1) * powr<2>(p2) -
                                  118. * powr<4>(p2))) +
              p2 * (18. * powr<6>(l1) - 8. * powr<6>(p1) - 52. * powr<4>(p1) * powr<2>(p2) -
                    32. * powr<2>(p1) * powr<4>(p2) - 6. * powr<6>(p2) +
                    powr<4>(l1) * (-12. * powr<2>(p1) + 12. * powr<2>(p2)) +
                    powr<2>(l1) * (-32. * powr<4>(p1) - 60. * powr<2>(p1) * powr<2>(p2) -
                                   4. * powr<4>(p2)))) +
         powr<4>(cosl1p2) * powr<4>(l1) *
             (6. * powr<6>(l1) +
              powr<4>(l1) * (10. * powr<2>(p1) + 36. * cosp1p2 * p1 * p2 +
                             (12. - 24. * powr<2>(cosp1p2)) * powr<2>(p2)) +
              powr<2>(l1) * (4. * powr<4>(p1) + 40. * cosp1p2 * powr<3>(p1) * p2 +
                             (58. - 22. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) +
                             cosp1p2 * (107. - 52. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) +
                             (68. - 28. * powr<2>(cosp1p2)) * powr<4>(p2)) +
              p2 * (36. * powr<4>(p1) * p2 + 64. * powr<2>(p1) * powr<3>(p2) -
                    4. * powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) + 26. * powr<5>(p2) +
                    powr<3>(cosp1p2) * (-56. * powr<3>(p1) * powr<2>(p2) - 20. * p1 * powr<4>(p2)) +
                    cosp1p2 * (8. * powr<5>(p1) + 92. * powr<3>(p1) * powr<2>(p2) +
                               65. * p1 * powr<4>(p2)) +
                    powr<2>(cosp1p2) * (-2. * powr<4>(p1) * p2 - 15. * powr<2>(p1) * powr<3>(p2) -
                                        8. * powr<5>(p2)))) +
         powr<2>(cosl1p2) * powr<2>(l1) *
             (-6. * powr<8>(l1) +
              powr<6>(l1) * ((-2. - 4. * powr<2>(cosp1p2)) * powr<2>(p1) +
                             cosp1p2 * (-3. - 6. * powr<2>(cosp1p2)) * p1 * p2 +
                             (-30. + 24. * powr<2>(cosp1p2)) * powr<2>(p2)) +
              powr<4>(l1) * ((8. - 6. * powr<2>(cosp1p2)) * powr<4>(p1) +
                             cosp1p2 * (65. - 32. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 +
                             (-44. + 30. * powr<2>(cosp1p2) - 8. * powr<4>(cosp1p2)) * powr<2>(p1) *
                                 powr<2>(p2) +
                             cosp1p2 * (8. + 25. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) +
                             (-70. + 32. * powr<2>(cosp1p2)) * powr<4>(p2)) +
              powr<2>(l1) * ((4. - 2. * powr<2>(cosp1p2)) * powr<6>(p1) +
                             cosp1p2 * (68. - 26. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 +
                             (-6. + 45. * powr<2>(cosp1p2) - 24. * powr<4>(cosp1p2)) * powr<4>(p1) *
                                 powr<2>(p2) +
                             cosp1p2 * (125. - 61. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2)) *
                                 powr<3>(p1) * powr<3>(p2) +
                             (-34. + 14. * powr<2>(cosp1p2) + 11. * powr<4>(cosp1p2)) *
                                 powr<2>(p1) * powr<4>(p2) +
                             cosp1p2 * (17. + powr<2>(cosp1p2)) * p1 * powr<5>(p2) +
                             (-22. + 8. * powr<2>(cosp1p2)) * powr<6>(p2)) +
              p1 * p2 *
                  (4. * powr<5>(p1) * p2 + 18. * powr<3>(p1) * powr<3>(p2) +
                   8. * powr<6>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + 4. * p1 * powr<5>(p2) +
                   powr<5>(cosp1p2) *
                       (6. * powr<4>(p1) * powr<2>(p2) + 18. * powr<2>(p1) * powr<4>(p2)) +
                   powr<2>(cosp1p2) * (16. * powr<5>(p1) * p2 + 16. * powr<3>(p1) * powr<3>(p2) -
                                       14. * p1 * powr<5>(p2)) +
                   powr<4>(cosp1p2) * (-6. * powr<5>(p1) * p2 - 22. * powr<3>(p1) * powr<3>(p2) +
                                       4. * p1 * powr<5>(p2)) +
                   powr<3>(cosp1p2) * (-4. * powr<6>(p1) - 58. * powr<4>(p1) * powr<2>(p2) -
                                       80. * powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2)) +
                   cosp1p2 * (12. * powr<6>(p1) + 68. * powr<4>(p1) * powr<2>(p2) +
                              78. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2)))) +
         cosl1p2 * l1 *
             (-2. * powr<7>(cosp1p2) * powr<5>(p1) * powr<4>(p2) +
              powr<6>(cosp1p2) * powr<4>(p1) * powr<3>(p2) *
                  (-6. * powr<2>(l1) - 4. * powr<2>(p1) - 12. * powr<2>(p2)) +
              powr<5>(cosp1p2) * powr<3>(p1) * powr<2>(p2) *
                  (-4. * powr<4>(l1) - 2. * powr<4>(p1) - 14. * powr<2>(p1) * powr<2>(p2) -
                   12. * powr<4>(p2) + powr<2>(l1) * (-4. * powr<2>(p1) - 29. * powr<2>(p2))) +
              powr<4>(cosp1p2) * powr<2>(p1) * p2 *
                  (18. * powr<4>(p1) * powr<2>(p2) + 32. * powr<2>(p1) * powr<4>(p2) -
                   2. * powr<6>(p2) + powr<4>(l1) * (6. * powr<2>(p1) - 35. * powr<2>(p2)) +
                   powr<2>(l1) *
                       (4. * powr<4>(p1) + 3. * powr<2>(p1) * powr<2>(p2) - 13. * powr<4>(p2))) +
              powr<2>(l1) * p2 *
                  (6. * powr<6>(l1) + 8. * powr<6>(p1) + 52. * powr<4>(p1) * powr<2>(p2) +
                   32. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2) +
                   powr<4>(l1) * (42. * powr<2>(p1) + 46. * powr<2>(p2)) +
                   powr<2>(l1) *
                       (40. * powr<4>(p1) + 102. * powr<2>(p1) * powr<2>(p2) + 38. * powr<4>(p2))) +
              powr<3>(cosp1p2) * p1 *
                  (10. * powr<6>(p1) * powr<2>(p2) + 58. * powr<4>(p1) * powr<4>(p2) +
                   46. * powr<2>(p1) * powr<6>(p2) +
                   powr<6>(l1) * (4. * powr<2>(p1) - 24. * powr<2>(p2)) +
                   powr<4>(l1) *
                       (6. * powr<4>(p1) - powr<2>(p1) * powr<2>(p2) - 26. * powr<4>(p2)) +
                   powr<2>(l1) * (2. * powr<6>(p1) + 28. * powr<4>(p1) * powr<2>(p2) +
                                  72. * powr<2>(p1) * powr<4>(p2) - 4. * powr<6>(p2))) +
              powr<2>(cosp1p2) * p2 *
                  (-6. * powr<8>(l1) - 18. * powr<6>(p1) * powr<2>(p2) -
                   26. * powr<4>(p1) * powr<4>(p2) + 6. * powr<2>(p1) * powr<6>(p2) +
                   powr<6>(l1) * (-10. * powr<2>(p1) - 14. * powr<2>(p2)) +
                   powr<4>(l1) *
                       (-22. * powr<4>(p1) - 7. * powr<2>(p1) * powr<2>(p2) - 10. * powr<4>(p2)) +
                   powr<2>(l1) * (-12. * powr<6>(p1) - 39. * powr<4>(p1) * powr<2>(p2) +
                                  5. * powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2))) +
              cosp1p2 * p1 *
                  (-12. * powr<6>(p1) * powr<2>(p2) - 36. * powr<4>(p1) * powr<4>(p2) -
                   30. * powr<2>(p1) * powr<6>(p2) +
                   powr<6>(l1) * (-12. * powr<2>(p1) + 16. * powr<2>(p2)) +
                   powr<4>(l1) *
                       (-18. * powr<4>(p1) + powr<2>(p1) * powr<2>(p2) + 42. * powr<4>(p2)) +
                   powr<2>(l1) * (-6. * powr<6>(p1) - 24. * powr<4>(p1) * powr<2>(p2) -
                                  29. * powr<2>(p1) * powr<4>(p2) + 8. * powr<6>(p2)))) +
         powr<2>(cosl1p1) * l1 *
             ((-6. + powr<2>(cosl1p2) * (6. + 6. * powr<2>(cosp1p2))) * powr<9>(l1) +
              cosl1p2 * powr<8>(l1) *
                  ((-48. + 48. * powr<2>(cosl1p2)) * cosp1p2 * p1 - 6. * powr<3>(cosp1p2) * p1 +
                   (24. - 24. * powr<2>(cosl1p2)) * p2 +
                   (12. - 24. * powr<2>(cosl1p2)) * powr<2>(cosp1p2) * p2) +
              powr<7>(l1) * ((-30. + 26. * powr<4>(cosl1p2) + 24. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (-24. - 34. * powr<2>(cosp1p2))) *
                                 powr<2>(p1) +
                             cosp1p2 *
                                 (-3. - 144. * powr<4>(cosl1p2) - 6. * powr<2>(cosp1p2) +
                                  powr<2>(cosl1p2) * (156. + 60. * powr<2>(cosp1p2))) *
                                 p1 * p2 +
                             (-2. - 4. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (-24. - 34. * powr<2>(cosp1p2)) +
                              powr<4>(cosl1p2) * (26. + 26. * powr<2>(cosp1p2))) *
                                 powr<2>(p2)) +
              powr<5>(l1) *
                  ((-70. + 2. * powr<4>(cosl1p2) + 32. * powr<2>(cosp1p2) +
                    powr<2>(cosl1p2) * (6. - 38. * powr<2>(cosp1p2))) *
                       powr<4>(p1) +
                   cosp1p2 *
                       (8. - 22. * powr<4>(cosl1p2) + 25. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (123. + 12. * powr<2>(cosp1p2))) *
                       powr<3>(p1) * p2 +
                   (-44. + 10. * powr<6>(cosl1p2) + 30. * powr<2>(cosp1p2) - 8. * powr<4>(cosp1p2) +
                    powr<4>(cosl1p2) * (78. - 216. * powr<2>(cosp1p2)) +
                    powr<2>(cosl1p2) * (-270. + 130. * powr<4>(cosp1p2))) *
                       powr<2>(p1) * powr<2>(p2) +
                   cosp1p2 *
                       (65. - 8. * powr<6>(cosl1p2) - 32. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (123. + 12. * powr<2>(cosp1p2)) +
                        powr<4>(cosl1p2) * (-114. + 62. * powr<2>(cosp1p2))) *
                       p1 * powr<3>(p2) +
                   (8. - 6. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (6. - 38. * powr<2>(cosp1p2)) +
                    powr<4>(cosl1p2) * (-14. + 2. * powr<2>(cosp1p2))) *
                       powr<4>(p2)) +
              powr<3>(l1) *
                  ((-22. + 8. * powr<2>(cosp1p2) +
                    powr<2>(cosl1p2) * (-4. + 2. * powr<2>(cosp1p2))) *
                       powr<6>(p1) +
                   cosp1p2 *
                       (17. + 2. * powr<4>(cosl1p2) + powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (23. + 10. * powr<2>(cosp1p2))) *
                       powr<5>(p1) * p2 +
                   (-34. - 4. * powr<4>(cosl1p2) + 14. * powr<2>(cosp1p2) + 11. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (-222. - 81. * powr<2>(cosp1p2) + 82. * powr<4>(cosp1p2))) *
                       powr<4>(p1) * powr<2>(p2) +
                   cosp1p2 *
                       (125. - 61. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2) +
                        powr<4>(cosl1p2) * (55. - 20. * powr<2>(cosp1p2)) +
                        powr<2>(cosl1p2) *
                            (-15. - 167. * powr<2>(cosp1p2) + 54. * powr<4>(cosp1p2))) *
                       powr<3>(p1) * powr<3>(p2) +
                   (-6. + 45. * powr<2>(cosp1p2) - 24. * powr<4>(cosp1p2) +
                    powr<4>(cosl1p2) * (-18. - 36. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) +
                    powr<2>(cosl1p2) * (-222. - 81. * powr<2>(cosp1p2) + 82. * powr<4>(cosp1p2))) *
                       powr<2>(p1) * powr<4>(p2) +
                   cosp1p2 *
                       (68. - 26. * powr<2>(cosp1p2) +
                        powr<4>(cosl1p2) * (-40. + 10. * powr<2>(cosp1p2)) +
                        powr<2>(cosl1p2) * (23. + 10. * powr<2>(cosp1p2))) *
                       p1 * powr<5>(p2) +
                   (4. - 2. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-4. + 2. * powr<2>(cosp1p2))) *
                       powr<6>(p2)) +
              cosl1p2 * powr<6>(l1) *
                  ((144. + 20. * powr<2>(cosl1p2) - 52. * powr<4>(cosl1p2)) * powr<2>(p1) * p2 -
                   36. * powr<4>(cosp1p2) * powr<2>(p1) * p2 +
                   (-4. + 8. * powr<2>(cosl1p2) - 4. * powr<4>(cosl1p2)) * powr<3>(p2) +
                   cosp1p2 * p1 *
                       ((-52. + 14. * powr<2>(cosl1p2)) * powr<2>(p1) +
                        (-94. - 46. * powr<2>(cosl1p2) + 104. * powr<4>(cosl1p2)) * powr<2>(p2)) +
                   powr<3>(cosp1p2) *
                       (-14. * powr<3>(p1) + (34. - 124. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                   powr<2>(cosp1p2) *
                       ((-98. + 180. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                        (32. + 20. * powr<2>(cosl1p2) - 4. * powr<4>(cosl1p2)) * powr<3>(p2))) +
              cosl1p2 * powr<2>(p1) * powr<2>(p2) *
                  (30. * powr<4>(p1) * p2 + 36. * powr<2>(p1) * powr<3>(p2) +
                   2. * powr<6>(cosp1p2) * powr<2>(p1) * powr<3>(p2) + 12. * powr<5>(p2) +
                   powr<3>(cosp1p2) * (2. * powr<5>(p1) - 32. * powr<3>(p1) * powr<2>(p2) -
                                       18. * p1 * powr<4>(p2)) +
                   powr<5>(cosp1p2) * (12. * powr<3>(p1) * powr<2>(p2) + 4. * p1 * powr<4>(p2)) +
                   cosp1p2 * (-6. * powr<5>(p1) + 26. * powr<3>(p1) * powr<2>(p2) +
                              18. * p1 * powr<4>(p2)) +
                   powr<2>(cosp1p2) * (-46. * powr<4>(p1) * p2 - 58. * powr<2>(p1) * powr<3>(p2) -
                                       10. * powr<5>(p2)) +
                   powr<4>(cosp1p2) * (12. * powr<4>(p1) * p2 + 14. * powr<2>(p1) * powr<3>(p2) +
                                       2. * powr<5>(p2))) +
              cosl1p2 * powr<4>(l1) *
                  ((204. - 16. * powr<2>(cosl1p2) - 2. * powr<4>(cosl1p2)) * powr<4>(p1) * p2 -
                   32. * powr<5>(cosp1p2) * powr<3>(p1) * powr<2>(p2) +
                   (176. + 136. * powr<2>(cosl1p2) - 26. * powr<4>(cosl1p2)) * powr<2>(p1) *
                       powr<3>(p2) +
                   (-16. + 16. * powr<2>(cosl1p2)) * powr<5>(p2) +
                   powr<4>(cosp1p2) * powr<2>(p1) * p2 *
                       (-48. * powr<2>(p1) + (-32. - 104. * powr<2>(cosl1p2)) * powr<2>(p2)) +
                   powr<3>(cosp1p2) * p1 *
                       (-2. * powr<4>(p1) +
                        (38. + 52. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) +
                        (74. - 68. * powr<2>(cosl1p2) - 4. * powr<4>(cosl1p2)) * powr<4>(p2)) +
                   cosp1p2 * p1 *
                       ((-22. - 2. * powr<2>(cosl1p2)) * powr<4>(p1) +
                        (-75. - 92. * powr<2>(cosl1p2) - 8. * powr<4>(cosl1p2)) * powr<2>(p1) *
                            powr<2>(p2) +
                        (-192. + 100. * powr<2>(cosl1p2) + 20. * powr<4>(cosl1p2)) * powr<4>(p2)) +
                   powr<2>(cosp1p2) * ((-4. + 76. * powr<2>(cosl1p2)) * powr<4>(p1) * p2 +
                                       (-63. + 222. * powr<2>(cosl1p2) + 22. * powr<4>(cosl1p2)) *
                                           powr<2>(p1) * powr<3>(p2) +
                                       (16. - 4. * powr<2>(cosl1p2)) * powr<5>(p2))) +
              l1 * p1 * p2 *
                  ((4. - 60. * powr<2>(cosl1p2)) * powr<5>(p1) * p2 +
                   (18. - 128. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) +
                   (8. - 2. * powr<2>(cosl1p2)) * powr<6>(cosp1p2) * powr<3>(p1) * powr<3>(p2) +
                   (4. - 60. * powr<2>(cosl1p2)) * p1 * powr<5>(p2) +
                   powr<5>(cosp1p2) * powr<2>(p1) * powr<2>(p2) *
                       ((18. - 8. * powr<2>(cosl1p2)) * powr<2>(p1) +
                        (6. - 8. * powr<2>(cosl1p2)) * powr<2>(p2)) +
                   powr<4>(cosp1p2) * p1 * p2 *
                       ((4. - 4. * powr<2>(cosl1p2)) * powr<4>(p1) +
                        (-22. - 38. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) +
                        (-6. - 4. * powr<2>(cosl1p2)) * powr<4>(p2)) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       ((-14. + 28. * powr<2>(cosl1p2)) * powr<4>(p1) +
                        (16. + 170. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) +
                        (16. + 28. * powr<2>(cosl1p2)) * powr<4>(p2)) +
                   cosp1p2 * ((6. - 6. * powr<2>(cosl1p2)) * powr<6>(p1) +
                              (78. - 36. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) +
                              (68. - 36. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) +
                              (12. - 6. * powr<2>(cosl1p2)) * powr<6>(p2)) +
                   powr<3>(cosp1p2) * ((-2. + 2. * powr<2>(cosl1p2)) * powr<6>(p1) +
                                       (-80. + 26. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) +
                                       (-58. + 26. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) +
                                       (-4. + 2. * powr<2>(cosl1p2)) * powr<6>(p2))) +
              cosl1p2 * powr<2>(l1) * p1 *
                  ((48. + 4. * powr<2>(cosl1p2)) * powr<5>(p1) * p2 +
                   (94. + 116. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) -
                   8. * powr<6>(cosp1p2) * powr<3>(p1) * powr<3>(p2) +
                   (60. + 86. * powr<2>(cosl1p2)) * p1 * powr<5>(p2) +
                   powr<5>(cosp1p2) * powr<2>(p1) * powr<2>(p2) *
                       (-22. * powr<2>(p1) + (-40. + 4. * powr<2>(cosl1p2)) * powr<2>(p2)) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       ((24. - 2. * powr<2>(cosl1p2)) * powr<4>(p1) +
                        (-26. - 101. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) +
                        (-27. - 2. * powr<2>(cosl1p2)) * powr<4>(p2)) +
                   powr<4>(cosp1p2) * (-12. * powr<5>(p1) * p2 +
                                       (-13. + 14. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) -
                                       2. * p1 * powr<5>(p2)) +
                   powr<3>(cosp1p2) * (-2. * powr<6>(p1) +
                                       (75. + 2. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) +
                                       (151. + 34. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) +
                                       (14. - 8. * powr<2>(cosl1p2)) * powr<6>(p2)) +
                   cosp1p2 * (6. * powr<6>(p1) +
                              (-71. - 15. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) +
                              (-93. - 18. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) +
                              (-52. + 28. * powr<2>(cosl1p2)) * powr<6>(p2)))) +
         powr<3>(cosl1p1) * powr<2>(l1) *
             (powr<7>(l1) * ((18. + 6. * powr<2>(cosp1p2)) * p1 - 6. * cosp1p2 * p2) +
              powr<5>(cosl1p2) * powr<4>(l1) * p1 * p2 *
                  (4. * powr<2>(p1) - 20. * cosp1p2 * p1 * p2 +
                   (4. + 4. * powr<2>(cosp1p2)) * powr<2>(p2)) +
              powr<5>(l1) * ((12. - 12. * powr<2>(cosp1p2)) * powr<3>(p1) +
                             cosp1p2 * (-36. + 36. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                             (-12. + 14. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) -
                             10. * cosp1p2 * powr<3>(p2)) +
              powr<3>(l1) *
                  (-4. * powr<5>(p1) +
                   cosp1p2 * (-118. + 40. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 +
                   (-60. - 21. * powr<2>(cosp1p2) + 20. * powr<4>(cosp1p2)) * powr<3>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (-131. + 76. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) +
                   (-32. + 12. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) -
                   4. * cosp1p2 * powr<5>(p2)) +
              l1 * p1 *
                  ((-6. + 2. * powr<2>(cosp1p2)) * powr<6>(p1) +
                   cosp1p2 * (-38. + 12. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 +
                   (-32. + 9. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) * powr<4>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (-125. + 65. * powr<2>(cosp1p2) - 8. * powr<4>(cosp1p2)) *
                       powr<3>(p1) * powr<3>(p2) +
                   (-52. - 21. * powr<2>(cosp1p2) + 28. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       powr<4>(p2) +
                   cosp1p2 * (-68. + 28. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) +
                   (-8. + 4. * powr<2>(cosp1p2)) * powr<6>(p2)) +
              powr<4>(cosl1p2) *
                  (powr<3>(l1) * p1 * powr<2>(p2) *
                       ((4. + 16. * powr<2>(cosp1p2)) * powr<2>(p1) +
                        cosp1p2 * (26. - 12. * powr<2>(cosp1p2)) * p1 * p2 +
                        (14. - 2. * powr<2>(cosp1p2)) * powr<2>(p2)) +
                   powr<5>(l1) * (-4. * powr<3>(p1) + 104. * cosp1p2 * powr<2>(p1) * p2 +
                                  (-52. - 52. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                                  8. * cosp1p2 * powr<3>(p2))) +
              powr<3>(cosl1p2) * powr<2>(l1) *
                  (4. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) +
                   powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) *
                       (144. * powr<2>(l1) + 4. * powr<2>(p1) + 4. * powr<2>(p2)) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (72. * powr<4>(l1) + 4. * powr<4>(p1) - 10. * powr<2>(p1) * powr<2>(p2) +
                        4. * powr<4>(p2) + powr<2>(l1) * (-56. * powr<2>(p1) - 56. * powr<2>(p2))) +
                   cosp1p2 * (-52. * powr<2>(l1) * powr<2>(p1) * powr<2>(p2) -
                              12. * powr<4>(p1) * powr<2>(p2) - 12. * powr<2>(p1) * powr<4>(p2) +
                              powr<4>(l1) * (-52. * powr<2>(p1) - 52. * powr<2>(p2))) +
                   p1 * p2 *
                       (72. * powr<4>(l1) - 16. * powr<4>(p1) - 112. * powr<2>(p1) * powr<2>(p2) -
                        16. * powr<4>(p2) + powr<2>(l1) * (-8. * powr<2>(p1) - 8. * powr<2>(p2)))) +
              cosl1p2 *
                  (powr<5>(cosp1p2) * powr<4>(p1) * powr<2>(p2) *
                       (12. * powr<2>(l1) - 8. * powr<2>(p2)) +
                   powr<4>(cosp1p2) * powr<3>(p1) * p2 *
                       (52. * powr<4>(l1) - 22. * powr<2>(p1) * powr<2>(p2) - 10. * powr<4>(p2) +
                        powr<2>(l1) * (20. * powr<2>(p1) - 4. * powr<2>(p2))) +
                   powr<3>(cosp1p2) * powr<2>(p1) *
                       (24. * powr<6>(l1) - 12. * powr<4>(p1) * powr<2>(p2) -
                        powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2) +
                        powr<4>(l1) * (32. * powr<2>(p1) - 120. * powr<2>(p2)) +
                        powr<2>(l1) * (8. * powr<4>(p1) - 106. * powr<2>(p1) * powr<2>(p2) -
                                       108. * powr<4>(p2))) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (-96. * powr<6>(l1) - 2. * powr<6>(p1) + 69. * powr<4>(p1) * powr<2>(p2) +
                        47. * powr<2>(p1) * powr<4>(p2) +
                        powr<4>(l1) * (-96. * powr<2>(p1) - 110. * powr<2>(p2)) +
                        powr<2>(l1) * (-76. * powr<4>(p1) + 12. * powr<2>(p1) * powr<2>(p2) -
                                       30. * powr<4>(p2))) +
                   p1 * p2 *
                       (-48. * powr<6>(l1) + 6. * powr<6>(p1) - 42. * powr<4>(p1) * powr<2>(p2) -
                        32. * powr<2>(p1) * powr<4>(p2) +
                        powr<4>(l1) * (-44. * powr<2>(p1) + 52. * powr<2>(p2)) +
                        powr<2>(l1) * (-8. * powr<4>(p1) + 32. * powr<2>(p1) * powr<2>(p2) +
                                       32. * powr<4>(p2))) +
                   cosp1p2 * (-12. * powr<8>(l1) + 38. * powr<6>(p1) * powr<2>(p2) +
                              7. * powr<4>(p1) * powr<4>(p2) + 8. * powr<2>(p1) * powr<6>(p2) +
                              powr<6>(l1) * (24. * powr<2>(p1) + 4. * powr<2>(p2)) +
                              powr<4>(l1) * (-36. * powr<4>(p1) + 164. * powr<2>(p1) * powr<2>(p2) +
                                             12. * powr<4>(p2)) +
                              powr<2>(l1) * (-28. * powr<6>(p1) + 202. * powr<4>(p1) * powr<2>(p2) +
                                             198. * powr<2>(p1) * powr<4>(p2)))) +
              powr<2>(cosl1p2) *
                  (powr<7>(l1) * ((-24. - 24. * powr<2>(cosp1p2)) * p1 + 48. * cosp1p2 * p2) +
                   powr<5>(l1) * ((8. + 20. * powr<2>(cosp1p2)) * powr<3>(p1) +
                                  cosp1p2 * (-46. - 124. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                                  (20. + 180. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                                  14. * cosp1p2 * powr<3>(p2)) +
                   powr<3>(l1) *
                       ((16. - 4. * powr<2>(cosp1p2)) * powr<5>(p1) +
                        cosp1p2 * (100. - 68. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 +
                        (136. + 222. * powr<2>(cosp1p2) - 104. * powr<4>(cosp1p2)) * powr<3>(p1) *
                            powr<2>(p2) +
                        cosp1p2 * (-92. + 52. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) +
                        (-16. + 76. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) -
                        2. * cosp1p2 * powr<5>(p2)) +
                   l1 * p1 * p2 *
                       (86. * powr<4>(p1) * p2 + 4. * powr<5>(cosp1p2) * powr<3>(p1) * powr<2>(p2) +
                        116. * powr<2>(p1) * powr<3>(p2) +
                        14. * powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) + 4. * powr<5>(p2) +
                        cosp1p2 * (28. * powr<5>(p1) - 18. * powr<3>(p1) * powr<2>(p2) -
                                   15. * p1 * powr<4>(p2)) +
                        powr<3>(cosp1p2) * (-8. * powr<5>(p1) + 34. * powr<3>(p1) * powr<2>(p2) +
                                            2. * p1 * powr<4>(p2)) +
                        powr<2>(cosp1p2) *
                            (-2. * powr<4>(p1) * p2 - 101. * powr<2>(p1) * powr<3>(p2) -
                             2. * powr<5>(p2))))) +
         cosl1p1 *
             ((6. - 6. * powr<2>(cosp1p2)) * powr<9>(l1) * p1 +
              4. * powr<7>(cosl1p2) * powr<6>(l1) * p1 * powr<3>(p2) +
              powr<7>(l1) * ((46. - 14. * powr<2>(cosp1p2)) * powr<3>(p1) +
                             cosp1p2 * (16. - 24. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                             (42. - 10. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                             cosp1p2 * (-12. + 4. * powr<2>(cosp1p2)) * powr<3>(p2)) +
              powr<5>(l1) *
                  ((38. - 10. * powr<2>(cosp1p2)) * powr<5>(p1) +
                   cosp1p2 * (42. - 26. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 +
                   (102. - 7. * powr<2>(cosp1p2) - 35. * powr<4>(cosp1p2)) * powr<3>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (1. - powr<2>(cosp1p2) - 4. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       powr<3>(p2) +
                   (40. - 22. * powr<2>(cosp1p2) + 6. * powr<4>(cosp1p2)) * p1 * powr<4>(p2) +
                   cosp1p2 * (-18. + 6. * powr<2>(cosp1p2)) * powr<5>(p2)) +
              powr<3>(l1) *
                  ((6. - 2. * powr<2>(cosp1p2)) * powr<7>(p1) +
                   cosp1p2 * (8. - 4. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 +
                   (32. + 5. * powr<2>(cosp1p2) - 13. * powr<4>(cosp1p2)) * powr<5>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (-29. + 72. * powr<2>(cosp1p2) - 29. * powr<4>(cosp1p2)) *
                       powr<4>(p1) * powr<3>(p2) +
                   (52. - 39. * powr<2>(cosp1p2) + 3. * powr<4>(cosp1p2) - 6. * powr<6>(cosp1p2)) *
                       powr<3>(p1) * powr<4>(p2) +
                   cosp1p2 * (-24. + 28. * powr<2>(cosp1p2) - 4. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       powr<5>(p2) +
                   (8. - 12. * powr<2>(cosp1p2) + 4. * powr<4>(cosp1p2)) * p1 * powr<6>(p2) +
                   cosp1p2 * (-6. + 2. * powr<2>(cosp1p2)) * powr<7>(p2)) +
              powr<6>(cosl1p2) * powr<5>(l1) * powr<2>(p2) *
                  (powr<2>(l1) * (-52. * p1 + 8. * cosp1p2 * p2) +
                   p1 * (-4. * powr<2>(p1) - 10. * cosp1p2 * p1 * p2 +
                         (-14. + 4. * powr<2>(cosp1p2)) * powr<2>(p2))) +
              cosp1p2 * l1 * powr<2>(p1) * powr<2>(p2) *
                  (-30. * powr<4>(p1) * p2 - 36. * powr<2>(p1) * powr<3>(p2) -
                   2. * powr<6>(cosp1p2) * powr<2>(p1) * powr<3>(p2) - 12. * powr<5>(p2) +
                   cosp1p2 * (6. * powr<5>(p1) - 26. * powr<3>(p1) * powr<2>(p2) -
                              18. * p1 * powr<4>(p2)) +
                   powr<5>(cosp1p2) * (-12. * powr<3>(p1) * powr<2>(p2) - 4. * p1 * powr<4>(p2)) +
                   powr<3>(cosp1p2) * (-2. * powr<5>(p1) + 32. * powr<3>(p1) * powr<2>(p2) +
                                       18. * p1 * powr<4>(p2)) +
                   powr<4>(cosp1p2) * (-12. * powr<4>(p1) * p2 - 14. * powr<2>(p1) * powr<3>(p2) -
                                       2. * powr<5>(p2)) +
                   powr<2>(cosp1p2) * (46. * powr<4>(p1) * p2 + 58. * powr<2>(p1) * powr<3>(p2) +
                                       10. * powr<5>(p2))) +
              powr<5>(cosl1p2) * powr<4>(l1) * p2 *
                  (powr<4>(l1) * (72. * p1 - 52. * cosp1p2 * p2) +
                   powr<2>(l1) * (40. * powr<3>(p1) + 60. * cosp1p2 * powr<2>(p1) * p2 +
                                  (112. - 72. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                                  cosp1p2 * (-20. + 4. * powr<2>(cosp1p2)) * powr<3>(p2)) +
                   p1 * p2 *
                       (10. * powr<2>(p1) * p2 - 10. * powr<3>(cosp1p2) * p1 * powr<2>(p2) +
                        34. * powr<3>(p2) + cosp1p2 * (6. * powr<3>(p1) + 32. * p1 * powr<2>(p2)) +
                        powr<2>(cosp1p2) * (4. * powr<2>(p1) * p2 - 10. * powr<3>(p2)))) +
              powr<4>(cosl1p2) *
                  (powr<9>(l1) * (-24. * p1 + 48. * cosp1p2 * p2) +
                   powr<7>(l1) * (-20. * powr<3>(p1) - 38. * cosp1p2 * powr<2>(p1) * p2 +
                                  (-62. + 176. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                                  cosp1p2 * (56. - 26. * powr<2>(cosp1p2)) * powr<3>(p2)) +
                   powr<3>(l1) * p1 * powr<2>(p2) *
                       (8. * powr<2>(p1) * powr<2>(p2) +
                        12. * powr<4>(cosp1p2) * powr<2>(p1) * powr<2>(p2) - 26. * powr<4>(p2) +
                        cosp1p2 * (-28. * powr<3>(p1) * p2 - 65. * p1 * powr<3>(p2)) +
                        powr<3>(cosp1p2) * (8. * powr<3>(p1) * p2 + 20. * p1 * powr<3>(p2)) +
                        powr<2>(cosp1p2) * (-2. * powr<4>(p1) - 25. * powr<2>(p1) * powr<2>(p2) +
                                            8. * powr<4>(p2))) +
                   powr<5>(l1) * p2 *
                       (-106. * powr<3>(p1) * p2 - 122. * p1 * powr<3>(p2) -
                        10. * powr<4>(cosp1p2) * p1 * powr<3>(p2) +
                        powr<2>(cosp1p2) * (90. * powr<3>(p1) * p2 + 104. * p1 * powr<3>(p2)) +
                        powr<3>(cosp1p2) * (96. * powr<2>(p1) * powr<2>(p2) - 10. * powr<4>(p2)) +
                        cosp1p2 * (-26. * powr<4>(p1) - 194. * powr<2>(p1) * powr<2>(p2) +
                                   40. * powr<4>(p2)))) +
              powr<3>(cosl1p2) * powr<2>(l1) *
                  (powr<5>(cosp1p2) * powr<2>(p1) * (12. * powr<2>(l1) - 8. * powr<2>(p1)) *
                       powr<4>(p2) +
                   powr<4>(cosp1p2) * p1 * powr<3>(p2) *
                       (52. * powr<4>(l1) - 10. * powr<4>(p1) - 22. * powr<2>(p1) * powr<2>(p2) +
                        powr<2>(l1) * (-4. * powr<2>(p1) + 20. * powr<2>(p2))) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (-96. * powr<6>(l1) + 47. * powr<4>(p1) * powr<2>(p2) +
                        69. * powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2) +
                        powr<4>(l1) * (-110. * powr<2>(p1) - 96. * powr<2>(p2)) +
                        powr<2>(l1) * (-30. * powr<4>(p1) + 12. * powr<2>(p1) * powr<2>(p2) -
                                       76. * powr<4>(p2))) +
                   p1 * p2 *
                       (-48. * powr<6>(l1) - 32. * powr<4>(p1) * powr<2>(p2) -
                        42. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2) +
                        powr<4>(l1) * (52. * powr<2>(p1) - 44. * powr<2>(p2)) +
                        powr<2>(l1) * (32. * powr<4>(p1) + 32. * powr<2>(p1) * powr<2>(p2) -
                                       8. * powr<4>(p2))) +
                   powr<3>(cosp1p2) * powr<2>(p2) *
                       (24. * powr<6>(l1) - 2. * powr<6>(p1) - powr<4>(p1) * powr<2>(p2) -
                        12. * powr<2>(p1) * powr<4>(p2) +
                        powr<4>(l1) * (-120. * powr<2>(p1) + 32. * powr<2>(p2)) +
                        powr<2>(l1) * (-108. * powr<4>(p1) - 106. * powr<2>(p1) * powr<2>(p2) +
                                       8. * powr<4>(p2))) +
                   cosp1p2 *
                       (-12. * powr<8>(l1) + 8. * powr<6>(p1) * powr<2>(p2) +
                        7. * powr<4>(p1) * powr<4>(p2) + 38. * powr<2>(p1) * powr<6>(p2) +
                        powr<6>(l1) * (4. * powr<2>(p1) + 24. * powr<2>(p2)) +
                        powr<4>(l1) * (12. * powr<4>(p1) + 164. * powr<2>(p1) * powr<2>(p2) -
                                       36. * powr<4>(p2)) +
                        powr<2>(l1) * (198. * powr<4>(p1) * powr<2>(p2) +
                                       202. * powr<2>(p1) * powr<4>(p2) - 28. * powr<6>(p2)))) +
              powr<2>(cosl1p2) *
                  (powr<9>(l1) * ((24. + 12. * powr<2>(cosp1p2)) * p1 +
                                  cosp1p2 * (-48. - 6. * powr<2>(cosp1p2)) * p2) +
                   powr<7>(l1) * ((-4. + 32. * powr<2>(cosp1p2)) * powr<3>(p1) +
                                  cosp1p2 * (-94. + 34. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                                  (144. - 98. * powr<2>(cosp1p2) - 36. * powr<4>(cosp1p2)) * p1 *
                                      powr<2>(p2) +
                                  cosp1p2 * (-52. - 14. * powr<2>(cosp1p2)) * powr<3>(p2)) +
                   powr<5>(l1) *
                       ((-16. + 16. * powr<2>(cosp1p2)) * powr<5>(p1) +
                        cosp1p2 * (-192. + 74. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 +
                        (176. - 63. * powr<2>(cosp1p2) - 32. * powr<4>(cosp1p2)) * powr<3>(p1) *
                            powr<2>(p2) +
                        cosp1p2 * (-75. + 38. * powr<2>(cosp1p2) - 32. * powr<4>(cosp1p2)) *
                            powr<2>(p1) * powr<3>(p2) +
                        (204. - 4. * powr<2>(cosp1p2) - 48. * powr<4>(cosp1p2)) * p1 * powr<4>(p2) +
                        cosp1p2 * (-22. - 2. * powr<2>(cosp1p2)) * powr<5>(p2)) +
                   l1 * powr<2>(p1) * powr<2>(p2) *
                       ((12. - 10. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) * powr<5>(p1) +
                        cosp1p2 * (18. - 18. * powr<2>(cosp1p2) + 4. * powr<4>(cosp1p2)) *
                            powr<4>(p1) * p2 +
                        (36. - 58. * powr<2>(cosp1p2) + 14. * powr<4>(cosp1p2) +
                         2. * powr<6>(cosp1p2)) *
                            powr<3>(p1) * powr<2>(p2) +
                        cosp1p2 * (26. - 32. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2)) *
                            powr<2>(p1) * powr<3>(p2) +
                        (30. - 46. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2)) * p1 * powr<4>(p2) +
                        cosp1p2 * (-6. + 2. * powr<2>(cosp1p2)) * powr<5>(p2)) +
                   powr<3>(l1) * p2 *
                       (60. * powr<5>(p1) * p2 + 94. * powr<3>(p1) * powr<3>(p2) -
                        8. * powr<6>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + 48. * p1 * powr<5>(p2) +
                        powr<5>(cosp1p2) *
                            (-40. * powr<4>(p1) * powr<2>(p2) - 22. * powr<2>(p1) * powr<4>(p2)) +
                        powr<4>(cosp1p2) *
                            (-2. * powr<5>(p1) * p2 - 13. * powr<3>(p1) * powr<3>(p2) -
                             12. * p1 * powr<5>(p2)) +
                        powr<2>(cosp1p2) *
                            (-27. * powr<5>(p1) * p2 - 26. * powr<3>(p1) * powr<3>(p2) +
                             24. * p1 * powr<5>(p2)) +
                        powr<3>(cosp1p2) * (14. * powr<6>(p1) + 151. * powr<4>(p1) * powr<2>(p2) +
                                            75. * powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2)) +
                        cosp1p2 * (-52. * powr<6>(p1) - 93. * powr<4>(p1) * powr<2>(p2) -
                                   71. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2)))) +
              cosl1p2 *
                  (2. * powr<7>(cosp1p2) * powr<2>(l1) * powr<4>(p1) * powr<4>(p2) +
                   powr<6>(cosp1p2) * powr<3>(p1) * powr<3>(p2) *
                       (6. * powr<4>(l1) - 2. * powr<2>(p1) * powr<2>(p2) +
                        powr<2>(l1) * (12. * powr<2>(p1) + 12. * powr<2>(p2))) +
                   powr<5>(cosp1p2) * powr<2>(p1) * powr<2>(p2) *
                       (12. * powr<6>(l1) - 4. * powr<4>(p1) * powr<2>(p2) -
                        4. * powr<2>(p1) * powr<4>(p2) +
                        powr<4>(l1) * (28. * powr<2>(p1) + 28. * powr<2>(p2)) +
                        powr<2>(l1) * (12. * powr<4>(p1) + 55. * powr<2>(p1) * powr<2>(p2) +
                                       12. * powr<4>(p2))) +
                   p1 * p2 *
                       (-54. * powr<8>(l1) - 6. * powr<6>(p1) * powr<2>(p2) -
                        10. * powr<4>(p1) * powr<4>(p2) - 6. * powr<2>(p1) * powr<6>(p2) +
                        powr<6>(l1) * (-180. * powr<2>(p1) - 180. * powr<2>(p2)) +
                        powr<4>(l1) * (-98. * powr<4>(p1) - 238. * powr<2>(p1) * powr<2>(p2) -
                                       98. * powr<4>(p2)) +
                        powr<2>(l1) * (-12. * powr<6>(p1) - 42. * powr<4>(p1) * powr<2>(p2) -
                                       42. * powr<2>(p1) * powr<4>(p2) - 12. * powr<6>(p2))) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (66. * powr<8>(l1) + 8. * powr<6>(p1) * powr<2>(p2) +
                        16. * powr<4>(p1) * powr<4>(p2) + 8. * powr<2>(p1) * powr<6>(p2) +
                        powr<6>(l1) * (76. * powr<2>(p1) + 76. * powr<2>(p2)) +
                        powr<4>(l1) * (36. * powr<4>(p1) + 24. * powr<2>(p1) * powr<2>(p2) +
                                       36. * powr<4>(p2)) +
                        powr<2>(l1) * (-2. * powr<6>(p1) + 33. * powr<4>(p1) * powr<2>(p2) +
                                       33. * powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2))) +
                   powr<4>(cosp1p2) * p1 * p2 *
                       (6. * powr<8>(l1) - 2. * powr<6>(p1) * powr<2>(p2) -
                        4. * powr<4>(p1) * powr<4>(p2) - 2. * powr<2>(p1) * powr<6>(p2) +
                        powr<6>(l1) * (14. * powr<2>(p1) + 14. * powr<2>(p2)) +
                        powr<4>(l1) * (2. * powr<4>(p1) + 68. * powr<2>(p1) * powr<2>(p2) +
                                       2. * powr<4>(p2)) +
                        powr<2>(l1) * (2. * powr<6>(p1) - 19. * powr<4>(p1) * powr<2>(p2) -
                                       19. * powr<2>(p1) * powr<4>(p2) + 2. * powr<6>(p2))) +
                   powr<3>(cosp1p2) *
                       (16. * powr<6>(p1) * powr<4>(p2) + 16. * powr<4>(p1) * powr<6>(p2) +
                        powr<6>(l1) * (-12. * powr<4>(p1) + 64. * powr<2>(p1) * powr<2>(p2) -
                                       12. * powr<4>(p2)) +
                        powr<4>(l1) * (-8. * powr<6>(p1) - 50. * powr<4>(p1) * powr<2>(p2) -
                                       50. * powr<2>(p1) * powr<4>(p2) - 8. * powr<6>(p2)) +
                        powr<2>(l1) *
                            (-44. * powr<6>(p1) * powr<2>(p2) - 150. * powr<4>(p1) * powr<4>(p2) -
                             44. * powr<2>(p1) * powr<6>(p2))) +
                   cosp1p2 * (12. * powr<10>(l1) - 12. * powr<6>(p1) * powr<4>(p2) -
                              12. * powr<4>(p1) * powr<6>(p2) +
                              powr<8>(l1) * (28. * powr<2>(p1) + 28. * powr<2>(p2)) +
                              powr<6>(l1) * (56. * powr<4>(p1) + 56. * powr<4>(p2)) +
                              powr<4>(l1) * (28. * powr<6>(p1) + 12. * powr<4>(p1) * powr<2>(p2) +
                                             12. * powr<2>(p1) * powr<4>(p2) + 28. * powr<6>(p2)) +
                              powr<2>(l1) * (32. * powr<6>(p1) * powr<2>(p2) +
                                             57. * powr<4>(p1) * powr<4>(p2) +
                                             32. * powr<2>(p1) * powr<6>(p2)))))) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
            sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                 (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                  (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                  2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
            atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) *
                      powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                  -1. + 3. * l1 * (l1 - cosl1p1 * p1) *
                            powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
            sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                 (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                  (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                  2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
            atan2(-1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) *
                      powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                  -1. + 3. * l1 * (l1 - cosl1p2 * p2) *
                            powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                     l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
            0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                       (3. * powr<2>(p2) * powr<2>(-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) +
                        powr<2>(powr<2>(p2) -
                                2. * (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                                      p1 * (p1 + cosp1p2 * p2))))),
            atan2(1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
                  -1. + 3. *
                            powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                     l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                            (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                             p1 * (p1 + cosp1p2 * p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                     l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
            0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                       (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                        powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
            atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                  2. - 3. * powr<2>(p1) *
                           powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                    2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) +
    12. * powr<-1>(11. + 6. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-2>(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
        ((-4. + 1. * powr<4>(cosl1p1) + 1. * powr<4>(cosl1p2) -
          2. * powr<3>(cosl1p1) * cosl1p2 * cosp1p2 +
          cosl1p1 * cosl1p2 * (6. - 2. * powr<2>(cosl1p2)) * cosp1p2 +
          powr<2>(cosl1p2) * (3. - 2. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p1) * (-3. + powr<2>(cosl1p2) * (1. + 1. * powr<2>(cosp1p2)))) *
             powr<6>(l1) +
         (-2. * powr<5>(cosl1p2) + cosl1p1 * (36. - 2. * powr<2>(cosl1p1)) * cosp1p2 +
          4. * cosl1p1 * powr<4>(cosl1p2) * cosp1p2 +
          cosl1p1 * powr<2>(cosl1p2) * cosp1p2 *
              (-20. + 4. * powr<2>(cosl1p1) - 2. * powr<2>(cosp1p2)) +
          powr<3>(cosl1p2) *
              (-34. + 10. * powr<2>(cosp1p2) + powr<2>(cosl1p1) * (-2. - 2. * powr<2>(cosp1p2))) +
          cosl1p2 * (36. - 2. * powr<4>(cosl1p1) - 20. * powr<2>(cosp1p2) +
                     powr<2>(cosl1p1) * (-8. + 4. * powr<2>(cosp1p2)))) *
             powr<5>(l1) * p2 +
         (-36. + 1. * powr<4>(cosl1p1) * powr<2>(cosl1p2) + 1. * powr<6>(cosl1p2) +
          powr<3>(cosl1p1) * cosl1p2 * (4. - 2. * powr<2>(cosl1p2)) * cosp1p2 -
          28. * powr<2>(cosp1p2) + powr<4>(cosl1p2) * (68. - 11. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p2) * (-33. + 79. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) +
          powr<2>(cosl1p1) *
              (5. - powr<2>(cosp1p2) + powr<2>(cosl1p2) * (34. - 5. * powr<2>(cosp1p2)) +
               powr<4>(cosl1p2) * (1. + 1. * powr<2>(cosp1p2))) +
          cosl1p1 * cosl1p2 * cosp1p2 *
              (-94. - 2. * powr<4>(cosl1p2) - 2. * powr<2>(cosp1p2) +
               powr<2>(cosl1p2) * (16. + 2. * powr<2>(cosp1p2)))) *
             powr<4>(l1) * powr<2>(p2) +
         (-26. * powr<5>(cosl1p2) +
          cosl1p1 * (36. - 2. * powr<2>(cosl1p1)) * powr<2>(cosl1p2) * cosp1p2 +
          2. * cosl1p1 * powr<4>(cosl1p2) * powr<3>(cosp1p2) +
          cosl1p1 * cosp1p2 * (48. + 4. * powr<2>(cosp1p2)) +
          cosl1p2 * (126. - 24. * powr<2>(cosl1p1) + 54. * powr<2>(cosp1p2)) +
          powr<3>(cosl1p2) *
              (-100. - 18. * powr<2>(cosl1p1) - 96. * powr<2>(cosp1p2) - 4. * powr<4>(cosp1p2))) *
             powr<3>(l1) * powr<3>(p2) +
         (-49. - 43. * powr<2>(cosp1p2) - 2. * powr<4>(cosp1p2) +
          cosl1p1 * cosl1p2 * cosp1p2 * (-64. + 36. * powr<2>(cosl1p2) - 4. * powr<2>(cosp1p2)) +
          powr<4>(cosl1p2) * (81. + 36. * powr<2>(cosp1p2) + 1. * powr<4>(cosp1p2)) +
          powr<2>(cosl1p2) * (-32. + 21. * powr<2>(cosp1p2) + 3. * powr<4>(cosp1p2)) +
          powr<2>(cosl1p1) *
              (5. - powr<2>(cosp1p2) + powr<2>(cosl1p2) * (9. + 3. * powr<2>(cosp1p2)))) *
             powr<2>(l1) * powr<4>(p2) +
         (cosl1p1 * powr<2>(cosl1p2) * cosp1p2 * (-18. - 2. * powr<2>(cosp1p2)) +
          cosl1p1 * cosp1p2 * (18. + 2. * powr<2>(cosp1p2)) +
          powr<3>(cosl1p2) * (-68. - 54. * powr<2>(cosp1p2) - 2. * powr<4>(cosp1p2)) +
          cosl1p2 * (68. + 54. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2))) *
             l1 * powr<5>(p2) +
         (-17. - 18. * powr<2>(cosp1p2) - powr<4>(cosp1p2) +
          powr<2>(cosl1p2) * (17. + 18. * powr<2>(cosp1p2) + 1. * powr<4>(cosp1p2))) *
             powr<6>(p2)) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + 1. * powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
            sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                 (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                  (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                  2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
            atan2(-1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) *
                      powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                  -1. + 3. * l1 * (l1 - cosl1p2 * p2) *
                            powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
            sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                 (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                  (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                  2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
            atan2(1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) *
                      powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                  -1. + 3. * l1 * (l1 - cosl1p2 * p2) *
                            powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZA4SP(0.7071067811865475 *
              sqrt(powr<2>(l1) + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2))) +
    0.375 * powr<-1>(11. + 6. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(-0.5 * powr<2>(l1) - cosl1p1 * l1 * p1 - 0.5 * powr<2>(p1) +
                 1. * cosl1p2 * l1 * p2 + 1. * cosp1p2 * p1 * p2 - 0.5 * powr<2>(p2)) *
        ((47. + 1. * powr<4>(cosl1p1) + 1. * powr<4>(cosl1p2) -
          2. * powr<3>(cosl1p1) * cosl1p2 * cosp1p2 + 10. * powr<2>(cosp1p2) +
          15. * powr<4>(cosp1p2) +
          cosl1p1 * cosl1p2 * cosp1p2 * (28. - 2. * powr<2>(cosl1p2) - 26. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p2) * (34. + 13. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p1) *
              (34. + 13. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (1. + 1. * powr<2>(cosp1p2)))) *
             powr<2>(l1) +
         (69. + 35.5 * powr<2>(cosp1p2) + 7.5 * powr<4>(cosp1p2) +
          powr<2>(cosl1p2) * (15.5 - powr<2>(cosp1p2)) +
          cosl1p1 * cosl1p2 * cosp1p2 * (17. + 2. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p1) * (13. + 16. * powr<2>(cosp1p2) - 6.5 * powr<4>(cosp1p2))) *
             powr<2>(p1) +
         (powr<2>(cosl1p1) * (-61. * cosp1p2 + 2. * powr<3>(cosp1p2)) +
          cosl1p1 * cosl1p2 * (6. - 10. * powr<2>(cosp1p2) + 10. * powr<4>(cosp1p2)) +
          cosp1p2 * (-157. - 51. * powr<2>(cosp1p2) - 16. * powr<4>(cosp1p2) +
                     powr<2>(cosl1p2) * (-61. + 2. * powr<2>(cosp1p2)))) *
             p1 * p2 +
         (69. + 35.5 * powr<2>(cosp1p2) + 7.5 * powr<4>(cosp1p2) +
          powr<2>(cosl1p1) * (15.5 - powr<2>(cosp1p2)) +
          cosl1p1 * cosl1p2 * cosp1p2 * (17. + 2. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p2) * (13. + 16. * powr<2>(cosp1p2) - 6.5 * powr<4>(cosp1p2))) *
             powr<2>(p2) +
         l1 * (powr<3>(cosl1p1) * ((36. + 12. * powr<2>(cosp1p2)) * p1 - cosp1p2 * p2) +
               powr<2>(cosl1p1) * cosl1p2 *
                   (26. * cosp1p2 * p1 - 25. * powr<3>(cosp1p2) * p1 - 35. * p2 -
                    11. * powr<2>(cosp1p2) * p2) +
               cosl1p2 * ((48. + 1. * powr<2>(cosl1p2)) * cosp1p2 * p1 +
                          (-128. - 36. * powr<2>(cosl1p2)) * p2 +
                          (-47. - 12. * powr<2>(cosl1p2)) * powr<2>(cosp1p2) * p2 -
                          17. * powr<4>(cosp1p2) * p2) +
               cosl1p1 *
                   ((128. + 47. * powr<2>(cosp1p2) + 17. * powr<4>(cosp1p2) +
                     powr<2>(cosl1p2) * (35. + 11. * powr<2>(cosp1p2))) *
                        p1 +
                    cosp1p2 * (-48. + powr<2>(cosl1p2) * (-26. + 25. * powr<2>(cosp1p2))) * p2))) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + 1. * powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(-0.5 *
                     RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) -
                                        2. * cosl1p2 * l1 * p2 - 2. * cosp1p2 * p1 * p2 +
                                        powr<2>(p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (-0.5 * powr<2>(l1) - cosl1p1 * l1 * p1 - 0.5 * powr<2>(p1) +
                  1. * cosl1p2 * l1 * p2 + 1. * cosp1p2 * p1 * p2 - 0.5 * powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) -
                             2. * cosl1p2 * l1 * p2 - 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        powr<2>(ZA4SP(0.7071067811865475 *
                      sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 -
                           cosp1p2 * p1 * p2 + powr<2>(p2)))) -
    12. * powr<-1>(11. + 6. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
        ((4. + powr<4>(cosl1p1) + powr<4>(cosl1p2) - 2. * powr<3>(cosl1p1) * cosl1p2 * cosp1p2 +
          4. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-5. - 2. * powr<2>(cosp1p2)) +
          cosl1p1 * cosl1p2 * cosp1p2 * (4. - 2. * powr<2>(cosl1p2) + 2. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p1) *
              (-5. - 2. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (1. + powr<2>(cosp1p2)))) *
             powr<6>(l1) +
         powr<4>(l1) *
             ((16. + 8. * powr<2>(cosp1p2) +
               cosl1p1 * cosl1p2 * cosp1p2 * (41. - 4. * powr<2>(cosl1p2) - 7. * powr<2>(cosp1p2)) +
               powr<4>(cosl1p1) * (-6. - 5. * powr<2>(cosp1p2)) +
               powr<2>(cosl1p2) * (-16. - 3. * powr<2>(cosp1p2)) +
               powr<3>(cosl1p1) * cosl1p2 * cosp1p2 * (-11. + 5. * powr<2>(cosp1p2)) +
               powr<2>(cosl1p1) * (-10. + 15. * powr<2>(cosp1p2) +
                                   powr<2>(cosl1p2) * (-32. + 9. * powr<2>(cosp1p2)))) *
                  powr<2>(p1) +
              (powr<5>(cosl1p1) * cosl1p2 +
               powr<4>(cosl1p1) * (7. - 2. * powr<2>(cosl1p2)) * cosp1p2 +
               cosp1p2 * (-35. - 11. * powr<4>(cosl1p2) + 7. * powr<2>(cosp1p2) +
                          powr<2>(cosl1p2) * (100. - 36. * powr<2>(cosp1p2))) +
               powr<3>(cosl1p1) * cosl1p2 *
                   (6. - 33. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (1. + powr<2>(cosp1p2))) +
               powr<2>(cosl1p1) * cosp1p2 *
                   (-56. - 2. * powr<4>(cosl1p2) - 4. * powr<2>(cosp1p2) +
                    powr<2>(cosl1p2) * (1. + 13. * powr<2>(cosp1p2))) +
               cosl1p1 * cosl1p2 *
                   (29. + powr<4>(cosl1p2) + 58. * powr<2>(cosp1p2) + 17. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (-84. + 21. * powr<2>(cosp1p2)))) *
                  p1 * p2 +
              (15. + 2. * powr<6>(cosl1p2) + powr<4>(cosl1p1) * (1. + 2. * powr<2>(cosl1p2)) +
               powr<3>(cosl1p1) * cosl1p2 * (16. - 4. * powr<2>(cosl1p2)) * cosp1p2 -
               28. * powr<2>(cosp1p2) + 9. * powr<4>(cosp1p2) +
               powr<4>(cosl1p2) * (-1. + powr<2>(cosp1p2)) +
               powr<2>(cosl1p2) * (-16. - 2. * powr<2>(cosp1p2) + 4. * powr<4>(cosp1p2)) +
               cosl1p1 * cosl1p2 * cosp1p2 *
                   (-69. - 4. * powr<4>(cosl1p2) + 13. * powr<2>(cosp1p2) +
                    powr<2>(cosl1p2) * (35. - powr<2>(cosp1p2))) +
               powr<2>(cosl1p1) * (6. + powr<2>(cosl1p2) * (59. - 42. * powr<2>(cosp1p2)) +
                                   powr<4>(cosl1p2) * (2. + 2. * powr<2>(cosp1p2)))) *
                  powr<2>(p2)) +
         powr<5>(l1) *
             (-powr<5>(cosl1p1) * p1 + powr<4>(cosl1p1) * cosl1p2 * (2. * cosp1p2 * p1 - 3. * p2) +
              powr<3>(cosl1p1) *
                  ((13. + 9. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-1. - powr<2>(cosp1p2))) *
                       p1 +
                   (-6. + 6. * powr<2>(cosl1p2)) * cosp1p2 * p2) +
              powr<2>(cosl1p1) * cosl1p2 *
                  ((3. + 2. * powr<2>(cosl1p2)) * cosp1p2 * p1 - 9. * powr<3>(cosp1p2) * p1 +
                   (-8. - 3. * powr<2>(cosl1p2)) * p2 +
                   (20. - 3. * powr<2>(cosl1p2)) * powr<2>(cosp1p2) * p2) +
              cosl1p2 *
                  ((-27. + 4. * powr<2>(cosl1p2)) * cosp1p2 * p1 + 9. * powr<3>(cosp1p2) * p1 +
                   (-10. + 13. * powr<2>(cosl1p2) - 3. * powr<4>(cosl1p2)) * p2 +
                   (-10. + 5. * powr<2>(cosl1p2)) * powr<2>(cosp1p2) * p2 -
                   2. * powr<4>(cosp1p2) * p2) +
              cosl1p1 *
                  ((-12. - powr<4>(cosl1p2) - 22. * powr<2>(cosp1p2) +
                    powr<2>(cosl1p2) * (36. - 4. * powr<2>(cosp1p2))) *
                       p1 +
                   cosp1p2 *
                       (33. + 6. * powr<4>(cosl1p2) - 7. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (-23. - 5. * powr<2>(cosp1p2))) *
                       p2)) +
         powr<3>(p2) *
             (16. * powr<2>(p1) * p2 - 16. * powr<2>(cosl1p2) * powr<2>(p1) * p2 +
              5. * powr<6>(cosp1p2) * powr<2>(p1) * p2 + 12. * powr<3>(p2) -
              12. * powr<2>(cosl1p2) * powr<3>(p2) +
              powr<3>(cosp1p2) *
                  (4. * powr<3>(p1) + (-18. - 6. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
              powr<5>(cosp1p2) *
                  (5. * powr<3>(p1) + (9. - 4. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
              cosp1p2 * ((-9. + 24. * powr<2>(cosl1p2)) * powr<3>(p1) +
                         (9. + 2. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
              powr<4>(cosp1p2) * ((16. - 4. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                                  (4. - 4. * powr<2>(cosl1p2)) * powr<3>(p2)) +
              powr<2>(cosp1p2) * ((-37. + 48. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                                  (-16. + 4. * powr<2>(cosl1p2)) * powr<3>(p2)) +
              cosl1p1 * cosl1p2 *
                  ((-15. - 4. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2)) * powr<3>(p1) +
                   cosp1p2 * (-11. - 12. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       p2 +
                   (-11. + 24. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2)) * p1 * powr<2>(p2) +
                   12. * cosp1p2 * powr<3>(p2))) +
         powr<2>(l1) * p2 *
             ((32. - 24. * powr<2>(cosl1p2) - 8. * powr<4>(cosl1p2)) * powr<2>(p1) * p2 +
              (14. + 12. * powr<2>(cosl1p2)) * powr<5>(cosp1p2) * p1 * powr<2>(p2) +
              (23. + 15. * powr<2>(cosl1p2) - 38. * powr<4>(cosl1p2)) * powr<3>(p2) +
              powr<3>(cosp1p2) * p1 *
                  ((-7. - 10. * powr<2>(cosl1p2)) * powr<2>(p1) +
                   (-31. - 54. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * powr<2>(p2)) +
              powr<2>(cosp1p2) * p2 *
                  ((-97. + 75. * powr<2>(cosl1p2) + 2. * powr<4>(cosl1p2)) * powr<2>(p1) +
                   (-48. - 96. * powr<2>(cosl1p2) + 11. * powr<4>(cosl1p2)) * powr<2>(p2)) +
              cosp1p2 * p1 *
                  ((-41. + 37. * powr<2>(cosl1p2)) * powr<2>(p1) +
                   (-31. - 18. * powr<2>(cosl1p2) + 60. * powr<4>(cosl1p2)) * powr<2>(p2)) +
              powr<4>(cosp1p2) *
                  ((-7. - 11. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                   (13. + 18. * powr<2>(cosl1p2) - 8. * powr<4>(cosl1p2)) * powr<3>(p2)) +
              powr<3>(cosl1p1) * cosl1p2 *
                  ((-2. - 5. * powr<2>(cosp1p2)) * powr<3>(p1) +
                   cosp1p2 * (-16. - 10. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                   (14. + 10. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + 6. * cosp1p2 * powr<3>(p2)) +
              cosl1p1 * cosl1p2 *
                  ((21. - 17. * powr<2>(cosl1p2) + 26. * powr<2>(cosp1p2) + 5. * powr<4>(cosp1p2)) *
                       powr<3>(p1) +
                   cosp1p2 *
                       (-43. + 41. * powr<2>(cosp1p2) + 10. * powr<4>(cosp1p2) +
                        powr<2>(cosl1p2) * (43. - powr<2>(cosp1p2))) *
                       powr<2>(p1) * p2 +
                   (97. - 21. * powr<2>(cosp1p2) + 22. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (-108. + 13. * powr<2>(cosp1p2) - 13. * powr<4>(cosp1p2))) *
                       p1 * powr<2>(p2) +
                   cosp1p2 *
                       (11. - powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (69. - 29. * powr<2>(cosp1p2))) *
                       powr<3>(p2)) +
              powr<2>(cosl1p1) *
                  ((52. - 32. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                   (11. + 42. * powr<2>(cosl1p2)) * powr<3>(p2) +
                   powr<3>(cosp1p2) * ((-5. + 5. * powr<2>(cosl1p2)) * powr<3>(p1) +
                                       (-6. - 39. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                   cosp1p2 * ((-2. - 5. * powr<2>(cosl1p2)) * powr<3>(p1) +
                              (-4. + 82. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                   powr<2>(cosp1p2) * ((24. - 30. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                                       (2. - powr<2>(cosl1p2)) * powr<3>(p2)))) +
         l1 * powr<2>(p2) *
             (cosl1p1 *
                  ((-34. - 23. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (30. + 13. * powr<2>(cosp1p2) + 5. * powr<4>(cosp1p2))) *
                       powr<3>(p1) +
                   cosp1p2 *
                       (-22. - 50. * powr<2>(cosp1p2) - 10. * powr<4>(cosp1p2) +
                        powr<2>(cosl1p2) *
                            (-41. + 34. * powr<2>(cosp1p2) + 5. * powr<4>(cosp1p2))) *
                       powr<2>(p1) * p2 +
                   (-50. + 10. * powr<2>(cosp1p2) - 8. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (68. - 54. * powr<2>(cosp1p2) + 30. * powr<4>(cosp1p2))) *
                       p1 * powr<2>(p2) +
                   cosp1p2 *
                       (-3. - powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (-55. + 13. * powr<2>(cosp1p2))) *
                       powr<3>(p2)) +
              powr<2>(cosl1p1) * cosl1p2 *
                  (18. * powr<2>(p1) * p2 + 10. * powr<4>(cosp1p2) * powr<2>(p1) * p2 -
                   10. * powr<3>(p2) + cosp1p2 * (2. * powr<3>(p1) - 37. * p1 * powr<2>(p2)) +
                   powr<3>(cosp1p2) * (5. * powr<3>(p1) + p1 * powr<2>(p2)) +
                   powr<2>(cosp1p2) * (18. * powr<2>(p1) * p2 - 5. * powr<3>(p2))) +
              cosl1p2 * ((-36. + 36. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 -
                         5. * powr<6>(cosp1p2) * powr<2>(p1) * p2 +
                         (-45. + 45. * powr<2>(cosl1p2)) * powr<3>(p2) +
                         cosp1p2 * ((35. - 31. * powr<2>(cosl1p2)) * powr<3>(p1) +
                                    (19. - 37. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                         powr<5>(cosp1p2) * (-5. * powr<3>(p1) +
                                             (-27. + 4. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                         powr<3>(cosp1p2) * (8. * powr<3>(p1) +
                                             (64. + 17. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                         powr<2>(cosp1p2) * ((101. - 56. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                                             (80. - 12. * powr<2>(cosl1p2)) * powr<3>(p2)) +
                         powr<4>(cosp1p2) * ((-6. + 4. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                                             (-19. + 12. * powr<2>(cosl1p2)) * powr<3>(p2)))) +
         powr<3>(l1) *
             (powr<4>(cosl1p1) * cosl1p2 * p2 *
                  ((6. + 5. * powr<2>(cosp1p2)) * powr<2>(p1) - 7. * cosp1p2 * p1 * p2 -
                   powr<2>(p2)) +
              powr<3>(cosl1p1) *
                  ((2. + 5. * powr<2>(cosp1p2)) * powr<3>(p1) +
                   cosp1p2 *
                       (16. + 10. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (11. - 5. * powr<2>(cosp1p2))) *
                       powr<2>(p1) * p2 +
                   (-3. - powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-30. + 15. * powr<2>(cosp1p2))) *
                       p1 * powr<2>(p2) +
                   (-6. - 10. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) +
              cosl1p1 *
                  ((-2. + 17. * powr<2>(cosl1p2) - 12. * powr<2>(cosp1p2)) * powr<3>(p1) +
                   cosp1p2 *
                       (108. + 4. * powr<4>(cosl1p2) - 8. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (-75. + 3. * powr<2>(cosp1p2))) *
                       powr<2>(p1) * p2 +
                   (-57. + 44. * powr<2>(cosp1p2) - 19. * powr<4>(cosp1p2) +
                    powr<4>(cosl1p2) * (34. - 25. * powr<2>(cosp1p2)) +
                    powr<2>(cosl1p2) * (58. - 44. * powr<2>(cosp1p2) - 24. * powr<4>(cosp1p2))) *
                       p1 * powr<2>(p2) +
                   cosp1p2 *
                       (30. - 8. * powr<2>(cosp1p2) +
                        powr<4>(cosl1p2) * (-12. + 6. * powr<2>(cosp1p2)) +
                        powr<2>(cosl1p2) * (-32. + 18. * powr<2>(cosp1p2))) *
                       powr<3>(p2)) +
              powr<2>(cosl1p1) * cosl1p2 *
                  ((-28. + 32. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 -
                   10. * powr<4>(cosp1p2) * powr<2>(p1) * p2 +
                   (-50. - 45. * powr<2>(cosl1p2)) * powr<3>(p2) +
                   cosp1p2 * (5. * powr<3>(p1) + (17. - 6. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                   powr<3>(cosp1p2) *
                       (-5. * powr<3>(p1) + (36. + 8. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                   powr<2>(cosp1p2) * ((-27. - 9. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                                       (7. + 21. * powr<2>(cosl1p2)) * powr<3>(p2))) +
              cosl1p2 *
                  ((-36. + 36. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 -
                   8. * powr<5>(cosp1p2) * p1 * powr<2>(p2) +
                   (-45. + 54. * powr<2>(cosl1p2) - 9. * powr<4>(cosl1p2)) * powr<3>(p2) +
                   powr<3>(cosp1p2) *
                       (5. * powr<3>(p1) + (13. + 34. * powr<2>(cosl1p2)) * p1 * powr<2>(p2)) +
                   cosp1p2 * (-15. * powr<3>(p1) +
                              (83. - 125. * powr<2>(cosl1p2) + 7. * powr<4>(cosl1p2)) * p1 *
                                  powr<2>(p2)) +
                   powr<4>(cosp1p2) *
                       (8. * powr<2>(p1) * p2 + (-31. + 4. * powr<2>(cosl1p2)) * powr<3>(p2)) +
                   powr<2>(cosp1p2) *
                       ((-48. + 7. * powr<2>(cosl1p2)) * powr<2>(p1) * p2 +
                        (110. + 5. * powr<2>(cosl1p2) - 6. * powr<4>(cosl1p2)) * powr<3>(p2))))) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
            sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                 (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                  (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                  2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
            atan2(-1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) *
                      powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                  -1. + 3. * l1 * (l1 - cosl1p2 * p2) *
                            powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                     l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
            0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                       (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                        powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
            atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                  2. - 3. * powr<2>(p1) *
                           powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                    2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        ZA4SP(0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 +
                                        powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2))) -
    6. * powr<-1>(11. + 6. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<2>(l1) *
        (2. * powr<4>(cosl1p1) * powr<2>(l1) + 2. * powr<4>(cosl1p2) * powr<2>(l1) +
         cosl1p2 * (2. - 2. * powr<2>(cosp1p2)) * l1 * p2 +
         cosp1p2 * (-1. + powr<2>(cosp1p2)) * p1 * p2 +
         powr<3>(cosl1p1) * l1 *
             (-4. * cosl1p2 * cosp1p2 * l1 - 2. * p1 + 2. * powr<2>(cosp1p2) * p1 -
              2. * cosp1p2 * p2) +
         powr<3>(cosl1p2) * l1 * (-2. * cosp1p2 * p1 - 2. * p2 + 2. * powr<2>(cosp1p2) * p2) +
         powr<2>(cosl1p2) * (-2. * powr<2>(l1) + cosp1p2 * (3. - 2. * powr<2>(cosp1p2)) * p1 * p2) +
         powr<2>(cosl1p1) * ((-2. + powr<2>(cosl1p2) * (2. + 2. * powr<2>(cosp1p2))) * powr<2>(l1) +
                             cosp1p2 * (3. - 2. * powr<2>(cosp1p2)) * p1 * p2 +
                             cosl1p2 * powr<2>(cosp1p2) * l1 * (-2. * cosp1p2 * p1 + 4. * p2)) +
         cosl1p1 *
             (-4. * powr<3>(cosl1p2) * cosp1p2 * powr<2>(l1) +
              (2. - 2. * powr<2>(cosp1p2)) * l1 * p1 +
              powr<2>(cosl1p2) * powr<2>(cosp1p2) * l1 * (4. * p1 - 2. * cosp1p2 * p2) +
              cosl1p2 * (4. * cosp1p2 * powr<2>(l1) - 2. * p1 * p2 -
                         2. * powr<2>(cosp1p2) * p1 * p2 + 2. * powr<4>(cosp1p2) * p1 * p2))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                    (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                    2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
              atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) *
                        powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    -1. + 3. * l1 * (l1 - cosl1p1 * p1) *
                              powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                   (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                    (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                    2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
              atan2(1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) *
                        powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                    -1. + 3. * l1 * (l1 - cosl1p2 * p2) *
                              powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                         (3. * powr<2>(p2) * powr<2>(-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) +
                          powr<2>(powr<2>(p2) -
                                  2. * (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                                        p1 * (p1 + cosp1p2 * p2))))),
              atan2(-1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
                    -1. + 3. *
                              powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                              (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                               p1 * (p1 + cosp1p2 * p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                         (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                          powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
              atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                    2. - 3. * powr<2>(p1) *
                             powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                      2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        (dtZc(k) * RB(powr<2>(k), powr<2>(l1)) + RBdot(powr<2>(k), powr<2>(l1)) * Zc(k) +
         RB(powr<2>(k), powr<2>(l1)) * (-50. * Zc(k) + 50. * Zc(1.02 * k))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * Zc(k) + powr<2>(l1) * Zc(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) * Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     Zc(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) * Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)))));
      // clang-format on
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(-2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)));
      const auto _interp9 = RB(powr<2>(k), fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp10 = ZA(sqrt(fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))));
      const auto _interp11 =
        RB(powr<2>(k), fma(2., cosp1p2 * p1 * p2, fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))));
      const auto _interp12 =
        ZA(sqrt(fma(2., cosp1p2 * p1 * p2, fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp16 =
        ZA3(0.816496580927726 * sqrt(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))),
            sqrt(fma(-2., cosl1p2 * powr<3>(l1) * p2,
                     fma(-1. + 4. * powr<2>(cosl1p2), powr<2>(l1) * powr<2>(p2),
                         fma(-2., cosl1p2 * l1 * powr<3>(p2), powr<4>(l1) + powr<4>(p2)))) *
                 powr<-2>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))),
            atan2(-1.732050807568877 * p2 * fma(-2., cosl1p2 * l1, p2) * powr<-1>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))),
                  fma(3., l1 * (l1 - cosl1p2 * p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                      -1.))); // clang-format off

const auto _interp20 = ZA3(0.816496580927726 * sqrt(fma(cosp1p2, p1 *p2,
                                 fma(-1., l1 *(cosl1p1 *p1 + 2. * cosl1p2 * p2),
                                     powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
    0.5 * sqrt(fma(3., powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2),
                   powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                           4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2))) *
               powr<-2>(fma(cosp1p2, p1 *p2,
                            fma(-1., l1 *(cosl1p1 *p1 + 2. * cosl1p2 * p2),
                                powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
    atan2(-1.732050807568877 * p1 * fma(-2., cosl1p1 *l1, fma(2., cosp1p2 *p2, p1)) *
              powr<-1>(fma(cosp1p2, p1 *p2,
                           fma(-1., l1 *(cosl1p1 *p1 + 2. * cosl1p2 * p2),
                               powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
          fma(-3.,
              powr<2>(p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                     2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
              2.)));
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _interp22 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                   (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 + (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                    2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
              atan2(1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                    -1. + 3. * l1 * (l1 - cosl1p2 * p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2))));
        const auto _interp23 = ZA4SP(0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)));
        const auto _cse1 = powr<2>(cosl1p2);
        const auto _cse2 = powr<2>(cosp1p2);
        const auto _cse3 = powr<2>(cosl1p1);
        const auto _cse4 = powr<2>(_cse1);
        const auto _cse5 = -2. * _cse2;
        const auto _cse6 = powr<2>(_cse3);
        const auto _cse7 = powr<3>(cosl1p1);
        const auto _cse8 = -2. * _cse1;
        const auto _cse9 = powr<2>(_cse2);
        const auto _cse10 = 1. * _cse2;
        const auto _cse11 = 1. + _cse10;
        const auto _cse12 = powr<2>(p2);
        const auto _cse13 = powr<5>(cosl1p2);
        const auto _cse14 = -2. * _cse3;
        const auto _cse15 = 36. + _cse14;
        const auto _cse16 = 4. * _cse2;
        const auto _cse17 = powr<3>(cosl1p2);
        const auto _cse18 = -_cse2;
        const auto _cse19 = powr<2>(l1);
        const auto _cse20 = 2. * _cse2;
        const auto _cse21 = -2. * _cse9;
        const auto _cse22 = 54. * _cse2;
        const auto _cse23 = 2. * _cse9;
        const auto _cse24 = 1. * _cse9;
        const auto _cse25 = -2. * cosl1p2 * l1 * p2;
        const auto _cse26 = _cse12 + _cse19 + _cse25;
        const auto _cse27 = powr<6>(k);
        const auto _cse28 = 1. * _cse27;
        const auto _cse29 = 1. + _cse28; // clang-format off
_acc += 12. * powr<-2>(_cse26) * powr<-1>(1. + _cse27) * _interp16 * _interp22
    * _interp23 *powr<-1>(fma(6., _cse2, 11. + _cse9)) *
    fma(powr<3>(_cse12), -17. - 18. * _cse2 + _cse1 * (17. + 18. * _cse2 + _cse24) - _cse9,
        fma(powr<2>(_cse12),
            _cse19 *
                (-49. - 43. * _cse2 + _cse21 + (5. + _cse18 + _cse1 * (9. + 3. * _cse2)) * _cse3 +
                 (81. + 36. * _cse2 + _cse24) * _cse4 + _cse1 * (-32. + 21. * _cse2 + 3. * _cse9) +
                 (-64. + 36. * _cse1 - 4. * _cse2) * cosl1p1 * cosl1p2 * cosp1p2),
            fma(_cse12,
                powr<2>(_cse19) *
                    (-36. + 1. * powr<3>(_cse1) - 28. * _cse2 +
                     _cse1 * (-33. + 79. * _cse2 + _cse23) + (68. - 11. * _cse2) * _cse4 +
                     _cse3 * (5. + _cse18 + _cse1 * (34. - 5. * _cse2) + _cse11 * _cse4) +
                     1. * _cse1 * _cse6 + _cse7 * (4. + _cse8) * cosl1p2 * cosp1p2 +
                     (-94. + _cse1 * (16. + _cse20) - 2. * _cse4 + _cse5) * cosl1p1 * cosl1p2 *
                         cosp1p2),
                fma(powr<3>(_cse19),
                    -4. + (-3. + _cse1 * _cse11) * _cse3 + 1. * _cse4 + _cse1 * (3. + _cse5) +
                        1. * _cse6 - 2. * _cse7 * cosl1p2 * cosp1p2 +
                        (6. + _cse8) * cosl1p1 * cosl1p2 * cosp1p2,
                    fma(-2. * _cse13 + _cse17 * (-34. + 10. * _cse2 + _cse3 * (-2. + _cse5)) +
                            (36. - 20. * _cse2 + (-8. + _cse16) * _cse3 - 2. * _cse6) * cosl1p2 +
                            _cse15 * cosl1p1 * cosp1p2 + 4. * _cse4 * cosl1p1 * cosp1p2 +
                            _cse1 * (-20. + 4. * _cse3 + _cse5) * cosl1p1 * cosp1p2,
                        powr<5>(l1) * p2,
                        fma(-26. * _cse13 +
                                _cse17 * (-100. - 96. * _cse2 - 18. * _cse3 - 4. * _cse9) +
                                (126. + _cse22 - 24. * _cse3) * cosl1p2 +
                                _cse1 * _cse15 * cosl1p1 * cosp1p2 +
                                (48. + _cse16) * cosl1p1 * cosp1p2 +
                                2. * _cse4 * cosl1p1 * powr<3>(cosp1p2),
                            powr<3>(l1) * powr<3>(p2),
                            fma(_cse17 * (-68. - 54. * _cse2 + _cse21) +
                                    (68. + _cse22 + _cse23) * cosl1p2 +
                                    (18. + _cse20) * cosl1p1 * cosp1p2 +
                                    _cse1 * (-18. + _cse5) * cosl1p1 * cosp1p2,
                                l1 *powr<5>(p2), 0.))))))) *
    powr<-2>(fma(_cse26, _interp10, fma(_interp4, _interp9, 0.))) *
    fma(_cse29, _interp1 *_interp2,
        fma(_cse29, _interp3 *_interp4,
            fma(_cse27, _interp2 * (-50. * _interp4 + 50. * _interp5), 0.))) *
    powr<-2>(fma(_interp2, _interp4, fma(_cse19, _interp6, 0.)));
        // clang-format on
      }
      { // subkernel 2
        const auto _interp24 =
          RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * (cosl1p2 * l1 + cosp1p2 * p1) * p2 + powr<2>(p2));
        const auto _interp25 =
          ZA(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * (cosl1p2 * l1 + cosp1p2 * p1) * p2 + powr<2>(p2)));
        const auto _interp26 = ZA4SP(
          0.7071067811865475 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - (cosl1p2 * l1 + cosp1p2 * p1) * p2 + powr<2>(p2)));
        const auto _cse1 = powr<2>(cosp1p2);
        const auto _cse2 = powr<2>(_cse1);
        const auto _cse3 = powr<2>(cosl1p2);
        const auto _cse4 = 13. * _cse1;
        const auto _cse5 = powr<2>(l1);
        const auto _cse6 = powr<2>(cosl1p1);
        const auto _cse7 = powr<2>(p1);
        const auto _cse8 = 2. * _cse1;
        const auto _cse9 = 35.5 * _cse1;
        const auto _cse10 = 7.5 * _cse2;
        const auto _cse11 = -_cse1;
        const auto _cse12 = 15.5 + _cse11;
        const auto _cse13 = 17. + _cse8;
        const auto _cse14 = _cse13 * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse15 = 16. * _cse1;
        const auto _cse16 = -6.5 * _cse2;
        const auto _cse17 = 13. + _cse15 + _cse16;
        const auto _cse18 = powr<2>(p2);
        const auto _cse19 = powr<3>(cosl1p1);
        const auto _cse20 = powr<3>(cosp1p2);
        const auto _cse21 = powr<6>(k);
        const auto _cse22 = 1. * _cse21;
        const auto _cse23 = 1. + _cse22;
        const auto _cse24 = -0.5 * _cse5;
        const auto _cse25 = -cosl1p1 * l1 * p1;
        const auto _cse26 = -0.5 * _cse7;
        const auto _cse27 = 1. * cosl1p2 * l1 * p2;
        const auto _cse28 = 1. * cosp1p2 * p1 * p2;
        const auto _cse29 = -0.5 * _cse18;
        const auto _cse30 = _cse24 + _cse25 + _cse26 + _cse27 + _cse28 + _cse29; // clang-format off
_acc += 0.375 * powr<-1>(1. + _cse21) * powr<-1>(_cse30) * powr<2>(_interp26) *
    powr<-1>(fma(6., _cse1, 11. + _cse2)) *
    fma(_cse18, 69. + _cse10 + _cse14 + _cse17 * _cse3 + _cse12 * _cse6 + _cse9,
        fma(_cse7, 69. + _cse10 + _cse14 + _cse12 * _cse3 + _cse17 * _cse6 + _cse9,
            fma(_cse5,
                47. + 10. * _cse1 + 15. * _cse2 + 1. * powr<2>(_cse3) + _cse3 * (34. + _cse4) +
                    (34. + (1. + 1. * _cse1) * _cse3 + _cse4) * _cse6 + 1. * powr<2>(_cse6) -
                    2. * _cse19 * cosl1p2 * cosp1p2 +
                    (28. - 26. * _cse1 - 2. * _cse3) * cosl1p1 * cosl1p2 * cosp1p2,
                fma((6. - 10. * _cse1 + 10. * _cse2) * cosl1p1 * cosl1p2 +
                        _cse6 * (2. * _cse20 - 61. * cosp1p2) +
                        (-157. - 51. * _cse1 - 16. * _cse2 + _cse3 * (-61. + _cse8)) * cosp1p2,
                    p1 *p2,
                    fma(l1,
                        _cse6 * cosl1p2 *
                                (-25. * _cse20 * p1 + 26. * cosp1p2 * p1 - 35. * p2 -
                                 11. * _cse1 * p2) +
                            cosl1p2 *
                                ((48. + 1. * _cse3) * cosp1p2 * p1 - 17. * _cse2 * p2 +
                                 (-128. - 36. * _cse3) * p2 + _cse1 * (-47. - 12. * _cse3) * p2) +
                            _cse19 * ((36. + 12. * _cse1) * p1 - cosp1p2 * p2) +
                            cosl1p1 *
                                ((128. + 47. * _cse1 + 17. * _cse2 + (35. + 11. * _cse1) * _cse3) *
                                     p1 +
                                 (-48. + (-26. + 25. * _cse1) * _cse3) * cosp1p2 * p2),
                        0.))))) *
    fma(_cse23, _interp1 *_interp2,
        fma(_cse23, _interp3 *_interp4,
            fma(_cse21, _interp2 * (-50. * _interp4 + 50. * _interp5), 0.))) *
    powr<-1>(fma(_cse30, _interp25, fma(-0.5, _interp24 *_interp4, 0.))) *
    powr<-2>(fma(_interp2, _interp4, fma(_cse5, _interp6, 0.)));
        // clang-format on
      }
      { // subkernel 3
        const auto _interp29 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 + (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                      2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
                atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                      -1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1))));
        const auto _interp30 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 + (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                      2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
                atan2(1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                      -1. + 3. * l1 * (l1 - cosl1p2 * p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))); // clang-format off

const auto _interp32 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
      0.5 *
          sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                        l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
               (3. * powr<2>(p2) * powr<2>(-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) +
                powr<2>(powr<2>(p2) - 2. * (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                                            p1 * (p1 + cosp1p2 * p2))))),
      atan2(-1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) *
                powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                         l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
            -1. + 3. *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                      (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                       p1 * (p1 + cosp1p2 * p2))));
        // clang-format on
        // clang-format off

const auto _interp34 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
      0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                          l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                 (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                  powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                          4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
      atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                         l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
            2. - 3. * powr<2>(p1) *
                     powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                              2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2))));
        // clang-format on

        const auto _interp35 = dtZc(k);
        const auto _interp36 = Zc(k);
        const auto _interp37 = Zc(1.02 * k);
        const auto _interp38 = Zc(l1);
        const auto _interp39 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
        const auto _interp40 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)));
        const auto _interp41 =
          Zc(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(cosp1p2);
        const auto _cse3 = -2. * _cse2;
        const auto _cse4 = powr<2>(cosl1p2);
        const auto _cse5 = 3. + _cse3;
        const auto _cse6 = _cse5 * cosp1p2 * p1 * p2;
        const auto _cse7 = -2. * cosp1p2 * p1;
        const auto _cse8 = powr<3>(cosl1p2);
        const auto _cse9 = 2. + _cse3;
        const auto _cse10 = -2. * cosp1p2 * p2;
        const auto _cse11 = powr<2>(_cse2);
        const auto _cse12 = -2. * cosl1p1 * l1 * p1;
        const auto _cse13 = powr<2>(p1);
        const auto _cse14 = -2. * cosl1p2 * l1 * p2;
        const auto _cse15 = powr<2>(p2); // clang-format off
_acc += - 6. *
    _cse1 * _interp29 * _interp30 * _interp32 *
    _interp34 *
        fma(2., _cse1 *powr<2>(_cse4),
            fma(_cse4, -2. * _cse1 + _cse6,
                fma(2., _cse1 *powr<4>(cosl1p1),
                    fma(powr<3>(cosl1p1),
                        l1 *(_cse10 - 4. * cosl1p2 * cosp1p2 * l1 - 2. * p1 + 2. * _cse2 * p1),
                        fma(_cse9, cosl1p2 * l1 * p2,
                            fma(-1. + _cse2, cosp1p2 * p1 * p2,
                                fma(_cse8, l1 *(_cse7 - 2. * p2 + 2. * _cse2 * p2),
                                    fma(powr<2>(cosl1p1),
                                        _cse1 * (-2. + (2. + 2. * _cse2) * _cse4) + _cse6 +
                                            _cse2 * cosl1p2 * l1 * (_cse7 + 4. * p2),
                                        fma(cosl1p1,
                                            -4. * _cse1 * _cse8 * cosp1p2 + _cse9 * l1 * p1 +
                                                _cse2 * _cse4 * l1 * (_cse10 + 4. * p1) +
                                                cosl1p2 *
                                                    (4. * _cse1 * cosp1p2 - 2. * p1 * p2 +
                                                     2. * _cse11 * p1 * p2 - 2. * _cse2 * p1 * p2),
                                            0.))))))))) *
    powr<-1>(fma(6., _cse2, 11. + _cse11)) *
    powr<-1>(fma(_cse1 + _cse12 + _cse13, _interp39, fma(_interp36, _interp7, 0.))) *
    powr<-1>(fma(_cse1 + _cse14 + _cse15, _interp40, fma(_interp36, _interp9, 0.))) *
    powr<-1>(fma(_interp11, _interp36,
                 fma(_interp41, _cse1 + _cse12 + _cse13 + _cse14 + _cse15 + 2. * cosp1p2 * p1 * p2,
                     0.))) *
    fma(_interp2, _interp35,
        fma(_interp3, _interp36, fma(_interp2, -50. * _interp36 + 50. * _interp37, 0.))) *
    powr<-2>(fma(_interp2, _interp36, fma(_cse1, _interp38, 0.)));
        // clang-format on
      }
      { // subkernel 4
        const auto _interp27 = ZA4SP(
          0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = -2. * cosl1p2 * l1 * p2;
        const auto _cse3 = powr<2>(p2);
        const auto _cse4 = powr<2>(cosp1p2);
        const auto _cse5 = powr<2>(cosl1p2);
        const auto _cse6 = -2. * _cse4;
        const auto _cse9 = powr<2>(cosl1p1);
        const auto _cse7 = powr<2>(_cse9);
        const auto _cse8 = powr<3>(cosl1p1);
        const auto _cse10 = powr<2>(p1);
        const auto _cse11 = -2. * _cse5;
        const auto _cse12 = powr<2>(_cse5);
        const auto _cse13 = 1. + _cse4;
        const auto _cse14 = _cse13 * _cse5;
        const auto _cse15 = powr<2>(_cse4);
        const auto _cse16 = -4. * _cse5;
        const auto _cse17 = 13. * _cse4;
        const auto _cse18 = 2. * _cse4;
        const auto _cse19 = powr<5>(cosl1p1);
        const auto _cse20 = 9. * _cse4;
        const auto _cse21 = -_cse4;
        const auto _cse22 = 2. * _cse5;
        const auto _cse23 = -3. * _cse5;
        const auto _cse24 = powr<3>(cosp1p2);
        const auto _cse25 = -4. * _cse4;
        const auto _cse26 = -7. * _cse4;
        const auto _cse27 = -5. * _cse4;
        const auto _cse28 = powr<3>(p2);
        const auto _cse29 = powr<3>(p1);
        const auto _cse30 = 16. + _cse16;
        const auto _cse31 = 4. * _cse5;
        const auto _cse32 = -5. * _cse15;
        const auto _cse33 = powr<5>(cosp1p2);
        const auto _cse34 = -8. * _cse12;
        const auto _cse35 = 5. * _cse5;
        const auto _cse36 = 5. * _cse15;
        const auto _cse37 = 10. * _cse4;
        const auto _cse38 = 5. * _cse29;
        const auto _cse39 = 18. * _cse10 * p2;
        const auto _cse40 = powr<3>(_cse4);
        const auto _cse41 = -27. + _cse31;
        const auto _cse42 = 12. * _cse5;
        const auto _cse43 = 5. * _cse4;
        const auto _cse44 = 15. * _cse4;
        const auto _cse45 = -10. * _cse5;
        const auto _cse46 = 17. * _cse5;
        const auto _cse47 = -12. * _cse4;
        const auto _cse48 = -8. * _cse4;
        const auto _cse49 = 6. * _cse4;
        const auto _cse50 = -6. * _cse5; // clang-format off
_acc += - 12. *
    powr<-1>(_cse1 + _cse2 + _cse3) * powr<-1>(11. + _cse15 + _cse49) * _interp16 * _interp20 *
    _interp27 *powr<-1>(1. + powr<6>(k)) *
    powr<-1>(fma(-2., cosl1p1 * l1 * p1,
                 fma(2., cosp1p2 * p1 * p2, _cse1 + _cse10 + _cse2 + _cse3))) *
    fma(powr<3>(_cse1),
        4. + _cse12 + 4. * _cse4 + _cse5 * (-5. + _cse6) + _cse7 + (-5. + _cse14 + _cse6) * _cse9 -
            2. * _cse8 * cosl1p2 * cosp1p2 + (4. + _cse11 + _cse18) * cosl1p1 * cosl1p2 * cosp1p2,
        fma(powr<2>(_cse1),
            _cse10 * (16. + 8. * _cse4 + (-16. - 3. * _cse4) * _cse5 + (-6. + _cse27) * _cse7 +
                      (-10. + _cse44 + (-32. + _cse20) * _cse5) * _cse9 +
                      (-11. + _cse43) * _cse8 * cosl1p2 * cosp1p2 +
                      (41. + _cse16 + _cse26) * cosl1p1 * cosl1p2 * cosp1p2) +
                _cse3 * (15. + 9. * _cse15 + _cse12 * (-1. + _cse4) - 28. * _cse4 +
                         2. * powr<3>(_cse5) + _cse5 * (-16. + 4. * _cse15 + _cse6) +
                         (1. + _cse22) * _cse7 +
                         (6. + _cse12 * (2. + _cse18) + (59. - 42. * _cse4) * _cse5) * _cse9 +
                         _cse30 * _cse8 * cosl1p2 * cosp1p2 +
                         (-69. - 4. * _cse12 + _cse17 + (35. + _cse21) * _cse5) * cosl1p1 *
                             cosl1p2 * cosp1p2) +
                (_cse19 * cosl1p2 + (6. + _cse14 - 33. * _cse4) * _cse8 * cosl1p2 +
                 (29. + _cse12 + 17. * _cse15 + 58. * _cse4 + (-84. + 21. * _cse4) * _cse5) *
                     cosl1p1 * cosl1p2 +
                 (-35. - 11. * _cse12 + 7. * _cse4 + (100. - 36. * _cse4) * _cse5) * cosp1p2 +
                 (7. + _cse11) * _cse7 * cosp1p2 +
                 (-56. - 2. * _cse12 + _cse25 + (1. + _cse17) * _cse5) * _cse9 * cosp1p2) *
                    p1 * p2,
            fma(_cse28,
                12. * _cse28 - 12. * _cse28 * _cse5 +
                    _cse33 * (_cse38 + (9. + _cse16) * _cse3 * p1) +
                    cosp1p2 * (_cse29 * (-9. + 24. * _cse5) + (9. + _cse22) * _cse3 * p1) +
                    _cse24 * (4. * _cse29 + _cse3 * (-18. + _cse50) * p1) + 16. * _cse10 * p2 +
                    5. * _cse10 * _cse40 * p2 - 16. * _cse10 * _cse5 * p2 +
                    _cse15 * ((4. + _cse16) * _cse28 + _cse10 * _cse30 * p2) +
                    _cse4 * (_cse28 * (-16. + _cse31) + _cse10 * (-37. + 48. * _cse5) * p2) +
                    cosl1p1 * cosl1p2 *
                        (_cse29 * (-15. + _cse25 + _cse32) + 12. * _cse28 * cosp1p2 +
                         _cse3 * (-11. + _cse32 + 24. * _cse4) * p1 +
                         _cse10 * (-11. + _cse32 + _cse47) * cosp1p2 * p2),
                fma(powr<5>(l1),
                    -_cse19 *p1 + _cse7 * cosl1p2 * (2. * cosp1p2 * p1 - 3. * p2) +
                        _cse9 * cosl1p2 *
                            (-9. * _cse24 * p1 + (3. + _cse22) * cosp1p2 * p1 +
                             (-8. + _cse23) * p2 + (20. + _cse23) * _cse4 * p2) +
                        cosl1p2 * (9. * _cse24 * p1 + _cse41 * cosp1p2 * p1 - 2. * _cse15 * p2 +
                                   (-10. + _cse35) * _cse4 * p2 +
                                   (-10. - 3. * _cse12 + 13. * _cse5) * p2) +
                        _cse8 * ((13. + _cse20 + (-1. + _cse21) * _cse5) * p1 +
                                 (-6. + 6. * _cse5) * cosp1p2 * p2) +
                        cosl1p1 *
                            ((-12. - _cse12 - 22. * _cse4 + (36. + _cse25) * _cse5) * p1 +
                             (33. + 6. * _cse12 + _cse26 + (-23. + _cse27) * _cse5) * cosp1p2 * p2),
                    fma(_cse3,
                        l1 *(_cse9 * cosl1p2 *
                                 (-10. * _cse28 + _cse39 + (-5. * _cse28 + _cse39) * _cse4 +
                                  cosp1p2 * (2. * _cse29 - 37. * _cse3 * p1) +
                                  _cse24 * (_cse38 + _cse3 * p1) + 10. * _cse10 * _cse15 * p2) +
                             cosl1p1 *
                                 (_cse29 * (-34. + _cse32 - 23. * _cse4 +
                                            (30. + _cse17 + _cse36) * _cse5) +
                                  _cse28 * (-3. + _cse21 + (-55. + _cse17) * _cse5) * cosp1p2 +
                                  _cse3 *
                                      (-50. - 8. * _cse15 + _cse37 +
                                       (68. + 30. * _cse15 - 54. * _cse4) * _cse5) *
                                      p1 +
                                  _cse10 *
                                      (-22. - 10. * _cse15 - 50. * _cse4 +
                                       (-41. + _cse36 + 34. * _cse4) * _cse5) *
                                      cosp1p2 * p2) +
                             cosl1p2 *
                                 (_cse28 * (-45. + 45. * _cse5) +
                                  _cse33 * (-5. * _cse29 + _cse3 * _cse41 * p1) +
                                  _cse24 * (8. * _cse29 + _cse3 * (64. + _cse46) * p1) +
                                  cosp1p2 * (_cse29 * (35. - 31. * _cse5) +
                                             _cse3 * (19. - 37. * _cse5) * p1) -
                                  5. * _cse10 * _cse40 * p2 + _cse10 * (-36. + 36. * _cse5) * p2 +
                                  _cse15 *
                                      (_cse28 * (-19. + _cse42) + _cse10 * (-6. + _cse31) * p2) +
                                  _cse4 * (_cse28 * (80. - 12. * _cse5) +
                                           _cse10 * (101. - 56. * _cse5) * p2))),
                        fma(_cse1,
                            p2 *(_cse28 * (23. - 38. * _cse12 + 15. * _cse5) +
                                 _cse3 * _cse33 * (14. + _cse42) * p1 +
                                 _cse24 *
                                     (_cse10 * (-7. + _cse45) +
                                      _cse3 * (-31. + _cse12 - 54. * _cse5)) *
                                     p1 +
                                 (_cse3 * (-31. + 60. * _cse12 - 18. * _cse5) +
                                  _cse10 * (-41. + 37. * _cse5)) *
                                     cosp1p2 * p1 +
                                 _cse10 * (32. + _cse34 - 24. * _cse5) * p2 +
                                 _cse4 *
                                     (_cse3 * (-48. + 11. * _cse12 - 96. * _cse5) +
                                      _cse10 * (-97. + 2. * _cse12 + 75. * _cse5)) *
                                     p2 +
                                 _cse15 * (_cse28 * (13. + _cse34 + 18. * _cse5) +
                                           _cse10 * (-7. - 11. * _cse5) * p2) +
                                 _cse8 * cosl1p2 *
                                     ((-2. + _cse27) * _cse29 + 6. * _cse28 * cosp1p2 +
                                      _cse3 * (14. + _cse37) * p1 +
                                      _cse10 * (-16. - 10. * _cse4) * cosp1p2 * p2) +
                                 cosl1p1 * cosl1p2 *
                                     (_cse29 * (21. + _cse36 + 26. * _cse4 - 17. * _cse5) +
                                      _cse28 * (11. + _cse21 + (69. - 29. * _cse4) * _cse5) *
                                          cosp1p2 +
                                      _cse3 *
                                          (97. + 22. * _cse15 - 21. * _cse4 +
                                           (-108. - 13. * _cse15 + _cse17) * _cse5) *
                                          p1 +
                                      _cse10 *
                                          (-43. + 10. * _cse15 + 41. * _cse4 +
                                           (43. + _cse21) * _cse5) *
                                          cosp1p2 * p2) +
                                 _cse9 * (_cse28 * (11. + 42. * _cse5) +
                                          _cse24 * (_cse29 * (-5. + _cse35) +
                                                    _cse3 * (-6. - 39. * _cse5) * p1) +
                                          cosp1p2 * (_cse29 * (-2. - 5. * _cse5) +
                                                     _cse3 * (-4. + 82. * _cse5) * p1) +
                                          _cse10 * (52. - 32. * _cse5) * p2 +
                                          _cse4 * (_cse28 * (2. - _cse5) +
                                                   _cse10 * (24. - 30. * _cse5) * p2))),
                            fma(powr<3>(l1),
                                _cse8 *(_cse29 * (2. + _cse43) + _cse28 * (-6. + _cse45) * cosp1p2 +
                                        _cse3 * (-3. + _cse21 + (-30. + _cse44) * _cse5) * p1 +
                                        _cse10 * (16. + _cse37 + (11. + _cse27) * _cse5) * cosp1p2 *
                                            p2) +
                                    cosl1p1 * (_cse29 * (-2. + _cse46 + _cse47) +
                                               _cse28 *
                                                   (30. + _cse48 + _cse12 * (-12. + _cse49) +
                                                    (-32. + 18. * _cse4) * _cse5) *
                                                   cosp1p2 +
                                               _cse3 *
                                                   (-57. - 19. * _cse15 +
                                                    _cse12 * (34. - 25. * _cse4) + 44. * _cse4 +
                                                    (58. - 24. * _cse15 - 44. * _cse4) * _cse5) *
                                                   p1 +
                                               _cse10 *
                                                   (108. + 4. * _cse12 + _cse48 +
                                                    (-75. + 3. * _cse4) * _cse5) *
                                                   cosp1p2 * p2) +
                                    _cse7 * cosl1p2 * p2 *
                                        (-_cse3 + _cse10 * (6. + _cse43) - 7. * cosp1p2 * p1 * p2) +
                                    _cse9 * cosl1p2 *
                                        (_cse28 * (-50. - 45. * _cse5) +
                                         _cse24 * (-5. * _cse29 + _cse3 * (36. + 8. * _cse5) * p1) +
                                         cosp1p2 * (_cse38 + _cse3 * (17. + _cse50) * p1) -
                                         10. * _cse10 * _cse15 * p2 +
                                         _cse10 * (-28. + 32. * _cse5) * p2 +
                                         _cse4 * (_cse28 * (7. + 21. * _cse5) +
                                                  _cse10 * (-27. - 9. * _cse5) * p2)) +
                                    cosl1p2 *
                                        (_cse28 * (-45. - 9. * _cse12 + 54. * _cse5) -
                                         8. * _cse3 * _cse33 * p1 +
                                         cosp1p2 *
                                             (-15. * _cse29 +
                                              _cse3 * (83. + 7. * _cse12 - 125. * _cse5) * p1) +
                                         _cse24 * (_cse38 + _cse3 * (13. + 34. * _cse5) * p1) +
                                         _cse10 * (-36. + 36. * _cse5) * p2 +
                                         _cse15 * (_cse28 * (-31. + _cse31) + 8. * _cse10 * p2) +
                                         _cse4 * (_cse28 * (110. - 6. * _cse12 + _cse35) +
                                                  _cse10 * (-48. + 7. * _cse5) * p2)),
                                0.))))))) *
    powr<-1>(fma(_cse1 + _cse2 + _cse3, _interp10, fma(_interp4, _interp9, 0.))) *
    powr<-1>(fma(_interp11, _interp4,
                 fma(_interp12,
                     _cse1 + _cse10 + _cse2 + _cse3 - 2. * cosl1p1 * l1 * p1 +
                         2. * cosp1p2 * p1 * p2,
                     0.))) *
    powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.))) *
    fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k),
        fma(_interp3, _interp4 * (1. + powr<6>(k)),
            fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.)));
        // clang-format on
      }
      { // subkernel 5
        const auto _interp8 = ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
        const auto _interp14 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 + (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                    2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
              atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    -1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))); // clang-format off

const auto _interp18 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                             l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
    0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                        l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
               (3. * powr<2>(p2) * powr<2>(-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) +
                powr<2>(powr<2>(p2) - 2. * (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                                            p1 * (p1 + cosp1p2 * p2))))),
    atan2(1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) *
              powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                       l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
          -1. + 3. *
                    powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                             l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                    (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                     p1 * (p1 + cosp1p2 * p2))));
        // clang-format on

        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = -2. * cosl1p1 * l1 * p1;
        const auto _cse3 = powr<2>(p1);
        const auto _cse4 = -2. * cosl1p2 * l1 * p2;
        const auto _cse5 = powr<2>(p2);
        const auto _cse6 = powr<4>(_cse1);
        const auto _cse7 = powr<3>(_cse1);
        const auto _cse8 = powr<3>(cosp1p2);
        const auto _cse9 = powr<3>(p1);
        const auto _cse10 = powr<2>(_cse1);
        const auto _cse11 = powr<5>(p1);
        const auto _cse12 = powr<2>(cosp1p2);
        const auto _cse13 = powr<2>(_cse12);
        const auto _cse14 = powr<2>(_cse3);
        const auto _cse15 = powr<3>(_cse3);
        const auto _cse16 = powr<3>(p2);
        const auto _cse17 = powr<5>(cosp1p2);
        const auto _cse18 = powr<7>(p1);
        const auto _cse19 = powr<2>(_cse5);
        const auto _cse20 = powr<3>(_cse12);
        const auto _cse21 = powr<5>(p2);
        const auto _cse22 = powr<3>(_cse5);
        const auto _cse23 = powr<7>(p2);
        const auto _cse24 = -4. * _cse12;
        const auto _cse25 = 14. + _cse24;
        const auto _cse26 = powr<2>(cosl1p2);
        const auto _cse27 = powr<5>(l1);
        const auto _cse28 = 10. * _cse12;
        const auto _cse29 = powr<3>(l1);
        const auto _cse30 = 26. * _cse12;
        const auto _cse31 = 4. * _cse26;
        const auto _cse32 = -10. * _cse12;
        const auto _cse33 = 4. * _cse12;
        const auto _cse34 = -34. + _cse28;
        const auto _cse35 = 20. * _cse12;
        const auto _cse36 = -8. * _cse26;
        const auto _cse37 = powr<7>(l1);
        const auto _cse38 = -52. * _cse26;
        const auto _cse39 = 104. * _cse26;
        const auto _cse40 = powr<2>(_cse26);
        const auto _cse41 = -8. * _cse40;
        const auto _cse42 = 8. * _cse12;
        const auto _cse43 = -20. * _cse12;
        const auto _cse44 = -4. * _cse26;
        const auto _cse45 = 2. * _cse26;
        const auto _cse46 = -2. * _cse26;
        const auto _cse47 = 26. * _cse26;
        const auto _cse48 = 6. * _cse7; // clang-format off
_acc += 48. * powr<-1>(_cse1 + _cse2 + _cse3) * powr<-1>(_cse1 + _cse4 + _cse5) * _interp14 * _interp16 * _interp18 * _interp20 * powr<-1>(1. + powr<6>(k)) * fma(-4., _cse10 * _cse15, fma(8., _cse1 * _cse11 * _cse16 * _cse17, fma(2., _cse16 * _cse17 * _cse18, fma(-18., _cse1 * _cse14 * _cse19, fma(30., _cse1 * _cse12 * _cse14 * _cse19, fma(-20., _cse1 * _cse13 * _cse14 * _cse19, fma(12., _cse12 * _cse15 * _cse19, fma(-16., _cse13 * _cse15 * _cse19, fma(6., _cse1 * _cse14 * _cse19 * _cse20, fma(4., _cse15 * _cse19 * _cse20, fma(4., _cse11 * _cse17 * _cse21, fma(-4., _cse10 * _cse22, fma(12., _cse12 * _cse14 * _cse22, fma(-16., _cse13 * _cse14 * _cse22, fma(4., _cse14 * _cse20 * _cse22, fma(-30., _cse10 * _cse19 * _cse3, fma(_cse10, _cse12 * _cse19 * _cse3, fma(13., _cse10 * _cse13 * _cse19 * _cse3, fma(-4., _cse1 * _cse22 * _cse3, fma(-30., _cse10 * _cse14 * _cse5, fma(_cse10, _cse12 * _cse14 * _cse5, fma(13., _cse10 * _cse13 * _cse14 * _cse5, fma(-4., _cse1 * _cse15 * _cse5, fma(-8., _cse3 * _cse6, fma(-8., _cse5 * _cse6, fma(-12., _cse14 * _cse7, fma(-12., _cse19 * _cse7, fma(-34., _cse3 * _cse5 * _cse7, fma(-6., _cse12 * _cse3 * _cse5 * _cse7, fma(12., _cse13 * _cse3 * _cse5 * _cse7, fma(-22., _cse1 * _cse11 * _cse16 * _cse8, fma(-8., _cse16 * _cse18 * _cse8, fma(-16., _cse11 * _cse21 * _cse8, fma(13., _cse10 * _cse16 * _cse17 * _cse9, fma(8., _cse1 * _cse17 * _cse21 * _cse9, fma(2., _cse17 * _cse23 * _cse9, fma(-8., _cse10 * _cse16 * _cse8 * _cse9, fma(-22., _cse1 * _cse21 * _cse8 * _cse9, fma(-8., _cse23 * _cse8 * _cse9, fma(-4., _cse16 * _cse37 * powr<7>(cosl1p2), fma(6., _cse1 * _cse11 * _cse16 * cosp1p2, fma(6., _cse16 * _cse18 * cosp1p2, fma(10., _cse11 * _cse21 * cosp1p2, fma(-21., _cse10 * _cse16 * _cse9 * cosp1p2, fma(6., _cse1 * _cse21 * _cse9 * cosp1p2, fma(6., _cse23 * _cse9 * cosp1p2, fma(2., _cse11 * _cse21 * powr<7>(cosp1p2), fma(2., _cse10 * _cse21 * _cse8 * p1, fma(5., _cse16 * _cse7 * _cse8 * p1, fma(-10., _cse10 * _cse21 * cosp1p2 * p1, fma(-17., _cse16 * _cse7 * cosp1p2 * p1, fma(2., _cse10 * _cse11 * _cse8 * p2, fma(5., _cse7 * _cse8 * _cse9 * p2, fma(-10., _cse10 * _cse11 * cosp1p2 * p2, fma(-17., _cse7 * _cse9 * cosp1p2 * p2, fma(3., _cse6 * _cse8 * p1 * p2, fma(-3., _cse6 * cosp1p2 * p1 * p2, fma(_cse29, powr<3>(cosl1p2) * (_cse13 * _cse16 * _cse3 * (20. * _cse1 + 28. * _cse3 + 2. * _cse5) - 8. * _cse17 * _cse19 * _cse9 + _cse5 * (36. * _cse10 + 28. * _cse14 + 12. * _cse19 + 65. * _cse3 * _cse5 + _cse1 * (76. * _cse3 + 40. * _cse5)) * _cse8 * p1 + (-38. * _cse22 - 125. * _cse19 * _cse3 + _cse10 * (-10. * _cse3 - 36. * _cse5) - 68. * _cse14 * _cse5 + _cse1 * (-4. * _cse14 - 118. * _cse19 - 131. * _cse3 * _cse5) - 6. * _cse7) * cosp1p2 * p1 + _cse12 * (4. * _cse15 + 2. * _cse22 + 9. * _cse19 * _cse3 + _cse48 + _cse10 * (14. * _cse3 - 12. * _cse5) - 21. * _cse14 * _cse5 + _cse1 * (12. * _cse14 - 21. * _cse3 * _cse5)) * p2 + (-8. * _cse15 - 6. * _cse22 - 32. * _cse19 * _cse3 - 52. * _cse14 * _cse5 + _cse10 * (-12. * _cse3 + 12. * _cse5) + _cse1 * (-32. * _cse14 - 4. * _cse19 - 60. * _cse3 * _cse5) + 18. * _cse7) * p2), fma(cosl1p2, l1 * (_cse14 * _cse16 * _cse20 * (-6. * _cse1 - 4. * _cse3 - 12. * _cse5) + _cse17 * _cse5 * (-4. * _cse10 - 2. * _cse14 - 12. * _cse19 + _cse1 * (-4. * _cse3 - 29. * _cse5) - 14. * _cse3 * _cse5) * _cse9 - 2. * _cse11 * _cse19 * powr<7>(cosp1p2) + (58. * _cse14 * _cse19 + 46. * _cse22 * _cse3 + 10. * _cse15 * _cse5 + _cse1 * (2. * _cse15 - 4. * _cse22 + 72. * _cse19 * _cse3 + 28. * _cse14 * _cse5) + _cse10 * (6. * _cse14 - 26. * _cse19 - _cse3 * _cse5) + (4. * _cse3 - 24. * _cse5) * _cse7) * _cse8 * p1 + (-36. * _cse14 * _cse19 - 30. * _cse22 * _cse3 - 12. * _cse15 * _cse5 + _cse1 * (-6. * _cse15 + 8. * _cse22 - 29. * _cse19 * _cse3 - 24. * _cse14 * _cse5) + _cse10 * (-18. * _cse14 + 42. * _cse19 + _cse3 * _cse5) + (-12. * _cse3 + 16. * _cse5) * _cse7) * cosp1p2 * p1 + _cse13 * _cse3 * (-2. * _cse22 + 32. * _cse19 * _cse3 + _cse10 * (6. * _cse3 - 35. * _cse5) + 18. * _cse14 * _cse5 + _cse1 * (4. * _cse14 - 13. * _cse19 + 3. * _cse3 * _cse5)) * p2 + _cse1 * (8. * _cse15 + 6. * _cse22 + 32. * _cse19 * _cse3 + _cse48 + 52. * _cse14 * _cse5 + _cse10 * (42. * _cse3 + 46. * _cse5) + _cse1 * (40. * _cse14 + 38. * _cse19 + 102. * _cse3 * _cse5)) * p2 + _cse12 * (-26. * _cse14 * _cse19 + 6. * _cse22 * _cse3 - 18. * _cse15 * _cse5 + _cse1 * (-12. * _cse15 - 2. * _cse22 + 5. * _cse19 * _cse3 - 39. * _cse14 * _cse5) + _cse10 * (-22. * _cse14 - 10. * _cse19 - 7. * _cse3 * _cse5) - 6. * _cse6 + (-10. * _cse3 - 14. * _cse5) * _cse7) * p2), fma(_cse7, _cse9 * powr<7>(cosl1p1) * (-4. * l1 + 4. * cosl1p2 * p2), fma(powr<3>(_cse26), _cse5 * _cse7 * (26. * _cse1 + 20. * _cse3 + _cse25 * _cse5 + 10. * cosp1p2 * p1 * p2), fma(_cse27, powr<5>(cosl1p2) * p2 * (-24. * _cse10 - 8. * _cse14 + _cse19 * _cse34 + _cse3 * (-42. + _cse35) * _cse5 + _cse16 * (-32. + _cse28) * cosp1p2 * p1 - 38. * _cse9 * cosp1p2 * p2 + _cse1 * (-30. * _cse3 + (-54. + _cse30) * _cse5 - 52. * cosp1p2 * p1 * p2)), fma(_cse27, _cse3 * powr<6>(cosl1p1) * (26. * _cse29 + _cse1 * cosl1p2 * (8. * cosp1p2 * p1 - 52. * p2) + cosl1p2 * p2 * (_cse3 * (-14. + _cse33) - 4. * _cse5 - 10. * cosp1p2 * p1 * p2) + l1 * (_cse25 * _cse3 + (20. + 10. * _cse26) * _cse5 + (10. + _cse36) * cosp1p2 * p1 * p2)), fma(_cse10, powr<5>(cosl1p1) * p1 * (-24. * _cse27 + _cse10 * cosl1p2 * (-52. * cosp1p2 * p1 + 72. * p2) + _cse1 * cosl1p2 * (_cse16 * (40. + _cse31) + 4. * _cse8 * _cse9 - 20. * _cse9 * cosp1p2 + (60. - 20. * _cse26) * _cse5 * cosp1p2 * p1 + _cse12 * _cse3 * (-72. + _cse31) * p2 + _cse3 * (112. + _cse31) * p2) + cosl1p2 * p1 * p2 * ((34. + _cse32) * _cse9 + 6. * _cse16 * cosp1p2 + (10. + _cse33) * _cse5 * p1 + _cse3 * (32. + _cse32) * cosp1p2 * p2) + l1 * (_cse14 * _cse34 + _cse19 * (-8. + _cse46) + _cse3 * (-42. + (-26. + 22. * _cse12) * _cse26 + _cse35) * _cse5 + _cse16 * (-38. + _cse36) * cosp1p2 * p1 + (-32. + (20. + _cse24) * _cse26 + _cse28) * _cse9 * cosp1p2 * p2) + _cse29 * (_cse3 * (-54. + (-4. + _cse24) * _cse26 + _cse30) + (-30. + _cse38) * _cse5 + (-52. + _cse39) * cosp1p2 * p1 * p2)), fma(_cse1, _cse26 * (-6. * _cse6 + _cse1 * ((4. - 2. * _cse12) * _cse15 + (-34. + 14. * _cse12 + 11. * _cse13) * _cse19 * _cse3 + _cse22 * (-22. + _cse42) + (-6. + 45. * _cse12 - 24. * _cse13) * _cse14 * _cse5 + (125. - 61. * _cse12 + 12. * _cse13) * _cse16 * _cse9 * cosp1p2 + (17. + _cse12) * _cse21 * cosp1p2 * p1 + _cse11 * (68. - 26. * _cse12) * cosp1p2 * p2) + _cse10 * ((8. - 6. * _cse12) * _cse14 + (-70. + 32. * _cse12) * _cse19 + (-44. + 30. * _cse12 - 8. * _cse13) * _cse3 * _cse5 + (8. + 25. * _cse12) * _cse16 * cosp1p2 * p1 + (65. - 32. * _cse12) * _cse9 * cosp1p2 * p2) + _cse7 * ((-2. + _cse24) * _cse3 + (-30. + 24. * _cse12) * _cse5 + (-3. - 6. * _cse12) * cosp1p2 * p1 * p2) + p1 * p2 * (_cse17 * (18. * _cse19 * _cse3 + 6. * _cse14 * _cse5) + (-4. * _cse15 - 2. * _cse22 - 80. * _cse19 * _cse3 - 58. * _cse14 * _cse5) * _cse8 + 18. * _cse16 * _cse9 + 8. * _cse16 * _cse20 * _cse9 + (12. * _cse15 + 6. * _cse22 + 78. * _cse19 * _cse3 + 68. * _cse14 * _cse5) * cosp1p2 + 4. * _cse21 * p1 + 4. * _cse11 * p2 + _cse13 * (-22. * _cse16 * _cse9 + 4. * _cse21 * p1 - 6. * _cse11 * p2) + _cse12 * (16. * _cse16 * _cse9 - 14. * _cse21 * p1 + 16. * _cse11 * p2))), fma(_cse10, _cse40 * (_cse48 + _cse1 * (4. * _cse14 + (68. - 28. * _cse12) * _cse19 + (58. - 22. * _cse12) * _cse3 * _cse5 + (107. - 52. * _cse12) * _cse16 * cosp1p2 * p1 + 40. * _cse9 * cosp1p2 * p2) + _cse10 * (10. * _cse3 + (12. - 24. * _cse12) * _cse5 + 36. * cosp1p2 * p1 * p2) + p2 * (26. * _cse21 + 64. * _cse16 * _cse3 - 4. * _cse13 * _cse16 * _cse3 + _cse8 * (-56. * _cse5 * _cse9 - 20. * _cse19 * p1) + cosp1p2 * (8. * _cse11 + 92. * _cse5 * _cse9 + 65. * _cse19 * p1) + 36. * _cse14 * p2 + _cse12 * (-8. * _cse21 - 15. * _cse16 * _cse3 - 2. * _cse14 * p2))), fma(powr<2>(cosl1p1), l1 * ((-6. + (6. + 6. * _cse12) * _cse26) * powr<9>(l1) + _cse6 * cosl1p2 * (-6. * _cse8 * p1 + (-48. + 48. * _cse26) * cosp1p2 * p1 + _cse12 * (12. - 24. * _cse26) * p2 + (24. - 24. * _cse26) * p2) + _cse29 * (_cse22 * (4. - 2. * _cse12 + (-4. + 2. * _cse12) * _cse26) + _cse19 * _cse3 * (-6. + 45. * _cse12 - 24. * _cse13 + (-222. - 81. * _cse12 + 82. * _cse13) * _cse26 + (-18. - 36. * _cse12 + 2. * _cse13) * _cse40) + _cse15 * (-22. + (-4. + 2. * _cse12) * _cse26 + _cse42) + _cse14 * (-34. + 14. * _cse12 + 11. * _cse13 + (-222. - 81. * _cse12 + 82. * _cse13) * _cse26 - 4. * _cse40) * _cse5 + _cse16 * (125. - 61. * _cse12 + 12. * _cse13 + (-15. - 167. * _cse12 + 54. * _cse13) * _cse26 + _cse40 * (55. + _cse43)) * _cse9 * cosp1p2 + _cse21 * (68. - 26. * _cse12 + _cse26 * (23. + _cse28) + (-40. + _cse28) * _cse40) * cosp1p2 * p1 + _cse11 * (17. + _cse12 + _cse26 * (23. + _cse28) + 2. * _cse40) * cosp1p2 * p2) + _cse27 * (_cse14 * (-70. + 32. * _cse12 + (6. - 38. * _cse12) * _cse26 + 2. * _cse40) + _cse19 * (8. - 6. * _cse12 + (6. - 38. * _cse12) * _cse26 + (-14. + 2. * _cse12) * _cse40) + _cse3 * (-44. + 30. * _cse12 - 8. * _cse13 + (-270. + 130. * _cse13) * _cse26 + 10. * powr<3>(_cse26) + (78. - 216. * _cse12) * _cse40) * _cse5 + _cse16 * (65. - 32. * _cse12 + (123. + 12. * _cse12) * _cse26 - 8. * powr<3>(_cse26) + (-114. + 62. * _cse12) * _cse40) * cosp1p2 * p1 + (8. + 25. * _cse12 + (123. + 12. * _cse12) * _cse26 - 22. * _cse40) * _cse9 * cosp1p2 * p2) + l1 * p1 * p2 * (_cse17 * _cse3 * _cse5 * (_cse3 * (18. + _cse36) + (6. + _cse36) * _cse5) + (_cse22 * (-4. + _cse45) + _cse15 * (-2. + _cse45) + _cse19 * _cse3 * (-58. + _cse47) + _cse14 * (-80. + _cse47) * _cse5) * _cse8 + _cse16 * (18. - 128. * _cse26) * _cse9 + _cse16 * _cse20 * (8. + _cse46) * _cse9 + (_cse15 * (6. - 6. * _cse26) + _cse22 * (12. - 6. * _cse26) + _cse19 * (68. - 36. * _cse26) * _cse3 + _cse14 * (78. - 36. * _cse26) * _cse5) * cosp1p2 + _cse21 * (4. - 60. * _cse26) * p1 + _cse11 * (4. - 60. * _cse26) * p2 + _cse13 * (_cse19 * (-6. + _cse44) + _cse14 * (4. + _cse44) + (-22. - 38. * _cse26) * _cse3 * _cse5) * p1 * p2 + _cse12 * (_cse14 * (-14. + 28. * _cse26) + _cse19 * (16. + 28. * _cse26) + (16. + 170. * _cse26) * _cse3 * _cse5) * p1 * p2) + _cse37 * (_cse3 * (-30. + 24. * _cse12 + (-24. - 34. * _cse12) * _cse26 + 26. * _cse40) + (-2. + _cse24 + (-24. - 34. * _cse12) * _cse26 + (26. + _cse30) * _cse40) * _cse5 + (-3. - 6. * _cse12 + (156. + 60. * _cse12) * _cse26 - 144. * _cse40) * cosp1p2 * p1 * p2) + _cse1 * cosl1p2 * p1 * (_cse17 * _cse3 * _cse5 * (-22. * _cse3 + (-40. + _cse31) * _cse5) + (-2. * _cse15 + _cse19 * (151. + 34. * _cse26) * _cse3 + _cse22 * (14. + _cse36) + _cse14 * (75. + _cse45) * _cse5) * _cse8 - 8. * _cse16 * _cse20 * _cse9 + _cse16 * (94. + 116. * _cse26) * _cse9 + (6. * _cse15 + _cse22 * (-52. + 28. * _cse26) + _cse19 * (-93. - 18. * _cse26) * _cse3 + _cse14 * (-71. - 15. * _cse26) * _cse5) * cosp1p2 + _cse21 * (60. + 86. * _cse26) * p1 + _cse11 * (48. + _cse31) * p2 + _cse12 * (_cse19 * (-27. + _cse46) + _cse14 * (24. + _cse46) + (-26. - 101. * _cse26) * _cse3 * _cse5) * p1 * p2 + _cse13 * (_cse16 * (-13. + 14. * _cse26) * _cse9 - 2. * _cse21 * p1 - 12. * _cse11 * p2)) + _cse3 * _cse5 * cosl1p2 * (12. * _cse21 + 36. * _cse16 * _cse3 + 2. * _cse16 * _cse20 * _cse3 + _cse8 * (2. * _cse11 - 32. * _cse5 * _cse9 - 18. * _cse19 * p1) + _cse17 * (12. * _cse5 * _cse9 + 4. * _cse19 * p1) + cosp1p2 * (-6. * _cse11 + 26. * _cse5 * _cse9 + 18. * _cse19 * p1) + 30. * _cse14 * p2 + _cse12 * (-10. * _cse21 - 58. * _cse16 * _cse3 - 46. * _cse14 * p2) + _cse13 * (2. * _cse21 + 14. * _cse16 * _cse3 + 12. * _cse14 * p2)) + _cse10 * cosl1p2 * (_cse21 * (-16. + 16. * _cse26) + _cse16 * _cse3 * (176. + 136. * _cse26 - 26. * _cse40) - 32. * _cse17 * _cse5 * _cse9 + (-2. * _cse14 + _cse19 * (74. - 68. * _cse26 - 4. * _cse40) + (38. + 52. * _cse26) * _cse3 * _cse5) * _cse8 * p1 + (_cse19 * (-192. + 100. * _cse26 + 20. * _cse40) + _cse14 * (-22. + _cse46) + _cse3 * (-75. - 92. * _cse26 + _cse41) * _cse5) * cosp1p2 * p1 + _cse14 * (204. - 16. * _cse26 - 2. * _cse40) * p2 + _cse13 * _cse3 * (-48. * _cse3 + (-32. - 104. * _cse26) * _cse5) * p2 + _cse12 * (_cse16 * _cse3 * (-63. + 222. * _cse26 + 22. * _cse40) + _cse21 * (16. + _cse44) + _cse14 * (-4. + 76. * _cse26) * p2)) + _cse7 * cosl1p2 * (_cse16 * (-4. + 8. * _cse26 - 4. * _cse40) + ((-52. + 14. * _cse26) * _cse3 + (-94. - 46. * _cse26 + 104. * _cse40) * _cse5) * cosp1p2 * p1 + _cse8 * (-14. * _cse9 + (34. - 124. * _cse26) * _cse5 * p1) - 36. * _cse13 * _cse3 * p2 + _cse3 * (144. + 20. * _cse26 - 52. * _cse40) * p2 + _cse12 * (_cse16 * (32. + 20. * _cse26 - 4. * _cse40) + (-98. + 180. * _cse26) * _cse3 * p2))), fma(_cse29, powr<4>(cosl1p1) * (6. * _cse37 + _cse7 * cosl1p2 * (48. * cosp1p2 * p1 - 24. * p2) + _cse10 * cosl1p2 * (_cse16 * (-20. + _cse44) - 26. * _cse8 * _cse9 + (56. + 8. * _cse26) * _cse9 * cosp1p2 + (-38. + _cse39) * _cse5 * cosp1p2 * p1 + _cse3 * (-62. + _cse38) * p2 + _cse12 * _cse3 * (176. + _cse38) * p2) + l1 * p1 * (_cse11 * (26. - 8. * _cse12) + (64. - 15. * _cse12 - 4. * _cse13 + (-18. - 36. * _cse12 + 2. * _cse13) * _cse26) * _cse5 * _cse9 + _cse16 * _cse3 * (92. - 56. * _cse12 + _cse26 * (55. + _cse43)) * cosp1p2 + _cse21 * (8. + _cse45) * cosp1p2 + _cse19 * (36. - 2. * _cse12 + _cse44) * p1 + _cse14 * (65. + _cse26 * (-40. + _cse28) + _cse43) * cosp1p2 * p2) + _cse3 * cosl1p2 * p2 * (-2. * _cse12 * _cse19 + _cse14 * (-26. + _cse42) + (8. - 25. * _cse12 + 12. * _cse13) * _cse3 * _cse5 + _cse16 * (-28. + _cse42) * cosp1p2 * p1 + (-65. + _cse35) * _cse9 * cosp1p2 * p2) + _cse29 * (_cse14 * (68. - 28. * _cse12 + (-14. + 2. * _cse12) * _cse26) + _cse19 * (4. + _cse45) + _cse3 * (58. - 22. * _cse12 + (78. - 216. * _cse12) * _cse26 + (10. + _cse28) * _cse40) * _cse5 + _cse16 * (40. - 22. * _cse26 + _cse41) * cosp1p2 * p1 + (107. - 52. * _cse12 + (-114. + 62. * _cse12) * _cse26 + _cse41) * _cse9 * cosp1p2 * p2) + _cse27 * (_cse3 * (12. - 24. * _cse12 + _cse26 * (26. + _cse30)) + (10. + _cse47) * _cse5 + (36. - 144. * _cse26) * cosp1p2 * p1 * p2) + _cse1 * cosl1p2 * p1 * ((-10. * _cse14 + (96. - 12. * _cse26) * _cse3 * _cse5) * _cse8 + (40. * _cse14 - 26. * _cse19 + _cse3 * (-194. + _cse47) * _cse5) * cosp1p2 + _cse16 * (-106. + _cse31) * p1 - 10. * _cse13 * _cse9 * p2 + (-122. + 14. * _cse26) * _cse9 * p2 + _cse12 * (_cse16 * (90. + 16. * _cse26) * p1 + (104. + _cse46) * _cse9 * p2))), fma(_cse1, powr<3>(cosl1p1) * (_cse37 * ((18. + 6. * _cse12) * p1 - 6. * cosp1p2 * p2) + l1 * p1 * ((-6. + 2. * _cse12) * _cse15 + (-52. - 21. * _cse12 + 28. * _cse13) * _cse19 * _cse3 + _cse22 * (-8. + _cse33) + (-32. + 9. * _cse12 + 2. * _cse13) * _cse14 * _cse5 + (-125. + 65. * _cse12 - 8. * _cse13) * _cse16 * _cse9 * cosp1p2 + (-68. + 28. * _cse12) * _cse21 * cosp1p2 * p1 + _cse11 * (-38. + 12. * _cse12) * cosp1p2 * p2) + _cse29 * (-4. * _cse11 + (-60. - 21. * _cse12 + 20. * _cse13) * _cse5 * _cse9 - 4. * _cse21 * cosp1p2 + (-131. + 76. * _cse12) * _cse16 * _cse3 * cosp1p2 + (-32. + 12. * _cse12) * _cse19 * p1 + (-118. + 40. * _cse12) * _cse14 * cosp1p2 * p2) + _cse27 * ((12. - 12. * _cse12) * _cse9 - 10. * _cse16 * cosp1p2 + (-12. + 14. * _cse12) * _cse5 * p1 + (-36. + 36. * _cse12) * _cse3 * cosp1p2 * p2) + _cse1 * powr<3>(cosl1p2) * (_cse3 * _cse5 * (144. * _cse1 + 4. * _cse3 + 4. * _cse5) * _cse8 + 4. * _cse13 * _cse16 * _cse9 + (-12. * _cse19 * _cse3 + _cse10 * (-52. * _cse3 - 52. * _cse5) - 12. * _cse14 * _cse5 - 52. * _cse1 * _cse3 * _cse5) * cosp1p2 + (72. * _cse10 - 16. * _cse14 - 16. * _cse19 + _cse1 * (-8. * _cse3 - 8. * _cse5) - 112. * _cse3 * _cse5) * p1 * p2 + _cse12 * (72. * _cse10 + 4. * _cse14 + 4. * _cse19 + _cse1 * (-56. * _cse3 - 56. * _cse5) - 10. * _cse3 * _cse5) * p1 * p2) + cosl1p2 * (_cse14 * _cse17 * (12. * _cse1 - 8. * _cse5) * _cse5 + _cse3 * (-2. * _cse22 - _cse19 * _cse3 + _cse10 * (32. * _cse3 - 120. * _cse5) - 12. * _cse14 * _cse5 + _cse1 * (8. * _cse14 - 108. * _cse19 - 106. * _cse3 * _cse5) + 24. * _cse7) * _cse8 + (7. * _cse14 * _cse19 + 8. * _cse22 * _cse3 + 38. * _cse15 * _cse5 + _cse1 * (-28. * _cse15 + 198. * _cse19 * _cse3 + 202. * _cse14 * _cse5) + _cse10 * (-36. * _cse14 + 12. * _cse19 + 164. * _cse3 * _cse5) - 12. * _cse6 + (24. * _cse3 + 4. * _cse5) * _cse7) * cosp1p2 + _cse13 * (52. * _cse10 - 10. * _cse19 + _cse1 * (20. * _cse3 - 4. * _cse5) - 22. * _cse3 * _cse5) * _cse9 * p2 + _cse12 * (-2. * _cse15 + 47. * _cse19 * _cse3 + _cse10 * (-96. * _cse3 - 110. * _cse5) + 69. * _cse14 * _cse5 + _cse1 * (-76. * _cse14 - 30. * _cse19 + 12. * _cse3 * _cse5) - 96. * _cse7) * p1 * p2 + (6. * _cse15 - 32. * _cse19 * _cse3 - 42. * _cse14 * _cse5 + _cse10 * (-44. * _cse3 + 52. * _cse5) + _cse1 * (-8. * _cse14 + 32. * _cse19 + 32. * _cse3 * _cse5) - 48. * _cse7) * p1 * p2) + _cse10 * powr<5>(cosl1p2) * p1 * p2 * (4. * _cse3 + (4. + _cse33) * _cse5 - 20. * cosp1p2 * p1 * p2) + _cse40 * (_cse27 * (-4. * _cse9 + 8. * _cse16 * cosp1p2 + (-52. - 52. * _cse12) * _cse5 * p1 + 104. * _cse3 * cosp1p2 * p2) + _cse29 * _cse5 * p1 * ((4. + 16. * _cse12) * _cse3 + (14. - 2. * _cse12) * _cse5 + (26. - 12. * _cse12) * cosp1p2 * p1 * p2)) + _cse26 * (_cse37 * ((-24. - 24. * _cse12) * p1 + 48. * cosp1p2 * p2) + _cse29 * (_cse11 * (16. + _cse24) + (136. + 222. * _cse12 - 104. * _cse13) * _cse5 * _cse9 - 2. * _cse21 * cosp1p2 + (-92. + 52. * _cse12) * _cse16 * _cse3 * cosp1p2 + (-16. + 76. * _cse12) * _cse19 * p1 + (100. - 68. * _cse12) * _cse14 * cosp1p2 * p2) + _cse27 * ((8. + _cse35) * _cse9 + 14. * _cse16 * cosp1p2 + (20. + 180. * _cse12) * _cse5 * p1 + (-46. - 124. * _cse12) * _cse3 * cosp1p2 * p2) + l1 * p1 * p2 * (4. * _cse21 + 116. * _cse16 * _cse3 + 14. * _cse13 * _cse16 * _cse3 + 4. * _cse17 * _cse5 * _cse9 + cosp1p2 * (28. * _cse11 - 18. * _cse5 * _cse9 - 15. * _cse19 * p1) + _cse8 * (-8. * _cse11 + 34. * _cse5 * _cse9 + 2. * _cse19 * p1) + 86. * _cse14 * p2 + _cse12 * (-2. * _cse21 - 101. * _cse16 * _cse3 - 2. * _cse14 * p2)))), fma(cosl1p1, 4. * _cse16 * _cse7 * powr<7>(cosl1p2) * p1 + (6. - 6. * _cse12) * powr<9>(l1) * p1 + _cse27 * (_cse11 * (38. + _cse32) + (102. - 7. * _cse12 - 35. * _cse13) * _cse5 * _cse9 + (-18. + 6. * _cse12) * _cse21 * cosp1p2 + (1. - _cse12 - 4. * _cse13) * _cse16 * _cse3 * cosp1p2 + (40. - 22. * _cse12 + 6. * _cse13) * _cse19 * p1 + (42. - 26. * _cse12) * _cse14 * cosp1p2 * p2) + _cse29 * ((6. - 2. * _cse12) * _cse18 + _cse11 * (32. + 5. * _cse12 - 13. * _cse13) * _cse5 + _cse19 * (52. - 39. * _cse12 + 3. * _cse13 - 6. * _cse20) * _cse9 + (-29. + 72. * _cse12 - 29. * _cse13) * _cse14 * _cse16 * cosp1p2 + (-6. + 2. * _cse12) * _cse23 * cosp1p2 + (-24. + 28. * _cse12 - 4. * _cse13) * _cse21 * _cse3 * cosp1p2 + (8. - 12. * _cse12 + 4. * _cse13) * _cse22 * p1 + _cse15 * (8. + _cse24) * cosp1p2 * p2) + _cse37 * ((46. - 14. * _cse12) * _cse9 + _cse16 * (-12. + _cse33) * cosp1p2 + (42. + _cse32) * _cse5 * p1 + (16. - 24. * _cse12) * _cse3 * cosp1p2 * p2) + _cse1 * powr<3>(cosl1p2) * (_cse17 * _cse19 * (12. * _cse1 - 8. * _cse3) * _cse3 + _cse5 * (-2. * _cse15 - 12. * _cse19 * _cse3 - _cse14 * _cse5 + _cse10 * (-120. * _cse3 + 32. * _cse5) + _cse1 * (-108. * _cse14 + 8. * _cse19 - 106. * _cse3 * _cse5) + 24. * _cse7) * _cse8 + (7. * _cse14 * _cse19 + 38. * _cse22 * _cse3 + 8. * _cse15 * _cse5 + _cse1 * (-28. * _cse22 + 202. * _cse19 * _cse3 + 198. * _cse14 * _cse5) + _cse10 * (12. * _cse14 - 36. * _cse19 + 164. * _cse3 * _cse5) - 12. * _cse6 + (4. * _cse3 + 24. * _cse5) * _cse7) * cosp1p2 + _cse13 * _cse16 * (52. * _cse10 - 10. * _cse14 - 22. * _cse3 * _cse5 + _cse1 * (-4. * _cse3 + 20. * _cse5)) * p1 + _cse12 * (-2. * _cse22 + 69. * _cse19 * _cse3 + _cse10 * (-110. * _cse3 - 96. * _cse5) + 47. * _cse14 * _cse5 + _cse1 * (-30. * _cse14 - 76. * _cse19 + 12. * _cse3 * _cse5) - 96. * _cse7) * p1 * p2 + (6. * _cse22 - 42. * _cse19 * _cse3 + _cse10 * (52. * _cse3 - 44. * _cse5) - 32. * _cse14 * _cse5 + _cse1 * (32. * _cse14 - 8. * _cse19 + 32. * _cse3 * _cse5) - 48. * _cse7) * p1 * p2) + cosl1p2 * (_cse17 * _cse3 * _cse5 * (-4. * _cse19 * _cse3 - 4. * _cse14 * _cse5 + _cse10 * (28. * _cse3 + 28. * _cse5) + _cse1 * (12. * _cse14 + 12. * _cse19 + 55. * _cse3 * _cse5) + 12. * _cse7) + (16. * _cse15 * _cse19 + 16. * _cse14 * _cse22 + _cse10 * (-8. * _cse15 - 8. * _cse22 - 50. * _cse19 * _cse3 - 50. * _cse14 * _cse5) + _cse1 * (-150. * _cse14 * _cse19 - 44. * _cse22 * _cse3 - 44. * _cse15 * _cse5) + (-12. * _cse14 - 12. * _cse19 + 64. * _cse3 * _cse5) * _cse7) * _cse8 + _cse16 * _cse20 * (6. * _cse10 - 2. * _cse3 * _cse5 + _cse1 * (12. * _cse3 + 12. * _cse5)) * _cse9 + (12. * powr<5>(_cse1) - 12. * _cse15 * _cse19 - 12. * _cse14 * _cse22 + _cse10 * (28. * _cse15 + 28. * _cse22 + 12. * _cse19 * _cse3 + 12. * _cse14 * _cse5) + _cse1 * (57. * _cse14 * _cse19 + 32. * _cse22 * _cse3 + 32. * _cse15 * _cse5) + (28. * _cse3 + 28. * _cse5) * _cse6 + (56. * _cse14 + 56. * _cse19) * _cse7) * cosp1p2 + 2. * _cse1 * _cse14 * _cse19 * powr<7>(cosp1p2) + (-10. * _cse14 * _cse19 - 6. * _cse22 * _cse3 - 6. * _cse15 * _cse5 + _cse1 * (-12. * _cse15 - 12. * _cse22 - 42. * _cse19 * _cse3 - 42. * _cse14 * _cse5) + _cse10 * (-98. * _cse14 - 98. * _cse19 - 238. * _cse3 * _cse5) - 54. * _cse6 + (-180. * _cse3 - 180. * _cse5) * _cse7) * p1 * p2 + _cse13 * (-4. * _cse14 * _cse19 - 2. * _cse22 * _cse3 - 2. * _cse15 * _cse5 + _cse1 * (2. * _cse15 + 2. * _cse22 - 19. * _cse19 * _cse3 - 19. * _cse14 * _cse5) + _cse10 * (2. * _cse14 + 2. * _cse19 + 68. * _cse3 * _cse5) + 6. * _cse6 + (14. * _cse3 + 14. * _cse5) * _cse7) * p1 * p2 + _cse12 * (16. * _cse14 * _cse19 + 8. * _cse22 * _cse3 + 8. * _cse15 * _cse5 + _cse1 * (-2. * _cse15 - 2. * _cse22 + 33. * _cse19 * _cse3 + 33. * _cse14 * _cse5) + _cse10 * (36. * _cse14 + 36. * _cse19 + 24. * _cse3 * _cse5) + 66. * _cse6 + (76. * _cse3 + 76. * _cse5) * _cse7) * p1 * p2) + _cse3 * _cse5 * cosp1p2 * l1 * (-12. * _cse21 - 36. * _cse16 * _cse3 - 2. * _cse16 * _cse20 * _cse3 + cosp1p2 * (6. * _cse11 - 26. * _cse5 * _cse9 - 18. * _cse19 * p1) + _cse17 * (-12. * _cse5 * _cse9 - 4. * _cse19 * p1) + _cse8 * (-2. * _cse11 + 32. * _cse5 * _cse9 + 18. * _cse19 * p1) - 30. * _cse14 * p2 + _cse13 * (-2. * _cse21 - 14. * _cse16 * _cse3 - 12. * _cse14 * p2) + _cse12 * (10. * _cse21 + 58. * _cse16 * _cse3 + 46. * _cse14 * p2)) + powr<3>(_cse26) * _cse27 * _cse5 * (_cse1 * (-52. * p1 + 8. * cosp1p2 * p2) + p1 * (-4. * _cse3 + (-14. + _cse33) * _cse5 - 10. * cosp1p2 * p1 * p2)) + _cse26 * (powr<9>(l1) * ((24. + 12. * _cse12) * p1 + (-48. - 6. * _cse12) * cosp1p2 * p2) + _cse27 * (_cse11 * (-16. + 16. * _cse12) + (176. - 63. * _cse12 - 32. * _cse13) * _cse5 * _cse9 + (-22. - 2. * _cse12) * _cse21 * cosp1p2 + (-75. + 38. * _cse12 - 32. * _cse13) * _cse16 * _cse3 * cosp1p2 + _cse19 * (204. - 48. * _cse13 + _cse24) * p1 + (-192. + 74. * _cse12) * _cse14 * cosp1p2 * p2) + _cse3 * _cse5 * l1 * (_cse11 * (12. + 2. * _cse13 + _cse32) + (36. - 58. * _cse12 + 14. * _cse13 + 2. * _cse20) * _cse5 * _cse9 + (-6. + 2. * _cse12) * _cse21 * cosp1p2 + (26. - 32. * _cse12 + 12. * _cse13) * _cse16 * _cse3 * cosp1p2 + (30. - 46. * _cse12 + 12. * _cse13) * _cse19 * p1 + (18. - 18. * _cse12 + 4. * _cse13) * _cse14 * cosp1p2 * p2) + _cse37 * ((-4. + 32. * _cse12) * _cse9 + (-52. - 14. * _cse12) * _cse16 * cosp1p2 + (144. - 98. * _cse12 - 36. * _cse13) * _cse5 * p1 + (-94. + 34. * _cse12) * _cse3 * cosp1p2 * p2) + _cse29 * p2 * (_cse17 * (-22. * _cse19 * _cse3 - 40. * _cse14 * _cse5) + (14. * _cse15 - 2. * _cse22 + 75. * _cse19 * _cse3 + 151. * _cse14 * _cse5) * _cse8 + 94. * _cse16 * _cse9 - 8. * _cse16 * _cse20 * _cse9 + (-52. * _cse15 + 6. * _cse22 - 71. * _cse19 * _cse3 - 93. * _cse14 * _cse5) * cosp1p2 + 48. * _cse21 * p1 + 60. * _cse11 * p2 + _cse12 * (-26. * _cse16 * _cse9 + 24. * _cse21 * p1 - 27. * _cse11 * p2) + _cse13 * (-13. * _cse16 * _cse9 - 12. * _cse21 * p1 - 2. * _cse11 * p2))) + _cse10 * powr<5>(cosl1p2) * p2 * (_cse10 * (72. * p1 - 52. * cosp1p2 * p2) + _cse1 * (40. * _cse9 + _cse16 * (-20. + _cse33) * cosp1p2 + (112. - 72. * _cse12) * _cse5 * p1 + 60. * _cse3 * cosp1p2 * p2) + p1 * p2 * (34. * _cse16 - 10. * _cse5 * _cse8 * p1 + cosp1p2 * (6. * _cse9 + 32. * _cse5 * p1) + 10. * _cse3 * p2 + _cse12 * (-10. * _cse16 + 4. * _cse3 * p2))) + _cse40 * (powr<9>(l1) * (-24. * p1 + 48. * cosp1p2 * p2) + _cse37 * (-20. * _cse9 + (56. - 26. * _cse12) * _cse16 * cosp1p2 + (-62. + 176. * _cse12) * _cse5 * p1 - 38. * _cse3 * cosp1p2 * p2) + _cse29 * _cse5 * p1 * (-26. * _cse19 + 8. * _cse3 * _cse5 + 12. * _cse13 * _cse3 * _cse5 + _cse12 * (-2. * _cse14 + 8. * _cse19 - 25. * _cse3 * _cse5) + cosp1p2 * (-65. * _cse16 * p1 - 28. * _cse9 * p2) + _cse8 * (20. * _cse16 * p1 + 8. * _cse9 * p2)) + _cse27 * p2 * ((-10. * _cse19 + 96. * _cse3 * _cse5) * _cse8 + (-26. * _cse14 + 40. * _cse19 - 194. * _cse3 * _cse5) * cosp1p2 - 122. * _cse16 * p1 - 10. * _cse13 * _cse16 * p1 - 106. * _cse9 * p2 + _cse12 * (104. * _cse16 * p1 + 90. * _cse9 * p2))), 0.)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) * powr<-1>(fma(2., cosp1p2 * p1 * p2, _cse1 + _cse2 + _cse3 + _cse4 + _cse5)) * powr<-1>(fma(6., _cse12, 11. + _cse13)) * powr<-1>(fma(_cse1 + _cse4 + _cse5, _interp10, fma(_interp4, _interp9, 0.))) * powr<-1>(fma(_interp11, _interp4, fma(_interp12, _cse1 + _cse2 + _cse3 + _cse4 + _cse5 + 2. * cosp1p2 * p1 * p2, 0.))) * powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.))) * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * powr<-1>(fma(_interp4, _interp7, fma(_cse1 + _cse2 + _cse3, _interp8, 0.)));
                                        // clang-format on
      }
      return _acc; // clang-format off

}

static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double &S0, const double &S1, const double &SPhi, const double &k,
                                                 const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
                                                 const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
                                                 const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
  using namespace DiFfRG;
                   // clang-format on

      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_FORCEINLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return Regulator::RBdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return Regulator::RFdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return Regulator::dq2RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return Regulator::dq2RF(k2, p2); }
  };
} // namespace DiFfRG
using DiFfRG::ZA4tadpole_kernel;