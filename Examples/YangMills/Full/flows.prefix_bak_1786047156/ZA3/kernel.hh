#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA3_kernel
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
      const double p2 = 0.7071067811865475 * sqrt(powr<2>(S0) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)));
      const double cosp1p2 = 0.7071067811865475 * powr<2>(S0) * (-1. - 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)) *
        sqrt(powr<-1>(-powr<4>(S0) * (-1. + S1 * sin(SPhi)) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi))));
      const double cosl1p1 = cos1;
      const double cosl1p2 = 1.414213562373095 * powr<-1>(l1) *
        sqrt(powr<-1>(powr<2>(S0) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)))) *
        (-0.5 * cos1 * l1 * sqrt(powr<2>(S0) * (1. - S1 * sin(SPhi))) -
         0.866025403784439 * l1 * sqrt(powr<2>(S0) * (1. + S1 * sin(SPhi))) *
           (1.414213562373095 * cos2 *
              sqrt((-1. + powr<2>(cos1)) * (-1. + powr<2>(S1)) * powr<-1>(2. - powr<2>(S1) + powr<2>(S1) * cos(2. * SPhi))) +
            cos1 * powr<2>(S0) * S1 * cos(SPhi) * sqrt(powr<-1>(powr<4>(S0) * (1. - powr<2>(S1) * powr<2>(sin(SPhi)))))));
      // clang-format off
using _T = decltype(18. * powr<-1>(-1. + powr<2>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
        powr<-1>(3. * powr<4>(p1) + 6. * cosp1p2 * powr<3>(p1) * p2 +
                 (8. + powr<2>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) +
                 6. * cosp1p2 * p1 * powr<3>(p2) + 3. * powr<4>(p2)) *
        (powr<5>(cosl1p2) * powr<5>(l1) * (-cosp1p2 * p1 - p2) * powr<4>(p2) +
         powr<6>(cosl1p1) * powr<4>(l1) * powr<5>(p1) * (-2. * p1 - 2. * cosp1p2 * p2) +
         powr<5>(cosl1p1) * powr<3>(l1) * powr<4>(p1) *
             ((15. - 2. * powr<2>(cosp1p2)) * powr<3>(p1) +
              cosp1p2 * (28. - 2. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
              (14. + powr<2>(cosp1p2)) * p1 * powr<2>(p2) + 2. * cosp1p2 * powr<3>(p2) +
              powr<2>(l1) * (13. * p1 + 13. * cosp1p2 * p2) +
              cosl1p2 * l1 *
                  (2. * cosp1p2 * powr<2>(p1) - 5. * p1 * p2 + 2. * powr<2>(cosp1p2) * p1 * p2 -
                   5. * cosp1p2 * powr<2>(p2))) +
         powr<4>(cosl1p2) * powr<4>(l1) * powr<3>(p2) *
             (6. * powr<2>(l1) * p2 + 11. * powr<2>(p1) * p2 +
              4. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + 6. * powr<3>(p2) +
              cosp1p2 * (6. * powr<2>(l1) * p1 + 7. * powr<3>(p1) + 14. * p1 * powr<2>(p2))) +
         powr<2>(p1) *
             (powr<6>(l1) * ((3. - 3. * powr<2>(cosp1p2)) * powr<2>(p1) +
                             cosp1p2 * (6. - 6. * powr<2>(cosp1p2)) * p1 * p2 +
                             (3. - 3. * powr<2>(cosp1p2)) * powr<2>(p2)) +
              powr<4>(l1) *
                  ((11. - 11. * powr<2>(cosp1p2)) * powr<4>(p1) +
                   cosp1p2 * (33. - 33. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 +
                   (21. + powr<2>(cosp1p2) - 22. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) +
                   cosp1p2 * (31. - 31. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) +
                   (10. - 10. * powr<2>(cosp1p2)) * powr<4>(p2)) +
              powr<2>(l1) * ((11. - 11. * powr<2>(cosp1p2)) * powr<6>(p1) +
                             cosp1p2 * (44. - 44. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 +
                             (29. + 22. * powr<2>(cosp1p2) - 51. * powr<4>(cosp1p2)) * powr<4>(p1) *
                                 powr<2>(p2) +
                             cosp1p2 * (72. - 58. * powr<2>(cosp1p2) - 14. * powr<4>(cosp1p2)) *
                                 powr<3>(p1) * powr<3>(p2) +
                             (21. + 14. * powr<2>(cosp1p2) - 35. * powr<4>(cosp1p2)) * powr<2>(p1) *
                                 powr<4>(p2) +
                             cosp1p2 * (20. - 20. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) +
                             (3. - 3. * powr<2>(cosp1p2)) * powr<6>(p2)) +
              p1 *
                  ((3. - 3. * powr<2>(cosp1p2)) * powr<7>(p1) +
                   cosp1p2 * (15. - 15. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 +
                   (11. + 14. * powr<2>(cosp1p2) - 25. * powr<4>(cosp1p2)) * powr<5>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (39. - 24. * powr<2>(cosp1p2) - 15. * powr<4>(cosp1p2)) * powr<4>(p1) *
                       powr<3>(p2) +
                   (11. + 30. * powr<2>(cosp1p2) - 39. * powr<4>(cosp1p2) - 2. * powr<6>(cosp1p2)) *
                       powr<3>(p1) * powr<4>(p2) +
                   cosp1p2 * (23. - 10. * powr<2>(cosp1p2) - 13. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       powr<5>(p2) +
                   (3. + 9. * powr<2>(cosp1p2) - 12. * powr<4>(cosp1p2)) * p1 * powr<6>(p2) +
                   cosp1p2 * (3. - 3. * powr<2>(cosp1p2)) * powr<7>(p2))) +
         powr<3>(cosl1p2) * powr<3>(l1) * powr<2>(p2) *
             (-4. * powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) +
              powr<2>(cosp1p2) * powr<2>(p1) * p2 *
                  (-11. * powr<2>(l1) - 17. * powr<2>(p1) - 25. * powr<2>(p2)) +
              cosp1p2 * p1 *
                  (-3. * powr<4>(l1) - 5. * powr<4>(p1) - 41. * powr<2>(p1) * powr<2>(p2) -
                   18. * powr<4>(p2) + powr<2>(l1) * (-6. * powr<2>(p1) - 23. * powr<2>(p2))) +
              p2 * (-3. * powr<4>(l1) - 12. * powr<4>(p1) - 11. * powr<2>(p1) * powr<2>(p2) -
                    3. * powr<4>(p2) + powr<2>(l1) * (-11. * powr<2>(p1) - 7. * powr<2>(p2)))) +
         powr<2>(cosl1p2) * powr<2>(l1) * p2 *
             (5. * powr<6>(p1) * p2 - 6. * powr<4>(l1) * powr<3>(p2) +
              3. * powr<4>(p1) * powr<3>(p2) - 2. * powr<4>(cosp1p2) * powr<4>(p1) * powr<3>(p2) +
              powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) *
                  (-14. * powr<2>(l1) - 6. * powr<2>(p1) - powr<2>(p2)) +
              powr<2>(l1) *
                  (3. * powr<4>(p1) * p2 - 12. * powr<2>(p1) * powr<3>(p2) - 6. * powr<5>(p2)) +
              powr<2>(cosp1p2) * powr<2>(p1) * p2 *
                  (-9. * powr<4>(l1) - 14. * powr<4>(p1) + 8. * powr<2>(p1) * powr<2>(p2) +
                   6. * powr<4>(p2) + powr<2>(l1) * (-31. * powr<2>(p1) - 13. * powr<2>(p2))) +
              cosp1p2 * p1 *
                  (-6. * powr<6>(p1) + 6. * powr<4>(p1) * powr<2>(p2) +
                   10. * powr<2>(p1) * powr<4>(p2) + 3. * powr<6>(p2) +
                   powr<4>(l1) * (-6. * powr<2>(p1) - 9. * powr<2>(p2)) +
                   powr<2>(l1) * (-14. * powr<4>(p1) - 20. * powr<2>(p1) * powr<2>(p2) -
                                  11. * powr<4>(p2)))) +
         powr<4>(cosl1p1) * powr<2>(l1) * powr<3>(p1) *
             (powr<4>(l1) * (-12. * p1 - 12. * cosp1p2 * p2) +
              cosl1p2 * powr<3>(l1) *
                  (-13. * cosp1p2 * powr<2>(p1) + 26. * p1 * p2 - 13. * powr<2>(cosp1p2) * p1 * p2 +
                   26. * cosp1p2 * powr<2>(p2)) +
              powr<2>(l1) *
                  ((-39. + 13. * powr<2>(cosp1p2)) * powr<3>(p1) +
                   cosp1p2 * (-65. + 5. * powr<2>(cosl1p2) + 13. * powr<2>(cosp1p2)) * powr<2>(p1) *
                       p2 +
                   (-35. + 14. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-2. + powr<2>(cosp1p2))) *
                       p1 * powr<2>(p2) +
                   (5. - 6. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) +
              p1 * ((-25. + 13. * powr<2>(cosp1p2)) * powr<4>(p1) +
                    cosp1p2 * (-62. + 26. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 +
                    (-51. - 5. * powr<2>(cosp1p2) + 5. * powr<4>(cosp1p2)) * powr<2>(p1) *
                        powr<2>(p2) +
                    cosp1p2 * (-58. + 16. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) +
                    (-22. + 7. * powr<2>(cosp1p2)) * powr<4>(p2)) +
              cosl1p2 * l1 *
                  (31. * powr<3>(p1) * p2 - 6. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) +
                   33. * p1 * powr<3>(p2) +
                   powr<2>(cosp1p2) * (-31. * powr<3>(p1) * p2 + 10. * p1 * powr<3>(p2)) +
                   cosp1p2 *
                       (-13. * powr<4>(p1) + 53. * powr<2>(p1) * powr<2>(p2) + 9. * powr<4>(p2)))) +
         powr<3>(cosl1p1) * l1 * powr<2>(p1) *
             (powr<6>(l1) * (3. * p1 + 3. * cosp1p2 * p2) +
              cosl1p2 * powr<5>(l1) *
                  (12. * cosp1p2 * powr<2>(p1) - 18. * p1 * p2 + 12. * powr<2>(cosp1p2) * p1 * p2 -
                   18. * cosp1p2 * powr<2>(p2)) +
              powr<4>(l1) * ((10. - 12. * powr<2>(cosp1p2)) * powr<3>(p1) +
                             cosp1p2 * (8. - 26. * powr<2>(cosl1p2) - 12. * powr<2>(cosp1p2)) *
                                 powr<2>(p1) * p2 +
                             (20. + powr<2>(cosl1p2) - 43. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                             (-21. + 27. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) +
              powr<2>(l1) * ((24. - 26. * powr<2>(cosp1p2)) * powr<5>(p1) +
                             cosp1p2 * (46. - 26. * powr<2>(cosl1p2) - 52. * powr<2>(cosp1p2)) *
                                 powr<4>(p1) * p2 +
                             (58. - 72. * powr<2>(cosp1p2) - 16. * powr<4>(cosp1p2) +
                              powr<2>(cosl1p2) * (6. - 52. * powr<2>(cosp1p2))) *
                                 powr<3>(p1) * powr<2>(p2) +
                             cosp1p2 *
                                 (24. - 74. * powr<2>(cosp1p2) +
                                  powr<2>(cosl1p2) * (4. - 2. * powr<2>(cosp1p2))) *
                                 powr<2>(p1) * powr<3>(p2) +
                             (31. - 77. * powr<2>(cosp1p2) +
                              powr<2>(cosl1p2) * (14. + 21. * powr<2>(cosp1p2))) *
                                 p1 * powr<4>(p2) +
                             (-22. + 13. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) +
              p1 * ((15. - 12. * powr<2>(cosp1p2)) * powr<6>(p1) +
                    cosp1p2 * (48. - 36. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 +
                    (44. + 6. * powr<2>(cosp1p2) - 26. * powr<4>(cosp1p2)) * powr<4>(p1) *
                        powr<2>(p2) +
                    cosp1p2 * (84. - 52. * powr<2>(cosp1p2) - 2. * powr<4>(cosp1p2)) * powr<3>(p1) *
                        powr<3>(p2) +
                    (33. + 4. * powr<2>(cosp1p2) - 13. * powr<4>(cosp1p2)) * powr<2>(p1) *
                        powr<4>(p2) +
                    cosp1p2 * (24. - 12. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) +
                    (6. - 3. * powr<2>(cosp1p2)) * powr<6>(p2)) +
              cosl1p2 * powr<3>(l1) *
                  (-65. * powr<3>(p1) * p2 + 16. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) +
                   (-69. + 4. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) +
                   powr<2>(cosp1p2) * (78. * powr<3>(p1) * p2 +
                                       (-11. - 2. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) +
                   cosp1p2 * (26. * powr<4>(p1) +
                              (-102. + 6. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) +
                              (-7. - 4. * powr<2>(cosl1p2)) * powr<4>(p2))) +
              cosl1p2 * l1 *
                  (-44. * powr<5>(p1) * p2 - 88. * powr<3>(p1) * powr<3>(p2) +
                   6. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) - 34. * p1 * powr<5>(p2) +
                   powr<3>(cosp1p2) *
                       (72. * powr<4>(p1) * powr<2>(p2) + 16. * powr<2>(p1) * powr<4>(p2)) +
                   powr<2>(cosp1p2) * (62. * powr<5>(p1) * p2 + 40. * powr<3>(p1) * powr<3>(p2) +
                                       p1 * powr<5>(p2)) +
                   cosp1p2 * (12. * powr<6>(p1) - 78. * powr<4>(p1) * powr<2>(p2) -
                              76. * powr<2>(p1) * powr<4>(p2) - 3. * powr<6>(p2)))) +
         cosl1p2 * l1 *
             (5. * powr<5>(cosp1p2) * powr<5>(p1) * powr<4>(p2) +
              powr<4>(cosp1p2) * powr<4>(p1) * powr<3>(p2) *
                  (33. * powr<2>(l1) + 31. * powr<2>(p1) + 26. * powr<2>(p2)) +
              powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) *
                  (34. * powr<4>(l1) + 44. * powr<4>(p1) + 65. * powr<2>(p1) * powr<2>(p2) +
                   18. * powr<4>(p2) + powr<2>(l1) * (88. * powr<2>(p1) + 69. * powr<2>(p2))) +
              powr<2>(cosp1p2) * powr<2>(p1) * p2 *
                  (6. * powr<6>(l1) + 21. * powr<6>(p1) + 37. * powr<4>(p1) * powr<2>(p2) +
                   16. * powr<2>(p1) * powr<4>(p2) + 3. * powr<6>(p2) +
                   powr<4>(l1) * (39. * powr<2>(p1) + 44. * powr<2>(p2)) +
                   powr<2>(l1) *
                       (58. * powr<4>(p1) + 97. * powr<2>(p1) * powr<2>(p2) + 43. * powr<4>(p2))) +
              p2 * (-6. * powr<8>(p1) + 3. * powr<6>(l1) * powr<2>(p2) -
                    14. * powr<6>(p1) * powr<2>(p2) - 6. * powr<4>(p1) * powr<4>(p2) +
                    powr<4>(l1) *
                        (-6. * powr<4>(p1) + 5. * powr<2>(p1) * powr<2>(p2) + 8. * powr<4>(p2)) +
                    powr<2>(l1) * (-14. * powr<6>(p1) - 14. * powr<4>(p1) * powr<2>(p2) +
                                   5. * powr<2>(p1) * powr<4>(p2) + 3. * powr<6>(p2))) +
              cosp1p2 * p1 *
                  (3. * powr<8>(p1) - 8. * powr<6>(p1) * powr<2>(p2) -
                   16. * powr<4>(p1) * powr<4>(p2) - 3. * powr<2>(p1) * powr<6>(p2) +
                   powr<6>(l1) * (3. * powr<2>(p1) + 6. * powr<2>(p2)) +
                   powr<4>(l1) *
                       (11. * powr<4>(p1) + 18. * powr<2>(p1) * powr<2>(p2) + 27. * powr<4>(p2)) +
                   powr<2>(l1) * (11. * powr<6>(p1) + 2. * powr<4>(p1) * powr<2>(p2) +
                                  26. * powr<2>(p1) * powr<4>(p2) + 15. * powr<6>(p2)))) +
         cosl1p1 *
             (powr<7>(l1) *
                  (-3. * powr<3>(p1) + (-6. - 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 +
                   (-6. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-3. + 3. * powr<2>(cosp1p2))) *
                       p1 * powr<2>(p2) +
                   (-3. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) +
              powr<5>(l1) *
                  ((-23. + 12. * powr<2>(cosp1p2)) * powr<5>(p1) +
                   cosp1p2 * (-57. + 2. * powr<2>(cosl1p2) + 24. * powr<2>(cosp1p2)) * powr<4>(p1) *
                       p2 +
                   (-20. - 10. * powr<2>(cosl1p2) - 32. * powr<2>(cosp1p2)) * powr<3>(p1) *
                       powr<2>(p2) +
                   cosp1p2 *
                       (-27. - 14. * powr<4>(cosl1p2) - 22. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (2. + 13. * powr<2>(cosp1p2))) *
                       powr<2>(p1) * powr<3>(p2) +
                   (-27. * powr<2>(cosp1p2) + powr<4>(cosl1p2) * (-14. + powr<2>(cosp1p2)) +
                    powr<2>(cosl1p2) * (11. + 23. * powr<2>(cosp1p2))) *
                       p1 * powr<4>(p2) +
                   (-8. + 7. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * cosp1p2 * powr<5>(p2)) +
              cosl1p2 * powr<3>(p1) * p2 *
                  ((-3. + 3. * powr<2>(cosp1p2)) * powr<6>(p1) +
                   cosp1p2 * (-12. + 12. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 +
                   (-11. - 2. * powr<2>(cosp1p2) + 13. * powr<4>(cosp1p2)) * powr<4>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (-28. + 26. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) * powr<3>(p1) *
                       powr<3>(p2) +
                   (-11. - 2. * powr<2>(cosp1p2) + 13. * powr<4>(cosp1p2)) * powr<2>(p1) *
                       powr<4>(p2) +
                   cosp1p2 * (-12. + 12. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) +
                   (-3. + 3. * powr<2>(cosp1p2)) * powr<6>(p2)) +
              powr<3>(l1) *
                  ((-39. + 28. * powr<2>(cosp1p2)) * powr<7>(p1) +
                   cosp1p2 * (-128. + 2. * powr<2>(cosl1p2) + 84. * powr<2>(cosp1p2)) *
                       powr<6>(p1) * p2 +
                   (-72. - 76. * powr<2>(cosp1p2) + 58. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (-6. - 11. * powr<2>(cosp1p2))) *
                       powr<5>(p1) * powr<2>(p2) +
                   cosp1p2 *
                       (-142. - 3. * powr<4>(cosl1p2) + 24. * powr<2>(cosp1p2) +
                        2. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-15. + 4. * powr<2>(cosp1p2))) *
                       powr<4>(p1) * powr<3>(p2) +
                   (-31. - 59. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2) +
                    powr<4>(cosl1p2) * (-7. - 4. * powr<2>(cosp1p2)) +
                    powr<2>(cosl1p2) * (20. - 13. * powr<2>(cosp1p2) + 6. * powr<4>(cosp1p2))) *
                       powr<3>(p1) * powr<4>(p2) +
                   cosp1p2 *
                       (-27. - 14. * powr<4>(cosl1p2) - 21. * powr<2>(cosp1p2) +
                        powr<2>(cosl1p2) * (2. + 27. * powr<2>(cosp1p2))) *
                       powr<2>(p1) * powr<5>(p2) +
                   (-6. * powr<4>(cosl1p2) - 15. * powr<2>(cosp1p2) +
                    powr<2>(cosl1p2) * (9. + 18. * powr<2>(cosp1p2))) *
                       p1 * powr<6>(p2) +
                   (-3. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<7>(p2)) +
              l1 * powr<2>(p1) *
                  ((-15. + 12. * powr<2>(cosp1p2)) * powr<7>(p1) +
                   cosp1p2 * (-63. - 3. * powr<2>(cosl1p2) + 48. * powr<2>(cosp1p2)) * powr<6>(p1) *
                       p2 +
                   (-44. - 54. * powr<2>(cosp1p2) + 62. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (6. - 18. * powr<2>(cosp1p2))) *
                       powr<5>(p1) * powr<2>(p2) +
                   cosp1p2 *
                       (-128. + 46. * powr<2>(cosp1p2) + 28. * powr<4>(cosp1p2) +
                        powr<2>(cosl1p2) * (2. - 26. * powr<2>(cosp1p2))) *
                       powr<4>(p1) * powr<3>(p2) +
                   (-33. - 88. * powr<2>(cosp1p2) + 65. * powr<4>(cosp1p2) + 2. * powr<6>(cosp1p2) +
                    powr<2>(cosl1p2) * (14. - 39. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2))) *
                       powr<3>(p1) * powr<4>(p2) +
                   cosp1p2 *
                       (-57. + 8. * powr<2>(cosp1p2) + 13. * powr<4>(cosp1p2) +
                        powr<2>(cosl1p2) * (2. - 26. * powr<2>(cosp1p2))) *
                       powr<2>(p1) * powr<5>(p2) +
                   (-6. - 21. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2) +
                    powr<2>(cosl1p2) * (6. - 18. * powr<2>(cosp1p2))) *
                       p1 * powr<6>(p2) +
                   cosp1p2 * (-6. - 3. * powr<2>(cosl1p2) + 3. * powr<2>(cosp1p2)) * powr<7>(p2)) +
              cosl1p2 * powr<6>(l1) *
                  (3. * powr<3>(p1) * p2 - 3. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) +
                   (-15. + 18. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) +
                   powr<2>(cosp1p2) *
                       (-21. * powr<3>(p1) * p2 + (9. - 6. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) +
                   cosp1p2 * (-12. * powr<4>(p1) +
                              (-15. + 18. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) +
                              (6. - 6. * powr<2>(cosl1p2)) * powr<4>(p2))) +
              cosl1p2 * powr<4>(l1) *
                  (16. * powr<5>(p1) * p2 +
                   (-26. + 41. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) -
                   9. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) +
                   (-27. + 23. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * p1 * powr<5>(p2) +
                   powr<3>(cosp1p2) * (-76. * powr<4>(p1) * powr<2>(p2) +
                                       (-7. - 4. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2)) +
                   powr<2>(cosp1p2) * (-86. * powr<5>(p1) * p2 +
                                       (-67. + 35. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) +
                                       (11. - 14. * powr<2>(cosl1p2)) * p1 * powr<5>(p2)) +
                   cosp1p2 * (-28. * powr<6>(p1) +
                              (-34. + 27. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) +
                              (-49. + 68. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * powr<2>(p1) *
                                  powr<4>(p2) +
                              (6. - 6. * powr<2>(cosl1p2)) * powr<6>(p2))) +
              cosl1p2 * powr<2>(l1) * p1 *
                  (8. * powr<6>(p1) * p2 +
                   (-2. + 5. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<3>(p2) -
                   5. * powr<5>(cosp1p2) * powr<3>(p1) * powr<4>(p2) +
                   (-18. + 6. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<5>(p2) +
                   (-6. + 3. * powr<2>(cosl1p2)) * powr<7>(p2) +
                   powr<4>(cosp1p2) *
                       (-53. * powr<4>(p1) * powr<3>(p2) - 26. * powr<2>(p1) * powr<5>(p2)) +
                   powr<3>(cosp1p2) * (-78. * powr<5>(p1) * powr<2>(p2) +
                                       (-102. + 6. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<4>(p2) -
                                       18. * p1 * powr<6>(p2)) +
                   cosp1p2 * (-12. * powr<7>(p1) +
                              (-18. + 6. * powr<2>(cosl1p2)) * powr<5>(p1) * powr<2>(p2) +
                              (-34. + 27. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<4>(p2) +
                              (-15. + 18. * powr<2>(cosl1p2)) * p1 * powr<6>(p2)) +
                   powr<2>(cosp1p2) * (-50. * powr<6>(p1) * p2 +
                                       (-92. + 16. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<3>(p2) +
                                       (-40. + 27. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<5>(p2) -
                                       3. * powr<7>(p2)))) +
         powr<2>(cosl1p1) * p1 *
             (powr<4>(cosl1p2) * powr<4>(l1) * powr<3>(p2) *
                  (4. * cosp1p2 * powr<2>(p1) + 4. * p1 * p2 - powr<2>(cosp1p2) * p1 * p2 -
                   cosp1p2 * powr<2>(p2)) +
              powr<6>(l1) * ((9. + 3. * powr<2>(cosp1p2)) * powr<3>(p1) +
                             cosp1p2 * (21. + 3. * powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                             (-3. + 30. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) +
                             15. * cosp1p2 * powr<3>(p2)) +
              powr<4>(l1) * ((30. - 2. * powr<2>(cosp1p2)) * powr<5>(p1) +
                             cosp1p2 * (88. - 4. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 +
                             (14. + 117. * powr<2>(cosp1p2) + 7. * powr<4>(cosp1p2)) * powr<3>(p1) *
                                 powr<2>(p2) +
                             cosp1p2 * (59. + 77. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) +
                             (-10. + 91. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) +
                             27. * cosp1p2 * powr<5>(p2)) +
              powr<3>(p1) * ((-3. + 3. * powr<2>(cosp1p2)) * powr<6>(p1) +
                             cosp1p2 * (-12. + 12. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 +
                             (-11. - 2. * powr<2>(cosp1p2) + 13. * powr<4>(cosp1p2)) * powr<4>(p1) *
                                 powr<2>(p2) +
                             cosp1p2 * (-28. + 26. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) *
                                 powr<3>(p1) * powr<3>(p2) +
                             (-11. - 2. * powr<2>(cosp1p2) + 13. * powr<4>(cosp1p2)) * powr<2>(p1) *
                                 powr<4>(p2) +
                             cosp1p2 * (-12. + 12. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) +
                             (-3. + 3. * powr<2>(cosp1p2)) * powr<6>(p2)) +
              powr<2>(l1) *
                  ((14. - 2. * powr<2>(cosp1p2)) * powr<7>(p1) +
                   cosp1p2 * (54. - 6. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 +
                   (22. + 88. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2)) * powr<5>(p1) *
                       powr<2>(p2) +
                   cosp1p2 * (76. + 72. * powr<2>(cosp1p2) - powr<4>(cosp1p2)) * powr<4>(p1) *
                       powr<3>(p2) +
                   (1. + 117. * powr<2>(cosp1p2) + 14. * powr<4>(cosp1p2)) * powr<3>(p1) *
                       powr<4>(p2) +
                   cosp1p2 * (32. + 43. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<5>(p2) +
                   (-3. + 30. * powr<2>(cosp1p2)) * p1 * powr<6>(p2) + 6. * cosp1p2 * powr<7>(p2)) +
              powr<3>(cosl1p2) * powr<3>(l1) * powr<2>(p2) *
                  (2. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) +
                   p1 * p2 * (-25. * powr<2>(l1) - 17. * powr<2>(p1) - 11. * powr<2>(p2)) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (12. * powr<2>(l1) - 27. * powr<2>(p1) + 12. * powr<2>(p2)) +
                   cosp1p2 *
                       (-16. * powr<4>(p1) - 35. * powr<2>(p1) * powr<2>(p2) + 6. * powr<4>(p2) +
                        powr<2>(l1) * (-27. * powr<2>(p1) + 14. * powr<2>(p2)))) +
              powr<2>(cosl1p2) * powr<2>(l1) * p2 *
                  (powr<3>(cosp1p2) * powr<2>(p1) * (-21. * powr<2>(l1) + 52. * powr<2>(p1)) *
                       powr<2>(p2) +
                   powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (-6. * powr<4>(l1) + 65. * powr<4>(p1) + 72. * powr<2>(p1) * powr<2>(p2) -
                        6. * powr<4>(p2) + powr<2>(l1) * (72. * powr<2>(p1) - 59. * powr<2>(p2))) +
                   p1 * p2 *
                       (6. * powr<4>(l1) - 14. * powr<4>(p1) - 31. * powr<2>(p1) * powr<2>(p2) -
                        9. * powr<4>(p2) + powr<2>(l1) * (8. * powr<2>(p1) - 13. * powr<2>(p2))) +
                   cosp1p2 *
                       (18. * powr<6>(p1) + 11. * powr<4>(p1) * powr<2>(p2) - 3. * powr<6>(p2) +
                        powr<4>(l1) * (18. * powr<2>(p1) - 18. * powr<2>(p2)) +
                        powr<2>(l1) * (39. * powr<4>(p1) + 13. * powr<2>(p1) * powr<2>(p2) -
                                       23. * powr<4>(p2)))) +
              cosl1p2 * l1 *
                  (-2. * powr<5>(cosp1p2) * powr<4>(p1) * powr<4>(p2) +
                   powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) *
                       (10. * powr<2>(l1) - 31. * powr<2>(p1) - 13. * powr<2>(p2)) +
                   powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) *
                       (-powr<4>(l1) - 62. * powr<4>(p1) - 78. * powr<2>(p1) * powr<2>(p2) -
                        12. * powr<4>(p2) +
                        powr<2>(l1) * (-40. * powr<2>(p1) + 11. * powr<2>(p2))) +
                   powr<2>(cosp1p2) * p1 * p2 *
                       (-3. * powr<6>(l1) - 30. * powr<6>(p1) - 33. * powr<4>(p1) * powr<2>(p2) -
                        14. * powr<2>(p1) * powr<4>(p2) - 3. * powr<6>(p2) +
                        powr<4>(l1) * (-14. * powr<2>(p1) - 15. * powr<2>(p2)) +
                        powr<2>(l1) * (-33. * powr<4>(p1) - 17. * powr<2>(p1) * powr<2>(p2) -
                                       15. * powr<4>(p2))) +
                   p1 * p2 *
                       (3. * powr<6>(l1) + 21. * powr<6>(p1) + 58. * powr<4>(p1) * powr<2>(p2) +
                        39. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2) +
                        powr<4>(l1) * (16. * powr<2>(p1) + 43. * powr<2>(p2)) +
                        powr<2>(l1) * (37. * powr<4>(p1) + 97. * powr<2>(p1) * powr<2>(p2) +
                                       44. * powr<4>(p2))) +
                   cosp1p2 *
                       (-3. * powr<8>(p1) + 50. * powr<6>(p1) * powr<2>(p2) +
                        86. * powr<4>(p1) * powr<4>(p2) + 21. * powr<2>(p1) * powr<6>(p2) +
                        powr<6>(l1) * (-3. * powr<2>(p1) + 3. * powr<2>(p2)) +
                        powr<4>(l1) * (2. * powr<4>(p1) + 40. * powr<2>(p1) * powr<2>(p2) -
                                       11. * powr<4>(p2)) +
                        powr<2>(l1) * (2. * powr<6>(p1) + 92. * powr<4>(p1) * powr<2>(p2) +
                                       67. * powr<2>(p1) * powr<4>(p2) - 9. * powr<6>(p2)))))) *
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
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                     powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
            0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                       (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                        3. * powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                     2. * cosp1p2 * p1 * p2 + p2 * (-2. * cosl1p2 * l1 + p2)))),
            atan2(-1.732050807568877 *
                      (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                       p2 * (-2. * cosl1p2 * l1 + p2)) *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                  -1. + 3. * l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                            powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                     powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
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
                             p1 * (p1 + cosp1p2 * p2)))) -
    13.5 * powr<-1>(-1. + powr<2>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
        powr<-1>(3. * powr<4>(p1) + 6. * cosp1p2 * powr<3>(p1) * p2 +
                 (8. + powr<2>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) +
                 6. * cosp1p2 * p1 * powr<3>(p2) + 3. * powr<4>(p2)) *
        ((3. + 2. * cosl1p1 * cosl1p2 * cosp1p2 - 3. * powr<2>(cosp1p2) +
          powr<2>(cosl1p1) * (-3. + powr<2>(cosp1p2))) *
             powr<6>(p1) +
         (cosp1p2 * (13. + 2. * powr<2>(cosl1p2) - 13. * powr<2>(cosp1p2)) +
          cosl1p1 * cosl1p2 * (-6. + 10. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p1) * (-9. * cosp1p2 + 3. * powr<3>(cosp1p2))) *
             powr<5>(p1) * p2 +
         (9. + 7. * powr<2>(cosp1p2) - 16. * powr<4>(cosp1p2) +
          powr<2>(cosl1p2) * (-3. + 9. * powr<2>(cosp1p2)) +
          cosl1p1 * cosl1p2 * cosp1p2 * (-16. + 14. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p1) * (-6. + 2. * powr<4>(cosp1p2))) *
             powr<4>(p1) * powr<2>(p2) +
         (powr<2>(cosl1p1) * (-7. * cosp1p2 + 11. * powr<3>(cosp1p2)) +
          cosl1p1 * cosl1p2 * (-12. + 4. * powr<4>(cosp1p2)) +
          cosp1p2 * (26. - 22. * powr<2>(cosp1p2) - 4. * powr<4>(cosp1p2) +
                     powr<2>(cosl1p2) * (-7. + 11. * powr<2>(cosp1p2)))) *
             powr<3>(p1) * powr<3>(p2) +
         (9. + 7. * powr<2>(cosp1p2) - 16. * powr<4>(cosp1p2) +
          powr<2>(cosl1p1) * (-3. + 9. * powr<2>(cosp1p2)) +
          cosl1p1 * cosl1p2 * cosp1p2 * (-16. + 14. * powr<2>(cosp1p2)) +
          powr<2>(cosl1p2) * (-6. + 2. * powr<4>(cosp1p2))) *
             powr<2>(p1) * powr<4>(p2) +
         (2. * powr<2>(cosl1p1) * cosp1p2 + cosl1p1 * cosl1p2 * (-6. + 10. * powr<2>(cosp1p2)) +
          cosp1p2 *
              (13. - 13. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-9. + 3. * powr<2>(cosp1p2)))) *
             p1 * powr<5>(p2) +
         (3. + 2. * cosl1p1 * cosl1p2 * cosp1p2 - 3. * powr<2>(cosp1p2) +
          powr<2>(cosl1p2) * (-3. + powr<2>(cosp1p2))) *
             powr<6>(p2) +
         powr<2>(l1) * ((5. + 4. * cosl1p1 * cosl1p2 * cosp1p2 - 5. * powr<2>(cosp1p2) +
                         powr<2>(cosl1p1) * (-5. + powr<2>(cosp1p2))) *
                            powr<4>(p1) +
                        (cosp1p2 * (12. + 3. * powr<2>(cosl1p2) - 12. * powr<2>(cosp1p2)) +
                         cosl1p1 * cosl1p2 * (-10. + 12. * powr<2>(cosp1p2)) +
                         powr<2>(cosl1p1) * (-6. * cosp1p2 + powr<3>(cosp1p2))) *
                            powr<3>(p1) * p2 +
                        (10. - 6. * powr<2>(cosp1p2) - 4. * powr<4>(cosp1p2) +
                         cosl1p1 * cosl1p2 * cosp1p2 * (-10. + 6. * powr<2>(cosp1p2)) +
                         powr<2>(cosl1p1) * (-5. + 7. * powr<2>(cosp1p2)) +
                         powr<2>(cosl1p2) * (-5. + 7. * powr<2>(cosp1p2))) *
                            powr<2>(p1) * powr<2>(p2) +
                        (3. * powr<2>(cosl1p1) * cosp1p2 +
                         cosl1p1 * cosl1p2 * (-10. + 12. * powr<2>(cosp1p2)) +
                         cosp1p2 * (12. - 12. * powr<2>(cosp1p2) +
                                    powr<2>(cosl1p2) * (-6. + powr<2>(cosp1p2)))) *
                            p1 * powr<3>(p2) +
                        (5. + 4. * cosl1p1 * cosl1p2 * cosp1p2 - 5. * powr<2>(cosp1p2) +
                         powr<2>(cosl1p2) * (-5. + powr<2>(cosp1p2))) *
                            powr<4>(p2)) +
         l1 * (powr<3>(cosl1p1) * powr<2>(p1) *
                   ((5. - powr<2>(cosp1p2)) * powr<3>(p1) +
                    cosp1p2 * (6. - powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
                    (5. - 7. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) - 3. * cosp1p2 * powr<3>(p2)) +
               cosl1p1 *
                   ((-5. + 7. * powr<2>(cosp1p2)) * powr<5>(p1) +
                    cosp1p2 * (-10. - 7. * powr<2>(cosl1p2) + 16. * powr<2>(cosp1p2)) *
                        powr<4>(p1) * p2 +
                    (-10. + 10. * powr<2>(cosp1p2) + 4. * powr<4>(cosp1p2) +
                     powr<2>(cosl1p2) * (15. - 19. * powr<2>(cosp1p2))) *
                        powr<3>(p1) * powr<2>(p2) +
                    cosp1p2 *
                        (-12. + 8. * powr<2>(cosp1p2) +
                         powr<2>(cosl1p2) * (16. - 7. * powr<2>(cosp1p2))) *
                        powr<2>(p1) * powr<3>(p2) +
                    (-5. - powr<2>(cosp1p2) + powr<2>(cosl1p2) * (15. - 13. * powr<2>(cosp1p2))) *
                        p1 * powr<4>(p2) +
                    (-2. - 4. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) +
               powr<2>(cosl1p1) * cosl1p2 * p1 *
                   (15. * powr<3>(p1) * p2 - 7. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) +
                    15. * p1 * powr<3>(p2) +
                    powr<2>(cosp1p2) * (-13. * powr<3>(p1) * p2 - 19. * p1 * powr<3>(p2)) +
                    cosp1p2 *
                        (-4. * powr<4>(p1) + 16. * powr<2>(p1) * powr<2>(p2) - 7. * powr<4>(p2))) +
               cosl1p2 *
                   (-5. * powr<4>(p1) * p2 +
                    (-10. + 5. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<3>(p2) +
                    4. * powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) +
                    (-5. + 5. * powr<2>(cosl1p2)) * powr<5>(p2) +
                    powr<3>(cosp1p2) * (8. * powr<3>(p1) * powr<2>(p2) +
                                        (16. - powr<2>(cosl1p2)) * p1 * powr<4>(p2)) +
                    cosp1p2 * (-2. * powr<5>(p1) +
                               (-12. - 3. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<2>(p2) +
                               (-10. + 6. * powr<2>(cosl1p2)) * p1 * powr<4>(p2)) +
                    powr<2>(cosp1p2) * (-powr<4>(p1) * p2 +
                                        (10. - 7. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<3>(p2) +
                                        (7. - powr<2>(cosl1p2)) * powr<5>(p2))))) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     ZA(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                     powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
            0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                       (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                        3. * powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                     2. * cosp1p2 * p1 * p2 + p2 * (-2. * cosl1p2 * l1 + p2)))),
            atan2(-1.732050807568877 *
                      (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                       p2 * (-2. * cosl1p2 * l1 + p2)) *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                  -1. + 3. * l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                            powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                     powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
        ZA4SP(0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 +
                                        powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2))) -
    4.5 * powr<-1>(-1. + powr<2>(cosp1p2)) * powr<2>(l1) *
        powr<-1>(3. * powr<4>(p1) + 6. * cosp1p2 * powr<3>(p1) * p2 +
                 (8. + powr<2>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) +
                 6. * cosp1p2 * p1 * powr<3>(p2) + 3. * powr<4>(p2)) *
        (powr<3>(cosl1p2) * l1 * (-cosp1p2 * p1 - p2) * powr<2>(p2) +
         powr<2>(cosl1p2) * cosp1p2 * p1 * powr<2>(p2) * (cosp1p2 * p1 + p2) +
         powr<3>(cosl1p1) * l1 * powr<2>(p1) * (p1 + cosp1p2 * p2) +
         powr<2>(p1) * ((1. - powr<2>(cosp1p2)) * powr<2>(p1) +
                        cosp1p2 * (2. - 2. * powr<2>(cosp1p2)) * p1 * p2 +
                        (1. - powr<2>(cosp1p2)) * powr<2>(p2)) +
         cosl1p2 * l1 *
             (2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + powr<3>(p2) +
              cosp1p2 * (powr<3>(p1) + 2. * p1 * powr<2>(p2))) +
         cosl1p1 * (cosl1p2 * p1 * p2 *
                        ((-1. + powr<2>(cosp1p2)) * powr<2>(p1) +
                         cosp1p2 * (-1. - powr<2>(cosp1p2)) * p1 * p2 +
                         (-1. - powr<2>(cosp1p2)) * powr<2>(p2)) +
                    l1 * (-powr<3>(p1) + (-2. - powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 +
                          (-2. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-1. + powr<2>(cosp1p2))) *
                              p1 * powr<2>(p2) +
                          (-1. + powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2))) +
         powr<2>(cosl1p1) * p1 *
             ((-1. + powr<2>(cosp1p2)) * powr<3>(p1) +
              cosp1p2 * (-1. + powr<2>(cosp1p2)) * powr<2>(p1) * p2 +
              (-1. + 2. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + cosp1p2 * powr<3>(p2) +
              cosl1p2 * l1 *
                  (p1 * p2 - powr<2>(cosp1p2) * p1 * p2 +
                   cosp1p2 * (-powr<2>(p1) + powr<2>(p2))))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                    (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                    2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
              atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) *
                        powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    -1. + 3. * l1 * (l1 - cosl1p1 * p1) *
                              powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                       powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                         (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                          3. * powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       2. * cosp1p2 * p1 * p2 + p2 * (-2. * cosl1p2 * l1 + p2)))),
              atan2(1.732050807568877 *
                        (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                         p2 * (-2. * cosl1p2 * l1 + p2)) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                    -1. + 3. * l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                              powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                       powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
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
        (dtZc(k) * RB(powr<2>(k), powr<2>(l1)) + RBdot(powr<2>(k), powr<2>(l1)) * Zc(k) +
         RB(powr<2>(k), powr<2>(l1)) * (-50. * Zc(k) + 50. * Zc(1.02 * k))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * Zc(k) + powr<2>(l1) * Zc(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) * Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     Zc(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)))) *
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
      const auto _interp9 =
        RB(powr<2>(k), fma(2., cosp1p2 * p1 * p2, fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))));
      const auto _interp10 = ZA(sqrt(fma(
        2., cosp1p2 * p1 * p2, fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))); // clang-format off

const auto _interp14 = ZA3(0.816496580927726 * sqrt(fma(2., cosp1p2 * p1 * p2,
                                 fma(-1., l1 *(cosl1p1 *p1 + cosl1p2 * p2),
                                     powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
    0.5 * sqrt(powr<-2>(fma(2., cosp1p2 * p1 * p2,
                            fma(-1., l1 *(cosl1p1 *p1 + cosl1p2 * p2),
                                powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))) *
               fma(3.,
                   powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                           p2 * (-2. * cosl1p2 * l1 + p2)),
                   powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                           2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))),
    atan2(-1.732050807568877 *
              fma(-2., cosl1p1 * l1 * p1,
                  fma(2., cosp1p2 * p1 * p2, fma(p2, -2. * cosl1p2 * l1 + p2, powr<2>(p1)))) *
              powr<-1>(fma(2., cosp1p2 * p1 * p2,
                           fma(-1., l1 *(cosl1p1 *p1 + cosl1p2 * p2),
                               powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
          fma(3.,
              l1 *(l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                  powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                           l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
              -1.)));
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _interp19 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 + (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                      2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
                atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                      -1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))); // clang-format off

const auto _interp21 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
      0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                          l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                 (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                          2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                  3. * powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                               p2 * (-2. * cosl1p2 * l1 + p2)))),
      atan2(1.732050807568877 *
                (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                 p2 * (-2. * cosl1p2 * l1 + p2)) *
                powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                         l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
            -1. + 3. * l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                      powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + cosl1p2 * p2))));
        // clang-format on
        // clang-format off

const auto _interp23 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
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

        const auto _interp24 = dtZc(k);
        const auto _interp25 = Zc(k);
        const auto _interp26 = Zc(1.02 * k);
        const auto _interp27 = Zc(l1);
        const auto _interp28 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
        const auto _interp29 =
          Zc(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)));
        const auto _cse1 = powr<2>(cosp1p2);
        const auto _cse2 = powr<2>(p2);
        const auto _cse3 = powr<2>(p1);
        const auto _cse4 = -_cse1;
        const auto _cse5 = 1. + _cse4;
        const auto _cse6 = powr<3>(p2);
        const auto _cse7 = powr<3>(p1);
        const auto _cse8 = -1. + _cse1;
        const auto _cse9 = -1. + _cse4;
        const auto _cse10 = powr<2>(cosl1p2);
        const auto _cse11 = -2. * _cse1;
        const auto _cse12 = powr<2>(l1);
        const auto _cse13 = -2. * cosl1p1 * l1 * p1; // clang-format off
_acc += - 4.5 *
    _cse12 *powr<-1>(_cse8) * _interp19 * _interp21 *
    _interp23 *powr<-1>(fma(3., powr<2>(_cse2),
                            fma(8. + _cse1, _cse2 *_cse3,
                                fma(3., powr<2>(_cse3),
                                    fma(6., _cse6 * cosp1p2 * p1,
                                        fma(6., _cse7 * cosp1p2 * p2, 0.)))))) *
    fma(_cse2, powr<3>(cosl1p2) * l1 * (-cosp1p2 * p1 - p2),
        fma(_cse10, _cse2 * cosp1p2 * p1 * (cosp1p2 * p1 + p2),
            fma(cosl1p2,
                l1 *(_cse6 + cosp1p2 * (_cse7 + 2. * _cse2 * p1) + 2. * _cse1 * _cse3 * p2),
                fma(_cse3, powr<3>(cosl1p1) * l1 * (p1 + cosp1p2 * p2),
                    fma(_cse3, _cse2 *_cse5 + _cse3 * _cse5 + (2. + _cse11) * cosp1p2 * p1 * p2,
                        fma(powr<2>(cosl1p1),
                            p1 *(_cse7 *_cse8 + _cse6 * cosp1p2 + (-1. + 2. * _cse1) * _cse2 * p1 +
                                 _cse3 * _cse8 * cosp1p2 * p2 +
                                 cosl1p2 * l1 *
                                     ((_cse2 - _cse3) * cosp1p2 + p1 * p2 - _cse1 * p1 * p2)),
                            fma(cosl1p1,
                                l1 * (-_cse7 + (-1. + _cse10) * _cse6 * cosp1p2 +
                                      _cse2 * (_cse11 + _cse10 * _cse8) * p1 +
                                      (-2. - _cse10) * _cse3 * cosp1p2 * p2) +
                                    cosl1p2 * p1 * p2 *
                                        (_cse3 * _cse8 + _cse2 * _cse9 + _cse9 * cosp1p2 * p1 * p2),
                                0.))))))) *
    powr<-1>(fma(_cse12 + _cse13 + _cse3, _interp28, fma(_interp25, _interp7, 0.))) *
    fma(_interp2, _interp24,
        fma(_interp2, -50. * _interp25 + 50. * _interp26, fma(_interp25, _interp3, 0.))) *
    powr<-2>(fma(_interp2, _interp25, fma(_cse12, _interp27, 0.))) *
    powr<-1>(fma(
        _interp25, _interp9,
        fma(_interp29,
            _cse12 + _cse13 + _cse2 + _cse3 - 2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2, 0.)));
        // clang-format on
      }
      { // subkernel 2
        const auto _interp17 = ZA4SP(
          0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)));
        const auto _cse1 = powr<2>(cosp1p2);
        const auto _cse2 = powr<2>(p1);
        const auto _cse3 = powr<2>(p2);
        const auto _cse4 = powr<2>(cosl1p1);
        const auto _cse5 = powr<2>(cosl1p2);
        const auto _cse6 = powr<2>(_cse1);
        const auto _cse7 = powr<2>(_cse2);
        const auto _cse8 = powr<3>(cosp1p2);
        const auto _cse9 = powr<3>(p1);
        const auto _cse10 = powr<3>(p2);
        const auto _cse11 = 7. * _cse1;
        const auto _cse12 = -16. * _cse6;
        const auto _cse13 = 9. * _cse1;
        const auto _cse14 = -3. + _cse13;
        const auto _cse15 = 14. * _cse1;
        const auto _cse16 = -16. + _cse15;
        const auto _cse17 = _cse16 * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse18 = 2. * _cse6;
        const auto _cse19 = -6. + _cse18;
        const auto _cse20 = powr<2>(_cse3);
        const auto _cse21 = 10. * _cse1;
        const auto _cse22 = -6. + _cse21;
        const auto _cse23 = _cse22 * cosl1p1 * cosl1p2;
        const auto _cse24 = -13. * _cse1;
        const auto _cse25 = 2. * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse26 = -3. * _cse1;
        const auto _cse27 = -3. + _cse1;
        const auto _cse28 = powr<2>(l1);
        const auto _cse29 = -4. * _cse6;
        const auto _cse30 = -5. + _cse11;
        const auto _cse31 = 12. * _cse1;
        const auto _cse32 = -10. + _cse31;
        const auto _cse33 = _cse32 * cosl1p1 * cosl1p2;
        const auto _cse34 = -12. * _cse1;
        const auto _cse35 = 4. * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse36 = -5. * _cse1;
        const auto _cse37 = -5. + _cse1;
        const auto _cse38 = -_cse1;
        const auto _cse39 = powr<5>(p1);
        const auto _cse40 = 4. * _cse6;
        const auto _cse41 = -7. * _cse1;
        const auto _cse42 = powr<5>(p2);
        const auto _cse43 = 5. * _cse5;
        const auto _cse44 = -7. * _cse5;
        const auto _cse45 = -_cse5;
        const auto _cse46 = powr<6>(k);
        const auto _cse47 = -2. * cosl1p1 * l1 * p1;
        const auto _cse48 = -2. * cosl1p2 * l1 * p2;
        const auto _cse49 = 2. * cosp1p2 * p1 * p2;
        const auto _cse50 = _cse2 + _cse28 + _cse3 + _cse47 + _cse48 + _cse49; // clang-format off
_acc += - 13.5 *
    powr<-1>(-1. + _cse1) * powr<-1>(1. + _cse46) * powr<-1>(_cse50) * _interp14 * _interp17 *
    powr<-1>(fma(3., _cse20,
                 fma(8. + _cse1, _cse2 *_cse3,
                     fma(3., _cse7,
                         fma(6., _cse10 * cosp1p2 * p1, fma(6., _cse9 * cosp1p2 * p2, 0.)))))) *
    fma(powr<3>(_cse2), 3. + _cse25 + _cse26 + _cse27 * _cse4,
        fma(_cse2, _cse20 * (9. + _cse11 + _cse12 + _cse17 + _cse14 * _cse4 + _cse19 * _cse5),
            fma(powr<3>(_cse3), 3. + _cse25 + _cse26 + _cse27 * _cse5,
                fma(_cse3,
                    (9. + _cse11 + _cse12 + _cse17 + _cse19 * _cse4 + _cse14 * _cse5) * _cse7,
                    fma(_cse10,
                        _cse9 *
                            ((-12. + _cse40) * cosl1p1 * cosl1p2 +
                             _cse4 * (11. * _cse8 - 7. * cosp1p2) +
                             (26. - 22. * _cse1 + _cse29 + (-7. + 11. * _cse1) * _cse5) * cosp1p2),
                        fma(_cse42,
                            (_cse23 + 2. * _cse4 * cosp1p2 +
                             (13. + _cse24 + (-9. + 3. * _cse1) * _cse5) * cosp1p2) *
                                p1,
                            fma(_cse39,
                                (_cse23 + _cse4 * (3. * _cse8 - 9. * cosp1p2) +
                                 (13. + _cse24 + 2. * _cse5) * cosp1p2) *
                                    p2,
                                fma(_cse28,
                                    _cse20 * (5. + _cse35 + _cse36 + _cse37 * _cse5) +
                                        (5. + _cse35 + _cse36 + _cse37 * _cse4) * _cse7 +
                                        _cse2 * _cse3 *
                                            (10. - 6. * _cse1 + _cse29 + _cse30 * _cse4 +
                                             _cse30 * _cse5 +
                                             (-10. + 6. * _cse1) * cosl1p1 * cosl1p2 * cosp1p2) +
                                        _cse10 *
                                            (_cse33 + 3. * _cse4 * cosp1p2 +
                                             (12. + _cse34 + (-6. + _cse1) * _cse5) * cosp1p2) *
                                            p1 +
                                        _cse9 *
                                            (_cse33 + _cse4 * (_cse8 - 6. * cosp1p2) +
                                             (12. + _cse34 + 3. * _cse5) * cosp1p2) *
                                            p2,
                                    fma(l1,
                                        _cse2 *powr<3>(cosl1p1) *
                                                ((5. + _cse38) * _cse9 - 3. * _cse10 * cosp1p2 +
                                                 _cse3 * (5. + _cse41) * p1 +
                                                 _cse2 * (6. + _cse38) * cosp1p2 * p2) +
                                            cosl1p1 *
                                                (_cse30 * _cse39 +
                                                 _cse3 *
                                                     (-10. + _cse21 + _cse40 +
                                                      (15. - 19. * _cse1) * _cse5) *
                                                     _cse9 +
                                                 _cse42 * (-2. - 4. * _cse5) * cosp1p2 +
                                                 _cse10 * _cse2 *
                                                     (-12. + 8. * _cse1 + (16. + _cse41) * _cse5) *
                                                     cosp1p2 +
                                                 _cse20 * (-5. + _cse38 + (15. + _cse24) * _cse5) *
                                                     p1 +
                                                 (-10. + 16. * _cse1 + _cse44) * _cse7 * cosp1p2 *
                                                     p2) +
                                            cosl1p2 *
                                                (_cse10 * _cse2 * (-10. + _cse43) +
                                                 _cse42 * (-5. + _cse43) +
                                                 4. * _cse10 * _cse2 * _cse6 +
                                                 _cse8 * (8. * _cse3 * _cse9 +
                                                          _cse20 * (16. + _cse45) * p1) +
                                                 cosp1p2 * (-2. * _cse39 +
                                                            _cse3 * (-12. - 3. * _cse5) * _cse9 +
                                                            _cse20 * (-10. + 6. * _cse5) * p1) -
                                                 5. * _cse7 * p2 +
                                                 _cse1 * (_cse10 * _cse2 * (10. + _cse44) +
                                                          _cse42 * (7. + _cse45) - _cse7 * p2)) +
                                            _cse4 * cosl1p2 * p1 *
                                                (-7. * _cse2 * _cse3 * _cse8 +
                                                 (-7. * _cse20 + 16. * _cse2 * _cse3 - 4. * _cse7) *
                                                     cosp1p2 +
                                                 15. * _cse10 * p1 + 15. * _cse9 * p2 +
                                                 _cse1 * (-19. * _cse10 * p1 - 13. * _cse9 * p2)),
                                        0.))))))))) *
    fma(1. + 1. * _cse46, _interp1 *_interp2,
        fma(1. + _cse46, _interp3 *_interp4,
            fma(_cse46, _interp2 * (-50. * _interp4 + 50. * _interp5), 0.))) *
    powr<-1>(fma(_cse50, _interp10, fma(_interp4, _interp9, 0.))) *
    powr<-2>(fma(_interp2, _interp4, fma(_cse28, _interp6, 0.)));
        // clang-format on
      }
      { // subkernel 3
        const auto _interp8 = ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
        const auto _interp12 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 + (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                    2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
              atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    -1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))); // clang-format off

const auto _interp16 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
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
        const auto _cse4 = powr<2>(cosp1p2);
        const auto _cse5 = powr<2>(p2);
        const auto _cse6 = powr<2>(_cse5);
        const auto _cse7 = powr<2>(_cse3);
        const auto _cse8 = powr<3>(p1);
        const auto _cse9 = -2. * _cse4;
        const auto _cse10 = powr<3>(p2);
        const auto _cse11 = powr<2>(_cse1);
        const auto _cse12 = -3. * _cse4;
        const auto _cse13 = 3. + _cse12;
        const auto _cse14 = -11. * _cse4;
        const auto _cse15 = 11. + _cse14;
        const auto _cse16 = powr<5>(p1);
        const auto _cse17 = powr<2>(_cse4);
        const auto _cse18 = powr<3>(_cse3);
        const auto _cse19 = 14. * _cse4;
        const auto _cse20 = -10. * _cse4;
        const auto _cse21 = powr<5>(p2);
        const auto _cse22 = powr<3>(_cse5);
        const auto _cse23 = powr<3>(l1);
        const auto _cse24 = -3. * _cse11;
        const auto _cse25 = powr<3>(cosp1p2);
        const auto _cse26 = -6. * _cse3;
        const auto _cse27 = -14. * _cse7;
        const auto _cse28 = powr<2>(cosl1p2);
        const auto _cse29 = 13. * _cse4;
        const auto _cse30 = powr<3>(_cse1);
        const auto _cse31 = powr<5>(l1);
        const auto _cse32 = -12. * _cse4;
        const auto _cse33 = -26. * _cse28;
        const auto _cse34 = -52. * _cse4;
        const auto _cse35 = -13. * _cse17;
        const auto _cse36 = 16. * _cse3 * _cse6;
        const auto _cse37 = 3. * _cse22;
        const auto _cse38 = powr<4>(_cse3);
        const auto _cse39 = -6. * _cse4;
        const auto _cse40 = powr<2>(_cse28);
        const auto _cse41 = 3. * _cse4;
        const auto _cse42 = -3. + _cse41;
        const auto _cse43 = 12. * _cse4;
        const auto _cse44 = 26. * _cse4;
        const auto _cse45 = 13. * _cse17;
        const auto _cse46 = -11. + _cse45 + _cse9;
        const auto _cse47 = -12. + _cse43;
        const auto _cse48 = powr<7>(p1);
        const auto _cse49 = 2. * _cse28;
        const auto _cse50 = 24. * _cse4;
        const auto _cse51 = 2. * _cse17; // clang-format off
_acc += 18. * powr<-1>(_cse1 + _cse2 + _cse3) * powr<-1>(-1. + _cse4) * _interp12 * _interp14
    * _interp16 *powr<-1>(1. + powr<6>(k)) *
    powr<-1>(fma(-2., cosl1p2 * l1 * p2,
                 fma(2., cosp1p2 * p1 * p2, _cse1 + _cse2 + _cse3 + _cse5))) *
    powr<-1>(fma(_cse3, (8. + _cse4) * _cse5,
                 fma(3., _cse6,
                     fma(3., _cse7,
                         fma(6., _cse10 * cosp1p2 * p1, fma(6., _cse8 * cosp1p2 * p2, 0.)))))) *
    fma(_cse31, _cse6 *powr<5>(cosl1p2) * (-cosp1p2 * p1 - p2),
        fma(_cse10,
            _cse11 * _cse40 *
                (6. * _cse10 + cosp1p2 * (7. * _cse8 + 6. * _cse1 * p1 + 14. * _cse5 * p1) +
                 6. * _cse1 * p2 + 11. * _cse3 * p2 + 4. * _cse3 * _cse4 * p2),
            fma(_cse23,
                _cse5 *powr<3>(cosl1p2) *
                    (-4. * _cse25 * _cse5 * _cse8 +
                     (_cse24 + _cse1 * (_cse26 - 23. * _cse5) - 41. * _cse3 * _cse5 - 18. * _cse6 -
                      5. * _cse7) *
                         cosp1p2 * p1 +
                     _cse3 * _cse4 * (-11. * _cse1 - 17. * _cse3 - 25. * _cse5) * p2 +
                     (_cse24 + _cse1 * (-11. * _cse3 - 7. * _cse5) - 11. * _cse3 * _cse5 -
                      3. * _cse6 - 12. * _cse7) *
                         p2),
                fma(cosl1p2,
                    l1 *(_cse10 * _cse17 * (33. * _cse1 + 31. * _cse3 + 26. * _cse5) * _cse7 +
                         _cse25 * _cse5 *
                             (34. * _cse11 + 65. * _cse3 * _cse5 +
                              _cse1 * (88. * _cse3 + 69. * _cse5) + 18. * _cse6 + 44. * _cse7) *
                             _cse8 +
                         5. * _cse16 * _cse6 * powr<5>(cosp1p2) +
                         (-3. * _cse22 * _cse3 + 3. * _cse38 - 8. * _cse18 * _cse5 +
                          _cse30 * (3. * _cse3 + 6. * _cse5) - 16. * _cse6 * _cse7 +
                          _cse11 * (18. * _cse3 * _cse5 + 27. * _cse6 + 11. * _cse7) +
                          _cse1 * (11. * _cse18 + 15. * _cse22 + 26. * _cse3 * _cse6 +
                                   2. * _cse5 * _cse7)) *
                             cosp1p2 * p1 +
                         _cse3 * _cse4 *
                             (21. * _cse18 + 6. * _cse30 + _cse36 + _cse37 +
                              _cse11 * (39. * _cse3 + 44. * _cse5) + 37. * _cse5 * _cse7 +
                              _cse1 * (97. * _cse3 * _cse5 + 43. * _cse6 + 58. * _cse7)) *
                             p2 +
                         (-6. * _cse38 - 14. * _cse18 * _cse5 + 3. * _cse30 * _cse5 +
                          _cse11 * (5. * _cse3 * _cse5 + 8. * _cse6 - 6. * _cse7) -
                          6. * _cse6 * _cse7 +
                          _cse1 *
                              (-14. * _cse18 + _cse37 + 5. * _cse3 * _cse6 - 14. * _cse5 * _cse7)) *
                             p2),
                    fma(_cse11, _cse16 *powr<6>(cosl1p1) * (-2. * p1 - 2. * cosp1p2 * p2),
                        fma(_cse1,
                            _cse28 * p2 *
                                (-6. * _cse10 * _cse11 + 3. * _cse10 * _cse7 -
                                 2. * _cse10 * _cse17 * _cse7 +
                                 _cse25 * (-14. * _cse1 + _cse26 - _cse5) * _cse5 * _cse8 +
                                 (-6. * _cse18 + _cse37 + _cse11 * (_cse26 - 9. * _cse5) +
                                  _cse1 * (_cse27 - 20. * _cse3 * _cse5 - 11. * _cse6) +
                                  10. * _cse3 * _cse6 + 6. * _cse5 * _cse7) *
                                     cosp1p2 * p1 +
                                 5. * _cse18 * p2 +
                                 _cse3 * _cse4 *
                                     (-9. * _cse11 + _cse27 + _cse1 * (-31. * _cse3 - 13. * _cse5) +
                                      8. * _cse3 * _cse5 + 6. * _cse6) *
                                     p2 +
                                 _cse1 * (-6. * _cse21 - 12. * _cse10 * _cse3 + 3. * _cse7 * p2)),
                            fma(_cse23,
                                _cse7 *powr<5>(cosl1p1) *
                                    (_cse8 * (15. + _cse9) + 2. * _cse10 * cosp1p2 +
                                     (14. + _cse4) * _cse5 * p1 +
                                     _cse3 * (28. + _cse9) * cosp1p2 * p2 +
                                     _cse1 * (13. * p1 + 13. * cosp1p2 * p2) +
                                     cosl1p2 * l1 *
                                         (2. * _cse3 * cosp1p2 - 5. * _cse5 * cosp1p2 -
                                          5. * p1 * p2 + 2. * _cse4 * p1 * p2)),
                                fma(powr<2>(cosl1p1),
                                    p1 *(
                                        _cse30 * ((9. + _cse41) * _cse8 + 15. * _cse10 * cosp1p2 +
                                                  (-3. + 30. * _cse4) * _cse5 * p1 +
                                                  _cse3 * (21. + _cse41) * cosp1p2 * p2) +
                                        _cse8 * (_cse18 * _cse42 + _cse22 * _cse42 +
                                                 _cse3 * _cse46 * _cse6 + _cse46 * _cse5 * _cse7 +
                                                 _cse10 * (-28. + _cse44 + _cse51) * _cse8 *
                                                     cosp1p2 +
                                                 _cse21 * _cse47 * cosp1p2 * p1 +
                                                 _cse16 * _cse47 * cosp1p2 * p2) +
                                        _cse11 *
                                            ((14. + 7. * _cse17 + 117. * _cse4) * _cse5 * _cse8 +
                                             _cse16 * (30. + _cse9) + 27. * _cse21 * cosp1p2 +
                                             _cse10 * _cse3 * (59. + 77. * _cse4) * cosp1p2 +
                                             (-10. + 91. * _cse4) * _cse6 * p1 +
                                             (88. - 4. * _cse4) * _cse7 * cosp1p2 * p2) +
                                        _cse10 * _cse11 * _cse40 *
                                            (4. * _cse3 * cosp1p2 - _cse5 * cosp1p2 + 4. * p1 * p2 -
                                             _cse4 * p1 * p2) +
                                        _cse23 * _cse5 * powr<3>(cosl1p2) *
                                            (2. * _cse25 * _cse3 * _cse5 +
                                             (-35. * _cse3 * _cse5 +
                                              _cse1 * (-27. * _cse3 + 14. * _cse5) + 6. * _cse6 -
                                              16. * _cse7) *
                                                 cosp1p2 +
                                             (-25. * _cse1 - 17. * _cse3 - 11. * _cse5) * p1 * p2 +
                                             _cse4 * (12. * _cse1 - 27. * _cse3 + 12. * _cse5) *
                                                 p1 * p2) +
                                        _cse1 * _cse28 * p2 *
                                            (_cse25 * _cse3 * (-21. * _cse1 + 52. * _cse3) * _cse5 +
                                             _cse10 * _cse17 * _cse8 +
                                             (18. * _cse18 - 3. * _cse22 +
                                              _cse11 * (18. * _cse3 - 18. * _cse5) +
                                              11. * _cse5 * _cse7 +
                                              _cse1 * (13. * _cse3 * _cse5 - 23. * _cse6 +
                                                       39. * _cse7)) *
                                                 cosp1p2 +
                                             (6. * _cse11 + _cse27 +
                                              _cse1 * (8. * _cse3 - 13. * _cse5) -
                                              31. * _cse3 * _cse5 - 9. * _cse6) *
                                                 p1 * p2 +
                                             _cse4 *
                                                 (-6. * _cse11 +
                                                  _cse1 * (72. * _cse3 - 59. * _cse5) +
                                                  72. * _cse3 * _cse5 - 6. * _cse6 + 65. * _cse7) *
                                                 p1 * p2) +
                                        cosl1p2 * l1 *
                                            (_cse25 * _cse3 * _cse5 *
                                                 (-_cse11 - 78. * _cse3 * _cse5 +
                                                  _cse1 * (-40. * _cse3 + 11. * _cse5) -
                                                  12. * _cse6 - 62. * _cse7) +
                                             _cse10 * _cse17 *
                                                 (10. * _cse1 - 31. * _cse3 - 13. * _cse5) * _cse8 +
                                             (21. * _cse22 * _cse3 - 3. * _cse38 +
                                              50. * _cse18 * _cse5 +
                                              _cse30 * (-3. * _cse3 + 3. * _cse5) +
                                              86. * _cse6 * _cse7 +
                                              _cse11 *
                                                  (40. * _cse3 * _cse5 - 11. * _cse6 + 2. * _cse7) +
                                              _cse1 * (2. * _cse18 - 9. * _cse22 +
                                                       67. * _cse3 * _cse6 + 92. * _cse5 * _cse7)) *
                                                 cosp1p2 -
                                             2. * _cse6 * _cse7 * powr<5>(cosp1p2) +
                                             _cse4 *
                                                 (-30. * _cse18 - 3. * _cse22 - 3. * _cse30 +
                                                  _cse11 * (-14. * _cse3 - 15. * _cse5) -
                                                  14. * _cse3 * _cse6 +
                                                  _cse1 * (-17. * _cse3 * _cse5 - 15. * _cse6 -
                                                           33. * _cse7) -
                                                  33. * _cse5 * _cse7) *
                                                 p1 * p2 +
                                             (21. * _cse18 + 6. * _cse22 + 3. * _cse30 +
                                              _cse11 * (16. * _cse3 + 43. * _cse5) +
                                              39. * _cse3 * _cse6 + 58. * _cse5 * _cse7 +
                                              _cse1 * (97. * _cse3 * _cse5 + 44. * _cse6 +
                                                       37. * _cse7)) *
                                                 p1 * p2) +
                                        _cse1 *
                                            (_cse16 * (22. - 5. * _cse17 + 88. * _cse4) * _cse5 +
                                             (1. + 14. * _cse17 + 117. * _cse4) * _cse6 * _cse8 +
                                             _cse48 * (14. + _cse9) +
                                             _cse21 * _cse3 * (32. + 43. * _cse4) * cosp1p2 +
                                             _cse10 * (76. - _cse17 + 72. * _cse4) * _cse7 *
                                                 cosp1p2 +
                                             _cse22 * (-3. + 30. * _cse4) * p1 +
                                             _cse18 * (54. + _cse39) * cosp1p2 * p2 +
                                             6. * cosp1p2 * powr<7>(p2))),
                                    fma(_cse3,
                                        _cse1 *(_cse15 *_cse18 + _cse13 * _cse22 +
                                                (21. - 35. * _cse17 + _cse19) * _cse3 * _cse6 +
                                                (29. - 51. * _cse17 + 22. * _cse4) * _cse5 * _cse7 +
                                                _cse10 * (72. - 14. * _cse17 - 58. * _cse4) *
                                                    _cse8 * cosp1p2 +
                                                _cse21 * (20. - 20. * _cse4) * cosp1p2 * p1 +
                                                _cse16 * (44. - 44. * _cse4) * cosp1p2 * p2) +
                                            _cse11 * (_cse3 * (21. - 22. * _cse17 + _cse4) * _cse5 +
                                                      (10. + _cse20) * _cse6 + _cse15 * _cse7 +
                                                      _cse10 * (31. - 31. * _cse4) * cosp1p2 * p1 +
                                                      (33. - 33. * _cse4) * _cse8 * cosp1p2 * p2) +
                                            _cse30 * (_cse13 * _cse3 + _cse13 * _cse5 +
                                                      (6. + _cse39) * cosp1p2 * p1 * p2) +
                                            p1 * (_cse13 * _cse48 +
                                                  _cse16 * (11. - 25. * _cse17 + _cse19) * _cse5 +
                                                  (11. - 39. * _cse17 + 30. * _cse4 -
                                                   2. * powr<3>(_cse4)) *
                                                      _cse6 * _cse8 +
                                                  _cse21 * _cse3 * (23. + _cse20 + _cse35) *
                                                      cosp1p2 +
                                                  _cse10 * (39. - 15. * _cse17 - 24. * _cse4) *
                                                      _cse7 * cosp1p2 +
                                                  _cse22 * (3. - 12. * _cse17 + 9. * _cse4) * p1 +
                                                  _cse18 * (15. - 15. * _cse4) * cosp1p2 * p2 +
                                                  _cse13 * cosp1p2 * powr<7>(p2)),
                                        fma(_cse1,
                                            _cse8 *powr<4>(cosl1p1) *
                                                (_cse11 * (-12. * p1 - 12. * cosp1p2 * p2) +
                                                 _cse1 * ((-39. + _cse29) * _cse8 +
                                                          _cse10 * (5. - 6. * _cse28) * cosp1p2 +
                                                          (-35. + _cse19 + _cse28 * (-2. + _cse4)) *
                                                              _cse5 * p1 +
                                                          (-65. + 5. * _cse28 + _cse29) * _cse3 *
                                                              cosp1p2 * p2) +
                                                 p1 *
                                                     (_cse3 * (-51. + 5. * _cse17 - 5. * _cse4) *
                                                          _cse5 +
                                                      (-22. + 7. * _cse4) * _cse6 +
                                                      (-25. + _cse29) * _cse7 +
                                                      _cse10 * (-58. + 16. * _cse4) * cosp1p2 * p1 +
                                                      (-62. + _cse44) * _cse8 * cosp1p2 * p2) +
                                                 _cse23 * cosl1p2 *
                                                     (-13. * _cse3 * cosp1p2 +
                                                      26. * _cse5 * cosp1p2 + 26. * p1 * p2 -
                                                      13. * _cse4 * p1 * p2) +
                                                 cosl1p2 * l1 *
                                                     (-6. * _cse25 * _cse3 * _cse5 +
                                                      (53. * _cse3 * _cse5 + 9. * _cse6 -
                                                       13. * _cse7) *
                                                          cosp1p2 +
                                                      33. * _cse10 * p1 + 31. * _cse8 * p2 +
                                                      _cse4 *
                                                          (10. * _cse10 * p1 - 31. * _cse8 * p2))),
                                            fma(_cse3,
                                                powr<3>(cosl1p1) * l1 *
                                                    (_cse30 * (3. * p1 + 3. * cosp1p2 * p2) +
                                                     _cse11 *
                                                         ((10. + _cse32) * _cse8 +
                                                          _cse10 * (-21. + 27. * _cse28) * cosp1p2 +
                                                          (20. + _cse28 - 43. * _cse4) * _cse5 *
                                                              p1 +
                                                          _cse3 * (8. + _cse32 + _cse33) * cosp1p2 *
                                                              p2) +
                                                     p1 * ((6. + _cse12) * _cse22 +
                                                           _cse18 * (15. + _cse32) +
                                                           _cse3 * (33. + _cse35 + 4. * _cse4) *
                                                               _cse6 +
                                                           (44. - 26. * _cse17 + 6. * _cse4) *
                                                               _cse5 * _cse7 +
                                                           _cse10 * (84. - 2. * _cse17 + _cse34) *
                                                               _cse8 * cosp1p2 +
                                                           _cse21 * (24. + _cse32) * cosp1p2 * p1 +
                                                           _cse16 * (48. - 36. * _cse4) * cosp1p2 *
                                                               p2) +
                                                     _cse1 *
                                                         (_cse16 * (24. - 26. * _cse4) +
                                                          (58. - 16. * _cse17 +
                                                           _cse28 * (6. + _cse34) - 72. * _cse4) *
                                                              _cse5 * _cse8 +
                                                          _cse21 * (-22. + 13. * _cse28) * cosp1p2 +
                                                          _cse10 * _cse3 *
                                                              (24. - 74. * _cse4 +
                                                               _cse28 * (4. + _cse9)) *
                                                              cosp1p2 +
                                                          (31. - 77. * _cse4 +
                                                           _cse28 * (14. + 21. * _cse4)) *
                                                              _cse6 * p1 +
                                                          (46. + _cse33 + _cse34) * _cse7 *
                                                              cosp1p2 * p2) +
                                                     _cse31 * cosl1p2 *
                                                         (12. * _cse3 * cosp1p2 -
                                                          18. * _cse5 * cosp1p2 - 18. * p1 * p2 +
                                                          12. * _cse4 * p1 * p2) +
                                                     cosl1p2 * l1 *
                                                         (_cse25 * (_cse36 + 72. * _cse5 * _cse7) -
                                                          88. * _cse10 * _cse8 +
                                                          6. * _cse10 * _cse17 * _cse8 +
                                                          (12. * _cse18 - 3. * _cse22 -
                                                           76. * _cse3 * _cse6 -
                                                           78. * _cse5 * _cse7) *
                                                              cosp1p2 -
                                                          34. * _cse21 * p1 - 44. * _cse16 * p2 +
                                                          _cse4 *
                                                              (40. * _cse10 * _cse8 + _cse21 * p1 +
                                                               62. * _cse16 * p2)) +
                                                     _cse23 * cosl1p2 *
                                                         (16. * _cse25 * _cse3 * _cse5 +
                                                          ((-102. + 6. * _cse28) * _cse3 * _cse5 +
                                                           (-7. - 4. * _cse28) * _cse6 +
                                                           26. * _cse7) *
                                                              cosp1p2 +
                                                          _cse10 * (-69. + 4. * _cse28) * p1 -
                                                          65. * _cse8 * p2 +
                                                          _cse4 *
                                                              (_cse10 * (-11. - 2. * _cse28) * p1 +
                                                               78. * _cse8 * p2))),
                                                fma(cosl1p1,
                                                    powr<7>(l1) * (-3. * _cse8 +
                                                                   _cse10 * (-3. + 3. * _cse28) *
                                                                       cosp1p2 +
                                                                   (_cse39 + _cse28 * _cse42) *
                                                                       _cse5 * p1 +
                                                                   (-6. - 3. * _cse28) * _cse3 *
                                                                       cosp1p2 * p2) +
                                                        _cse8 * cosl1p2 * p2 *
                                                            (_cse18 * _cse42 + _cse22 * _cse42 +
                                                             _cse3 * _cse46 * _cse6 +
                                                             _cse46 * _cse5 * _cse7 +
                                                             _cse10 * (-28. + _cse44 + _cse51) *
                                                                 _cse8 * cosp1p2 +
                                                             _cse21 * _cse47 * cosp1p2 * p1 +
                                                             _cse16 * _cse47 * cosp1p2 * p2) +
                                                        _cse31 *
                                                            (_cse16 * (-23. + _cse43) +
                                                             (-20. - 10. * _cse28 - 32. * _cse4) *
                                                                 _cse5 * _cse8 +
                                                             _cse10 * _cse3 *
                                                                 (-27. + _cse28 * (2. + _cse29) -
                                                                  22. * _cse4 - 14. * _cse40) *
                                                                 cosp1p2 +
                                                             _cse21 * (-8. + 7. * _cse28 + _cse40) *
                                                                 cosp1p2 +
                                                             (-27. * _cse4 +
                                                              _cse28 * (11. + 23. * _cse4) +
                                                              (-14. + _cse4) * _cse40) *
                                                                 _cse6 * p1 +
                                                             (-57. + _cse49 + _cse50) * _cse7 *
                                                                 cosp1p2 * p2) +
                                                        _cse23 *
                                                            ((-39. + 28. * _cse4) * _cse48 +
                                                             _cse16 *
                                                                 (-72. + 58. * _cse17 +
                                                                  (-6. + _cse14) * _cse28 -
                                                                  76. * _cse4) *
                                                                 _cse5 +
                                                             (-31. - 5. * _cse17 +
                                                              _cse28 * (20. + 6. * _cse17 -
                                                                        13. * _cse4) -
                                                              59. * _cse4 +
                                                              (-7. - 4. * _cse4) * _cse40) *
                                                                 _cse6 * _cse8 +
                                                             _cse21 * _cse3 *
                                                                 (-27. - 21. * _cse4 +
                                                                  _cse28 * (2. + 27. * _cse4) -
                                                                  14. * _cse40) *
                                                                 cosp1p2 +
                                                             _cse10 *
                                                                 (-142. +
                                                                  _cse28 * (-15. + 4. * _cse4) -
                                                                  3. * _cse40 + _cse50 + _cse51) *
                                                                 _cse7 * cosp1p2 +
                                                             _cse22 *
                                                                 (-15. * _cse4 +
                                                                  _cse28 * (9. + 18. * _cse4) -
                                                                  6. * _cse40) *
                                                                 p1 +
                                                             _cse18 *
                                                                 (-128. + 84. * _cse4 + _cse49) *
                                                                 cosp1p2 * p2 +
                                                             (-3. + 3. * _cse28) * cosp1p2 *
                                                                 powr<7>(p2)) +
                                                        _cse3 * l1 *
                                                            ((-15. + _cse43) * _cse48 +
                                                             _cse16 *
                                                                 (-44. + 62. * _cse17 +
                                                                  _cse28 * (6. - 18. * _cse4) -
                                                                  54. * _cse4) *
                                                                 _cse5 +
                                                             (-33. + 65. * _cse17 +
                                                              _cse28 * (14. - 5. * _cse17 -
                                                                        39. * _cse4) -
                                                              88. * _cse4 + 2. * powr<3>(_cse4)) *
                                                                 _cse6 * _cse8 +
                                                             _cse21 * _cse3 *
                                                                 (-57. +
                                                                  _cse28 * (2. - 26. * _cse4) +
                                                                  8. * _cse4 + _cse45) *
                                                                 cosp1p2 +
                                                             _cse10 *
                                                                 (-128. + 28. * _cse17 +
                                                                  _cse28 * (2. - 26. * _cse4) +
                                                                  46. * _cse4) *
                                                                 _cse7 * cosp1p2 +
                                                             _cse22 *
                                                                 (-6. + 12. * _cse17 +
                                                                  _cse28 * (6. - 18. * _cse4) -
                                                                  21. * _cse4) *
                                                                 p1 +
                                                             _cse18 *
                                                                 (-63. - 3. * _cse28 +
                                                                  48. * _cse4) *
                                                                 cosp1p2 * p2 +
                                                             (-6. - 3. * _cse28 + _cse41) *
                                                                 cosp1p2 * powr<7>(p2)) +
                                                        _cse11 * cosl1p2 *
                                                            (_cse25 * ((-7. - 4. * _cse28) * _cse3 *
                                                                           _cse6 -
                                                                       76. * _cse5 * _cse7) -
                                                             9. * _cse10 * _cse17 * _cse8 +
                                                             _cse10 * (-26. + 41. * _cse28) *
                                                                 _cse8 +
                                                             (-28. * _cse18 +
                                                              _cse22 * (6. - 6. * _cse28) +
                                                              _cse3 *
                                                                  (-49. + 68. * _cse28 + _cse40) *
                                                                  _cse6 +
                                                              (-34. + 27. * _cse28) * _cse5 *
                                                                  _cse7) *
                                                                 cosp1p2 +
                                                             _cse21 *
                                                                 (-27. + 23. * _cse28 + _cse40) *
                                                                 p1 +
                                                             16. * _cse16 * p2 +
                                                             _cse4 *
                                                                 (_cse10 * (-67. + 35. * _cse28) *
                                                                      _cse8 +
                                                                  _cse21 * (11. - 14. * _cse28) *
                                                                      p1 -
                                                                  86. * _cse16 * p2)) +
                                                        _cse30 * cosl1p2 *
                                                            (-3. * _cse25 * _cse3 * _cse5 +
                                                             ((-15. + 18. * _cse28) * _cse3 *
                                                                  _cse5 +
                                                              (6. - 6. * _cse28) * _cse6 -
                                                              12. * _cse7) *
                                                                 cosp1p2 +
                                                             _cse10 * (-15. + 18. * _cse28) * p1 +
                                                             3. * _cse8 * p2 +
                                                             _cse4 *
                                                                 (_cse10 * (9. - 6. * _cse28) * p1 -
                                                                  21. * _cse8 * p2)) +
                                                        _cse1 * cosl1p2 * p1 *
                                                            (_cse21 * (-18. + 6. * _cse28) * _cse3 +
                                                             _cse10 * (-2. + 5. * _cse28) * _cse7 +
                                                             _cse17 * (-26. * _cse21 * _cse3 -
                                                                       53. * _cse10 * _cse7) -
                                                             5. * _cse6 * _cse8 * powr<5>(cosp1p2) +
                                                             _cse25 * (-78. * _cse16 * _cse5 +
                                                                       (-102. + 6. * _cse28) *
                                                                           _cse6 * _cse8 -
                                                                       18. * _cse22 * p1) +
                                                             cosp1p2 *
                                                                 (-12. * _cse48 +
                                                                  _cse16 * (-18. + 6. * _cse28) *
                                                                      _cse5 +
                                                                  (-34. + 27. * _cse28) * _cse6 *
                                                                      _cse8 +
                                                                  _cse22 * (-15. + 18. * _cse28) *
                                                                      p1) +
                                                             8. * _cse18 * p2 +
                                                             (-6. + 3. * _cse28) * powr<7>(p2) +
                                                             _cse4 *
                                                                 (_cse21 * (-40. + 27. * _cse28) *
                                                                      _cse3 +
                                                                  _cse10 * (-92. + 16. * _cse28) *
                                                                      _cse7 -
                                                                  50. * _cse18 * p2 -
                                                                  3. * powr<7>(p2))),
                                                    0.)))))))))))) *
    powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.))) *
    fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k),
        fma(_interp3, _interp4 * (1. + powr<6>(k)),
            fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) *
    powr<-1>(fma(_interp4, _interp7, fma(_cse1 + _cse2 + _cse3, _interp8, 0.))) *
    powr<-1>(fma(
        _interp4, _interp9,
        fma(_interp10,
            _cse1 + _cse2 + _cse3 + _cse5 - 2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2, 0.)));
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
using DiFfRG::ZA3_kernel;