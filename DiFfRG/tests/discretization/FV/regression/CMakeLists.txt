# ##############################################################################
# Regression tests
# ##############################################################################

setup_test(gross_neveu_kt_regression.cc)
setup_test(burgers_2D.cc)
setup_test(buckley_leverett_2D.cc)
setup_test(heat_diffusion_2D.cc)
setup_test(linear_advection_2D.cc)
setup_test(manufactured_advection_diffusion_2D.cc)
setup_test(on_model_kt_regression.cc)
setup_test(rotating_gaussian_2D.cc)
add_subdirectory(on_model_kt_3D)
add_subdirectory(o2_model_kt_2D)
add_subdirectory(zero_dim_2d_kt)
setup_test(quark_meson_kt_regression.cc)
