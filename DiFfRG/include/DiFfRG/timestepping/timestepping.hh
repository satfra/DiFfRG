#pragma once

/**
 * @file timestepping.hh
 * @brief Includes all timesteppers and linear solvers; see abstract_timestepper.hh for an overview.
 */

#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/PETScDirect.hh>
#include <DiFfRG/timestepping/linear_solver/PETScKrylov.hh>
#include <DiFfRG/timestepping/linear_solver/ScaledGMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>

#include <DiFfRG/timestepping/boost_abm.hh>
#include <DiFfRG/timestepping/boost_rk.hh>
#include <DiFfRG/timestepping/explicit_euler.hh>
#include <DiFfRG/timestepping/implicit_euler.hh>
#include <DiFfRG/timestepping/rk.hh>
#include <DiFfRG/timestepping/sundials_ida.hh>
#include <DiFfRG/timestepping/sundials_ida_boost_abm.hh>
#include <DiFfRG/timestepping/sundials_ida_boost_rk.hh>
#include <DiFfRG/timestepping/trbdf2.hh>
