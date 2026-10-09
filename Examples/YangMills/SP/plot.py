#!/usr/bin/env python3
import math
import os
import sys
import traceback

import h5py
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# rcParams
# ----------------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

plt.rcParams["font.size"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.minor.width"] = 0.6
plt.rcParams["ytick.minor.width"] = 0.6

# The HDF5 file holds everything this model writes: the scalars and the
# momentum-grid maps (there are no FE functions here -- YangMills/SP carries
# only Variables). The run writes into build/, a finished run is often copied
# next to this script -- take whichever of the two is newer.
_candidates = [f for f in ("./output.h5", "./build/output.h5") if os.path.exists(f)]
main_file = max(_candidates, key=os.path.getmtime)
figdir = "./figs"
os.makedirs(figdir, exist_ok=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
# model.hh::readouts writes exactly
#   scalars : k, m2A, ZA_pmin, Zc_pmin              (+ "time" from the stepper)
#   maps    : ZA, Zc, ZA3, ZAcbc, ZA4, dtZA, dtZc   (96 points, focused-log p grid)
#
# The grid is FocusedLogCoordinates1D, i.e. logarithmic but with the points clustered
# around /discretization/p_grid_center. Nothing here assumes an even spacing -- the
# momenta are read from the file -- so the same script serves a plain log grid.
#
# scalars : dict name => 1d array, rows = flow-time steps
# maps    : dict name => dict(p, data, keys); p/data are lists over flow steps
#           of (p-grid, values). Access final k via  maps[name]["data"][-1]

def _first_coord(coords):
    """Extract the first coordinate component (the momentum grid p)."""
    coords = np.asarray(coords)
    if coords.dtype.names:  # compound / structured datatype
        return np.asarray(coords[coords.dtype.names[0]], dtype=float).ravel()
    if coords.ndim >= 2:
        return np.asarray(coords[:, 0], dtype=float).ravel()
    return np.array([np.ravel(c)[0] for c in coords], dtype=float)


def read_map(file, name):
    g = file["maps"][name]
    ks = sorted(g.keys(), key=lambda x: int(x))
    p = [_first_coord(g[k]["coordinates"][()]) for k in ks]
    d = [np.asarray(g[k]["data"][()], dtype=float).ravel() for k in ks]
    return {"p": p, "data": d, "keys": ks}


def reload_data():
    with h5py.File(main_file, "r") as file:
        print("top-level:", list(file.keys()))
        sg = file["scalars"]
        time = np.asarray(sg["time"][()]).ravel()
        n = len(time)
        scalars = {s: np.asarray(sg[s][()]).ravel()[:n] for s in sg.keys()}
        maps = {s: read_map(file, s) for s in file["maps"].keys()}
    print("file:   ", main_file)
    print("scalars:", sorted(scalars.keys()))
    print("maps:   ", sorted(maps.keys()))
    return scalars, maps


scalars, maps = reload_data()

kIR = round(float(scalars["k"][-1]), 4)


def C(i):
    return plt.cm.tab10(i)


def mp(name, idx=-1):
    """(p-grid, values) of a map at flow step idx (default: the IR end)."""
    return maps[name]["p"][idx], maps[name]["data"][idx]


def logscale(ax, *arrays, axis="y"):
    """Log axis if the data is strictly positive, symlog if it crosses zero.

    A run that has left the physical branch (m_A^2 < 0, Z_c < 0 past the
    separatrix) must still be plottable -- that is usually exactly the run one
    wants to look at -- and a plain log scale silently drops those points.
    """
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    if (v > 0).all():
        ax.set_yscale("log") if axis == "y" else ax.set_xscale("log")
        return
    nz = np.abs(v[v != 0])
    lt = float(nz.min()) if nz.size else 1e-12
    if axis == "y":
        ax.set_yscale("symlog", linthresh=lt)
    else:
        ax.set_xscale("symlog", linthresh=lt)


# ----------------------------------------------------------------------------
# Figure : propagator dressings at the IR scale.
# The gluon is shown as Z_A^{-1}(p), i.e. the propagator dressing itself; the
# ghost dressing Z_c(p) is plotted log-log (it runs over decades and vanishes
# towards the scaling IR).
# ----------------------------------------------------------------------------
def figure_dressings():
    pZA, dZA = mp("ZA")
    pZc, dZc = mp("Zc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=600)

    ax1.plot(pZA, dZA ** -1, color=C(3))
    ax1.scatter(pZA, dZA ** -1, color=C(3), s=0.9)
    ax1.set_ylabel(r"$Z_A^{-1}(p)$")
    # median of the p grid: equally many momentum points to either side
    ax1.axvline(np.median(pZA), color="0.5", lw=0.6, ls="--", alpha=0.6, zorder=0)

    ax2.plot(pZc, dZc, color=C(0))
    ax2.scatter(pZc, dZc, color=C(0), s=0.9)
    ax2.set_ylabel(r"$Z_c(p)$")
    logscale(ax2, dZc)
    # Z_c may span well under a decade -- the default log locator then leaves a
    # single 10^n tick, so label the minor decades as plain numbers.
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.tick_params(axis="y", which="minor", labelsize=8)

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$p\textrm{ [GeV]}$")
        ax.set_xscale("log")

    fig.suptitle(rf"Propagator dressings at $k = {kIR}$ GeV")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dressings.png"))
    plt.close(fig)
    print("wrote dressings.png")


# ----------------------------------------------------------------------------
# Figure : the gluon propagator 1/(Z_A p^2) and its dimensionless form, at the
# IR scale. This is the object that carries the mass gap: the flat plateau at
# small p is m_A^{-2}.
# ----------------------------------------------------------------------------
def figure_propagator():
    p, dZA = mp("ZA")
    G = 1.0 / (dZA * p ** 2)

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=600)
    ax.plot(p, G, color=C(3))
    ax.scatter(p, G, color=C(3), s=0.9)

    # the IR plateau 1/m_A^2 implied by the read-out mass at the smallest grid p
    m2A = float(scalars["m2A"][-1])
    if m2A > 0:
        ax.axhline(1.0 / m2A, color="0.5", lw=0.6, ls="--", alpha=0.8, zorder=0)
        ax.text(p[0], 1.0 / m2A, rf"$1/m_A^2,\ m_A^2 = {m2A:.4g}$",
                fontsize=8, va="bottom", ha="left", color="0.4")

    ax.set_xlabel(r"$p\textrm{ [GeV]}$")
    ax.set_ylabel(r"$G_A(p) = 1/(Z_A(p)\,p^2)$")
    ax.set_xscale("log")
    logscale(ax, G)
    ax.set_title(rf"Gluon propagator at $k = {kIR}$ GeV")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "propagator.png"))
    plt.close(fig)
    print("wrote propagator.png")


# ----------------------------------------------------------------------------
# Figure : the propagator anomalous dimensions at the IR scale.
#
# dtZA/dtZc are the self-consistently iterated flows of the dressings (model.hh
# solves them in the eta fixed-point loop), so
#     eta_A(p) = -dt Z_A(p) / Z_A(p) ,   eta_c(p) = -dt Z_c(p) / Z_c(p) .
# In the scaling solution these approach the IR exponents 2 kappa + ... ; the
# sum rule eta_A + 2 eta_c = 0 at p -> 0 is drawn as a reference line.
# ----------------------------------------------------------------------------
def figure_eta():
    p, dZA = mp("ZA")
    _, dtZA = mp("dtZA")
    _, dZc = mp("Zc")
    _, dtZc = mp("dtZc")

    etaA = -dtZA / dZA
    etac = -dtZc / dZc

    fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=600)
    ax.plot(p, etaA, color=C(3), label=r"$\eta_A(p)$")
    ax.plot(p, etac, color=C(0), ls="--", label=r"$\eta_c(p)$")
    ax.plot(p, etaA + 2 * etac, color="0.4", ls=":", lw=1.0,
            label=r"$\eta_A + 2\eta_c$")
    ax.axhline(0.0, color="0.5", lw=0.6, alpha=0.6, zorder=0)

    ax.set_xlabel(r"$p\textrm{ [GeV]}$")
    ax.set_ylabel(r"$\eta = -\partial_t Z/Z$")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(rf"Anomalous dimensions at $k = {kIR}$ GeV")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "eta.png"))
    plt.close(fig)
    print("wrote eta.png")


# ----------------------------------------------------------------------------
# Figure : the bare vertex dressings at the IR scale. These are the flowed
# Variables themselves (not yet combined into couplings), each dimensionless.
# ----------------------------------------------------------------------------
def figure_vertices():
    fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=600)

    for name, lab, col, ls in (("ZA3", r"$Z_{A^3}$", C(3), "-"),
                               ("ZA4", r"$Z_{A^4}$", C(1), "--"),
                               ("ZAcbc", r"$Z_{A\bar c c}$", C(0), "-.")):
        p, d = mp(name)
        ax.plot(p, d, color=col, ls=ls, label=lab)

    ax.set_xlabel(r"$p\textrm{ [GeV]}$")
    ax.set_ylabel(r"$Z(p)$")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(rf"Vertex dressings, $k = {kIR}$ GeV")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "vertices.png"))
    plt.close(fig)
    print("wrote vertices.png")


# ----------------------------------------------------------------------------
# Perturbative running of the strong coupling. This is PURE Yang-Mills: there
# are no quarks in the flow, so the beta function coefficients are the Nf = 0
# ones. alpha0 and Lambda are fitted to the fRG coupling in the UV window.
# ----------------------------------------------------------------------------
_NF = 0
_BETA0 = (11 - 2 / 3 * _NF) / (4 * np.pi)
_BETA1 = (102 - 38 / 3 * _NF) / (4 * np.pi) ** 2
_B1 = _BETA1 / _BETA0 ** 2


def _lambertw_m1(x):
    """Lambert W, lower branch (W_{-1} <= -1), real for x in [-1/e, 0).

    scipy-free replacement for scipy.special.lambertw(x, -1); returns NaN
    outside the real domain of the -1 branch.
    """
    x = np.asarray(x, dtype=float)
    w = np.full_like(x, np.nan)
    dom = (x >= -1.0 / math.e) & (x < 0.0)
    xs = x[dom]
    # asymptotic seed for the lower branch (Corless et al. 1996)
    L1 = np.log(-xs)
    L2 = np.log(-L1)
    ws = L1 - L2 + L2 / L1
    # Halley iteration on w e^w = x (guard the exact x=-1/e boundary, 0/0)
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(60):
            ew = np.exp(ws)
            f = ws * ew - xs
            ws = ws - f / (ew * (ws + 1.0) - (ws + 2.0) * f / (2.0 * ws + 2.0))
    w[dom] = ws
    return w


def alpha_one_loop(mu, alpha0, Lam):
    return alpha0 / (_BETA0 * np.log(mu ** 2 / Lam ** 2))


def alpha_two_loop(mu, alpha0, Lam):
    xi = -1 / (math.e * _B1) * (Lam ** 2 / mu ** 2) ** (1 / _B1)
    return -alpha0 / (_BETA0 * _B1) / (1 + _lambertw_m1(xi))


def _fit_one_loop(mu, alpha):
    """Linear fit of 1/alpha vs ln(mu^2) -> (alpha0, Lambda)."""
    a, b = np.polyfit(np.log(mu ** 2), 1.0 / alpha, 1)
    alpha0 = _BETA0 / a
    Lam = np.exp(-b / (2.0 * a))
    return alpha0, Lam


def _fit_two_loop(mu, alpha, p0):
    """Gauss-Newton fit of alpha_two_loop, seeded from the one-loop fit."""
    p = np.array(p0, dtype=float)
    for _ in range(100):
        r = alpha_two_loop(mu, *p) - alpha
        eps = 1e-6 * np.maximum(np.abs(p), 1.0)
        J = np.empty((mu.size, 2))
        for j in range(2):
            pp = p.copy()
            pp[j] += eps[j]
            J[:, j] = (alpha_two_loop(mu, *pp) - alpha_two_loop(mu, *p)) / eps[j]
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return p[0], p[1]


# ----------------------------------------------------------------------------
# Figure : the gauge couplings alpha = g^2/(4pi) at the IR scale -- the full
#          log-log running, plus a lin-log zoom into the perturbative window
#          with the one- and two-loop curves fitted in the UV.
#
# The avatars are built from the flowed vertex dressings divided by the
# propagator dressings of the legs, which is what makes them RG invariant:
#     alpha_{A^3}   = Z_{A^3}^2   / (4pi Z_A^3) ,
#     alpha_{A^4}   = Z_{A^4}     / (4pi Z_A^2) ,
#     alpha_{Acbc}  = Z_{Acbc}^2  / (4pi Z_A Z_c^2) .
# ----------------------------------------------------------------------------
def figure_couplings():
    p, dZA = mp("ZA")
    dZc = maps["Zc"]["data"][-1]
    dZA3 = maps["ZA3"]["data"][-1]
    dZA4 = maps["ZA4"]["data"][-1]
    dZAcbc = maps["ZAcbc"]["data"][-1]

    aA3 = (dZA3 ** 2 / (dZA ** 3)) / (4 * np.pi)
    aA4 = (dZA4 / (dZA ** 2)) / (4 * np.pi)
    aAcbc = (dZAcbc ** 2 / (dZA * dZc ** 2)) / (4 * np.pi)

    couplings = [(aA3, r"$\alpha_{A^3}$", C(3)),
                 (aA4, r"$\alpha_{A^4}$", C(1)),
                 (aAcbc, r"$\alpha_{A\bar c c}$", C(0))]

    pmin, pmax = 4.0, min(100.0, float(p.max()))
    win = (p >= pmin) & (p <= pmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), dpi=600)

    # (1) full running, log-log
    for a, lab, col in couplings:
        ax1.plot(p, a, label=lab, color=col)
    logscale(ax1, *[a for a, _, _ in couplings])
    ax1.set_ylabel(r"$\alpha = g^2/4\pi$")
    ax1.set_title("full running")
    ax1.legend(frameon=False, fontsize=9)

    # (2) zoom into the perturbative window, lin-log
    for a, lab, col in couplings:
        ax2.plot(p, a, label=lab, color=col)
    ax2.set_xlim(pmin, pmax)

    # perturbative one- & two-loop running, fitted to the ghost-gluon coupling
    # in the UV window where perturbation theory applies (Acbc is the cleanest
    # avatar: its vertex is UV finite by the non-renormalisation theorem)
    fitwin = (p >= 20.0) & np.isfinite(aAcbc) & (aAcbc > 0)
    pf, af = p[fitwin], aAcbc[fitwin]
    a0_1, Lam1 = _fit_one_loop(pf, af)
    a0_2, Lam2 = _fit_two_loop(pf, af, (a0_1, Lam1))
    pc = np.geomspace(pmin, pmax, 800)
    y1 = alpha_one_loop(pc, a0_1, Lam1)
    y2 = alpha_two_loop(pc, a0_2, Lam2)
    m1, m2 = np.isfinite(y1) & (y1 > 0), np.isfinite(y2) & (y2 > 0)
    ax2.plot(pc[m1], y1[m1], color="k", ls=":", lw=1.2, label="1-loop")
    ax2.plot(pc[m2], y2[m2], color="k", ls="--", lw=1.2, label="2-loop")
    print(f"perturbative fit (Nf={_NF}): 1-loop Lambda={Lam1:.3g}, "
          f"2-loop Lambda={Lam2:.3g} GeV")

    # y-limits from the fRG couplings inside the window (with a small margin)
    yvals = np.concatenate([a[win] for a, _, _ in couplings])
    yvals = yvals[np.isfinite(yvals)]
    ylo, yhi = yvals.min(), yvals.max()
    pad = 0.05 * (yhi - ylo)
    ax2.set_ylim(ylo - pad, yhi + pad)
    ax2.set_title(rf"zoom: $p \in [{pmin:.0f}, {pmax:.0f}]$ GeV")
    ax2.legend(frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$p\textrm{ [GeV]}$")
        ax.set_xscale("log")

    fig.suptitle(rf"Gauge couplings, $k = {kIR}$ GeV")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "couplings.png"))
    plt.close(fig)
    print("wrote couplings.png")


# ----------------------------------------------------------------------------
# Figure : the gluon mass parameter over the flow. This is the one k-dependent
# plot -- everything else is read at the IR end of the flow.
#
# m2A is the read-out of model.hh: Z_A(p_min) p_min^2 - p_min^2, i.e. the mass
# term the gluon two-point function carries at the smallest grid momentum. It
# starts negative (the tuned initial condition) and crosses to its positive IR
# value; the axis switches to symlog when that happens.
# ----------------------------------------------------------------------------
def figure_m2A():
    k = scalars["k"]
    m2A = scalars["m2A"]
    print("m2A(k_IR) =", m2A[-1])

    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=600)

    ax.plot(k, m2A, color=C(3))
    ax.axhline(0.0, color="0.5", lw=0.6, alpha=0.6, zorder=0)
    ax.set_xlabel(r"$k\textrm{ [GeV]}$")
    ax.set_ylabel(r"$m_A^2\textrm{ [GeV}^2\textrm{]}$")
    ax.set_xscale("log")
    logscale(ax, m2A)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "m2A.png"))
    plt.close(fig)
    print("wrote m2A.png")


if __name__ == "__main__":
    # One figure must never take the rest down with it: these are read from a
    # flow that may well have diverged (that is usually WHY one is plotting),
    # and a fit or an axis limit that chokes on NaN is not a reason to lose the
    # other figures. The traceback is still printed.
    for fig_fn in (figure_dressings, figure_propagator, figure_eta,
                   figure_vertices, figure_couplings, figure_m2A):
        try:
            fig_fn()
        except Exception:
            plt.close("all")
            print(f"FAILED {fig_fn.__name__}:", file=sys.stderr)
            traceback.print_exc()
    print(f"done — figures in {figdir}")
