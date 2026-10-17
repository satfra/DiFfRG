(* ::Package:: *)

(* ::Title:: *)
(*Tutorial 9: Momentum-dependent flows with NumTracer*)


(* ::Text:: *)
(*The quark-meson model of Tutorial 8 at finite temperature, with momentum-dependent meson wave functions ZPi(p0, |p|) and ZSig(p0, |p|) on a Matsubara stack. The Yukawa coupling is constant and the quarks keep their classical wave function.*)
(*The kernels are traced with NumTracer and emitted with MakeNTKernelDiFfRG / MakeNTKernelGroupDiFfRG.*)
(*Run headless with:  wolfram -script QM.wl        (one flow only:  NT_FLOWS=V wolfram -script QM.wl)*)


(* ::Section:: *)
(*Setup*)


(* ::Input::Initialization:: *)
Get["DiFfRG`"];
Get["NumTracer`"];
SetDirectory[If[$InputFileName === "", Directory[], DirectoryName[$InputFileName]]];


(* ::Subsection:: *)
(*Fields and truncation*)


(* ::Text:: *)
(*Quarks are a Grassmann pair, mesons commuting fields. The quark two-point function uses the qbqFT basis of TensorBases, {i gamma_0, i gamma.p, 1}; the meson rules are written out by hand below.*)


(* ::Input::Initialization:: *)
fields = <|
   "Commuting" -> {sig[p], pion[p, {f}]},
   "Grassmann" -> {{qb[p, {d, Cc, Ff}], q[p, {d, Cc, Ff}]}}
   |>;
truncation = <|
   GammaN -> {{qb, q}, {pion, pion}, {sig, sig}, {qb, q, pion}, {qb, q, sig},
     {sig, pion, pion}, {sig, sig, sig}, {pion, pion, pion, pion}, {pion, pion, sig, sig}, {sig, sig, sig, sig}},
   Propagator -> {{qb, q}, {pion, pion}, {sig, sig}},
   R -> {{qb, q}, {pion, pion}, {sig, sig}},
   Rdot -> {{qb, q}, {pion, pion}, {sig, sig}},
   Field -> {{sig}}
   |>;
bases = <|
   GammaN -> {{qb, q} -> "qbqFT"},
   Propagator -> {{qb, q} -> "qbqFT"},
   R -> {{qb, q} -> {"qbqFT", 2}},
   Rdot -> {{qb, q} -> {"qbqFT", 2}}
   |>;
setup = <|
   "FieldSpace" -> fields,
   "Truncation" -> truncation,
   "FeynmanRules" -> bases,
   "DiagramStyling" -> <|"Styles" -> {pion -> {Blue, Dashed}, sig -> {Black, Dashed}, q -> {Black}}|>
   |>;
FSetGlobalSetup[setup];

(* Nf = 2 and Nc = 3 are fixed here: a closed quark loop then folds to a number. *)
SetNc[3];
SetNf[2];


(* ::Subsection:: *)
(*Meson vertices from the potential*)


(* ::Text:: *)
(*The meson vertices are the Taylor coefficients of V(rho) at the expansion point rhoPhi, with rho = (sigma^2 + pi^2)/2, written in terms of the kernel parameters d1V, d2V, d3V.*)


(* ::Input::Initialization:: *)
TFlav[0, f1_, f2_] := deltaFundFlav[f1, f2]/Sqrt[2*Nf];

FunctionalD[expr_, v : ((f_)[_] | {(f_)[_], _Integer}) .., OptionsPattern[]] :=
  Internal`InheritedBlock[{f},
   f /: D[f[x_], f[y_], NonConstants -> {f}] := DiracDelta[x - y];
   f /: D[f, f[y_], NonConstants -> {f}] := DiracDelta[#1 - y] &;
   D[expr, v, NonConstants -> {f}]];

Taylor[func_, var_, order_] := func +
   Total[Table[((1/n!)*Symbol["d" ~~ ToString[n] ~~ ToString[func]])*Product[var[], {i, 1, n}], {n, 1, order}]];

lamMeson[pions_, sigmas_] :=
  Module[{action, sigmaDeriv},
   action = Taylor[V, (1/2)*((sigma^2 - sigma0^2) + Module[{b}, pionAux[b]^2]) &, Length[pions] + sigmas];
   sigmaDeriv = D[action, {sigma, sigmas}];
   Simplify[If[Length[pions] >= 1,
       FunctionalD[sigmaDeriv, (pionAux[#1] &) /@ pions /. List -> Sequence] //.
          DiracDelta[(x_) - (y_)] -> deltaAdjFlav[x, y] //.
         DiracDelta[-1 + (y_)] -> deltaAdjFlav[1, y],
       sigmaDeriv] //.
     {pionAux[a_] -> 0, sigma0 -> sigma, sigma -> Sqrt[2*rhoPhi]}]];


(* ::Subsection:: *)
(*Diagrammatic rules*)


(* ::Text:: *)
(*Every two-point rule is a function of its second momentum. ZPi and ZSig are functions of the Matsubara frequency and the spatial momentum separately. The regulators are three-dimensional, RB(k^2, p_spatial^2), and carry the pion normalisation ZPi(0, p_min), which the model pins to one; their scale derivative then carries -etaPi0 RB.*)


(* ::Input::Initialization:: *)
GPhi[m2_, p_, Z_] := 1/(Z*sp[p, p] + m2 + RB[k^2, sps[p, p]]);
ZPiOf[p_] := ZPi[vec[p, 0], Sqrt[sps[p, p]]];
ZSigOf[p_] := ZSig[vec[p, 0], Sqrt[sps[p, p]]];
Mq = hPhi*Sqrt[2*rhoPhi]/Sqrt[2*Nf];

diagRules = Join[FMakeDiagrammaticRules[], {
    Propagator[{pion, pion}, {{p1_, {f1_}}, {p2_, {f2_}}}] :> deltaAdjFlav[f1, f2]*GPhi[d1V, p2, ZPiOf[p2]],
    Propagator[{sig, sig}, {{p1_}, {p2_}}] :> GPhi[d1V + 2*rhoPhi*d2V, p2, ZSigOf[p2]],
    GammaN[{pion, pion}, {{p1_, {f1_}}, {p2_, {f2_}}}] :> deltaAdjFlav[f1, f2]*(ZPiOf[p2]*sp[p2, p2] + d1V),
    GammaN[{sig, sig}, {{p1_}, {p2_}}] :> ZSigOf[p2]*sp[p2, p2] + d1V + 2*rhoPhi*d2V,
    R[{pion, pion}, {{p1_, {f1_}}, {p2_, {f2_}}}] :> deltaAdjFlav[f1, f2]*RB[k^2, sps[p2, p2]],
    R[{sig, sig}, {{p1_}, {p2_}}] :> RB[k^2, sps[p2, p2]],
    Rdot[{pion, pion}, {{p1_, {f1_}}, {p2_, {f2_}}}] :>
     deltaAdjFlav[f1, f2]*(RBdot[k^2, sps[p2, p2]] - etaPi0*RB[k^2, sps[p2, p2]]),
    Rdot[{sig, sig}, {{p1_}, {p2_}}] :> RBdot[k^2, sps[p2, p2]] - etaPi0*RB[k^2, sps[p2, p2]],

    GammaN[{sig, sig, sig}, {{p1_}, {p2_}, {p3_}}] :> lamMeson[{}, 3],
    GammaN[{sig, pion, pion}, {{p1_}, {p2_, {f2_}}, {p3_, {f3_}}}] :> lamMeson[{f2, f3}, 1],
    GammaN[{sig, sig, pion, pion}, {{p1_}, {p2_}, {p3_, {f3_}}, {p4_, {f4_}}}] :> lamMeson[{f3, f4}, 2],
    GammaN[{pion, pion, pion, pion}, {{p1_, {f1_}}, {p2_, {f2_}}, {p3_, {f3_}}, {p4_, {f4_}}}] :>
     lamMeson[{f1, f2, f3, f4}, 0],
    GammaN[{sig, sig, sig, sig}, {{p1_}, {p2_}, {p3_}, {p4_}}] :> lamMeson[{}, 4],

    GammaN[{sig, qb, q}, {{p1_}, {p2_, {d2_, A2_, F2_}}, {p3_, {d3_, A3_, F3_}}}] :>
     hPhi*(deltaFundFlav[F2, F3]/Sqrt[2*Nf])*deltaDirac[d2, d3]*deltaFundCol[A2, A3],
    GammaN[{pion, qb, q}, {{p1_, {f1_}}, {p2_, {d2_, A2_, F2_}}, {p3_, {d3_, A3_, F3_}}}] :>
     hPhi*TFlav[f1, F2, F3]*((-I)*gamma5[d2, d3])*deltaFundCol[A2, A3]
    }];


(* ::Text:: *)
(*The quark dressings in the qbqFT basis {i gamma_0, i gamma.p, 1}. The spatial element is built on the antiquark momentum p1 = -p2 while the dressings are written in the quark momentum p2, hence the minus signs (see Tutorial 8). The regulator sits on the spatial element only.*)


(* ::Input::Initialization:: *)
RFhat[p_] := RF[k^2, sps[p, p]]/Sqrt[sps[p, p]];
dressingRules[expr_] := UseLorentzLinearity[expr //. {
     dressing[GammaN, {qb, q}, 1, {p1_, p2_}] :> vec[p2, 0],
     dressing[GammaN, {qb, q}, 2, {p1_, p2_}] :> -1,
     dressing[GammaN, {qb, q}, 3, {p1_, p2_}] :> -Mq,
     dressing[InverseProp, {qb, q}, 1, {p1_, p2_}] :> vec[p2, 0],
     dressing[InverseProp, {qb, q}, 2, {p1_, p2_}] :> 1 + RFhat[p2],
     dressing[InverseProp, {qb, q}, 3, {p1_, p2_}] :> -Mq,
     dressing[R, {qb, q}, 2, {p1_, p2_}] :> -RFhat[p2],
     dressing[Rdot, {qb, q}, 2, {p1_, p2_}] :> -RFdot[k^2, sps[p2, p2]]/Sqrt[sps[p2, p2]]
     }];
FSetSymmetricDressing[GammaN, {qb, q}];


(* ::Subsection:: *)
(*Sanity checks*)


(* ::Text:: *)
(*Each qbqFT projector must return its own dressing on the tree-level two-point function, and (Gamma^(2) + R) G = 1 for the mesons.*)


(* ::Input::Initialization:: *)
Module[{g2, proj, got, want},
  g2 = GammaN[{qb, q}, {{p1, {d1, A1, F1}}, {p2, {d2, A2, F2}}}] /. diagRules;
  want = {vec[p2, 0], -1, -Mq};
  Do[
   proj = TBGetProjector["qbqFT", n, {p2, d2, A2, F2}, {p1, d1, A1, F1}];
   got = Simplify[UseLorentzLinearity[dressingRules[FormTrace[FTerm[proj, g2]]] //. p1 -> -p2]];
   Print["qbqFT element ", n, ": ", got, "  (want ", want[[n]], ")"];
   If[Simplify[got - want[[n]]] =!= 0, Print["qbqFT projection ", n, " is wrong. Aborting."]; Abort[]],
   {n, 1, 3}]];
Print["pion  (Gamma2 + R) G = ", Simplify[(ZPiOf[p1]*sp[p1, p1] + d1V + RB[k^2, sps[p1, p1]])*GPhi[d1V, p1, ZPiOf[p1]]]];
Print["sigma (Gamma2 + R) G = ",
  Simplify[(ZSigOf[p1]*sp[p1, p1] + d1V + 2*rhoPhi*d2V + RB[k^2, sps[p1, p1]])*GPhi[d1V + 2*rhoPhi*d2V, p1, ZSigOf[p1]]]];


(* ::Subsection:: *)
(*Kinematics*)


(* ::Text:: *)
(*Component 0 is the Matsubara direction. The two-point flows have one external momentum p1 = (p0, p, 0, 0) and the loop momentum l1 = (f0, l1 cos1, l1 sin1, 0); frameShiftedLoop adds the fermionic loop momentum lf1 with frequency f0 + pi T. The potential has no external momentum (a tiny one keeps the frame regular).*)


(* ::Input::Initialization:: *)
$Assumptions = k > 0 && T > 0 && l1 > 0 && p > 0 && Element[f0, Reals] && Element[p0, Reals] && -1 < cos1 < 1;
FSetRegisterSize[64];

frameProp = NumTracer`frameShiftedLoop[NumTracer`propFrameFT[p0, p, f0, l1, cos1, p1, l1], l1, lf1, Pi T];
argsProp = {l1, cos1, f0, p0, p, k, T};
frameV = NumTracer`frameShiftedLoop[NumTracer`propFrameFT[0, 10^(-3), f0, l1, cos1, p1, l1], l1, lf1, Pi T];
argsV = {l1, f0, k, T};
$loops = {l1, lf1};


(* ::Text:: *)
(*Scalar products, read off the frame: sp is the 4D product, sps the spatial one, vec[a, 0] the frequency. cosSyms are the cosines between the loop momentum and each external momentum.*)


(* ::Input::Initialization:: *)
ftSpatialMag[frame_, v_] := Simplify[Sqrt[Rest[frame[v]] . Rest[frame[v]]]];
ftScalarRules[frame_, loops_List, exts_List, cosSyms_List] :=
  Module[{mags = ftSpatialMag[frame, #] & /@ exts, lmag},
   lmag = If[loops === {}, 0, ftSpatialMag[frame, First[loops]]];
   Join[
    Table[vec[a, 0] -> frame[a][[1]], {a, Join[loops, exts]}],
    Flatten @ Table[sp[a, b] -> Simplify[frame[a] . frame[b]], {a, loops}, {b, loops}],
    Flatten @ Table[sp[a, b] -> Simplify[frame[a] . frame[b]], {a, exts}, {b, exts}],
    Flatten @ Table[sps[a, b] -> Simplify[Rest[frame[a]] . Rest[frame[b]]], {a, loops}, {b, loops}],
    Flatten @ Table[sps[a, b] -> Simplify[Rest[frame[a]] . Rest[frame[b]]], {a, exts}, {b, exts}],
    Flatten @ Table[
      With[{lv = loops[[j]], ev = exts[[i]], c = cosSyms[[i]], m = mags[[i]]},
       With[{val = Simplify[frame[lv][[1]]*frame[ev][[1]]] + lmag*m*c}, {sp[lv, ev] -> val, sp[ev, lv] -> val}]],
      {i, Length[exts]}, {j, Length[loops]}],
    Flatten @ Table[
      With[{lv = loops[[j]], ev = exts[[i]], c = cosSyms[[i]], m = mags[[i]]},
       {sps[lv, ev] -> lmag*m*c, sps[ev, lv] -> lmag*m*c}],
      {i, Length[exts]}, {j, Length[loops]}]]];
ftParam[expr_, frame_, loops_, exts_, cosSyms_] :=
  UseLorentzLinearity[expr] //. Join[ftScalarRules[frame, loops, exts, cosSyms],
    {Sqrt[l1^2] -> l1, Power[Power[l1, 2], Rational[n_, 2]] :> l1^n}];


(* ::Subsection:: *)
(*Kernel parameters*)


(* ::Text:: *)
(*The order of this list is the C++ call signature of every kernel; the device::tie calls in model.hh must follow it. d1V, d2V, d3V are AD parameters: the potential kernel is differentiated with respect to them for the jacobian. The dressings live on a bosonic Matsubara stack with a logarithmic radial grid.*)


(* ::Input::Initialization:: *)
bosCoords = "BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>";
bosInterp = "SplineInterpolator1DStack<double, " <> bosCoords <> ">";
kernelParameterList = {
   <|"Name" -> "k", "Type" -> "double"|>,
   <|"Name" -> "T", "Type" -> "double"|>,
   <|"Name" -> "hPhi", "Type" -> "double"|>,
   <|"Name" -> "etaPi0", "Type" -> "double"|>,
   <|"Name" -> "rhoPhi", "Type" -> "double"|>,
   <|"Name" -> "d1V", "Type" -> "double", "AD" -> True|>,
   <|"Name" -> "d2V", "Type" -> "double", "AD" -> True|>,
   <|"Name" -> "d3V", "Type" -> "double", "AD" -> True|>,
   <|"Name" -> "ZPi", "Type" -> bosInterp, "Const" -> True, "Reference" -> True|>,
   <|"Name" -> "ZSig", "Type" -> bosInterp, "Const" -> True, "Reference" -> True|>};

(* Options shared by every two-point flow: a map over the (p0, p) stack, summing the Matsubara frequency f0. *)
mesonOpts = Sequence[
   "Integrator" -> "Integrator_fT_p2_1ang",
   "Parameters" -> kernelParameterList,
   "Coordinates" -> {bosCoords},
   "CoordinateArguments" -> {"p0", "p"},
   "IntegrationVariables" -> {"l1", "cos1", "f0"}];


(* ::Section:: *)
(*Meson two-point functions*)


(* ::Text:: *)
(*The wave function is the momentum-dependent part of the two-point flow,*)
(*    dt Z(p0, p) = [ F_sym(p0, p) - F(0, 0) ] / (p0^2 + p^2),*)
(*with F_sym symmetrized in p -> -p. ZPi comes from the pion two-point function, ZSig from the sigma one.*)


(* ::Input::Initialization:: *)
proj2pt[projFac_, expr_] := Module[{mk, tp, tm, t0},
   mk[sub_] := (Times @@ # &) /@ List @@ FTerm[projFac, (((expr //. p2 -> -p1) /. sub) /. diagRules)];
   tp = mk[{}]; tm = mk[p1 -> -p1]; t0 = mk[p1 -> 0];
   Simplify[UseLorentzLinearity[((tp + tm)/2 - t0)/sp[p1, p1]]]];

traceMeson[list_] := NumTrace[
   FromFunKit[ftParam[dressingRules[FEx @@ FTerm /@ list], frameProp, $loops, {p1}, {cos1}]],
   "Frame" -> frameProp, "Args" -> argsProp];


(* ::Text:: *)
(*Both projections are of one loop: the same propagators, regulators and dressing lookups. MakeNTKernelGroupDiFfRG emits them as one kernel with two outputs, which computes everything they share once. "SharedGrids" declares that ZPi and ZSig live on one grid, so a lookup of both at the same momentum computes the grid coordinates once.*)


(* ::Input::Initialization:: *)
NTFlowGate[{"Mesons", "ZPi", "ZSig", "MesonsPlain"}, Module[{fRGPi, fRGSig, ntkPi, ntkSig},
   fRGPi = FTakeDerivatives[WetterichEquation, {pion[i1], pion[i2]}] // FTruncate // FRoute;
   fRGSig = FTakeDerivatives[WetterichEquation, {sig[i1], sig[i2]}] // FTruncate // FRoute;
   ntkPi = traceMeson[proj2pt[deltaAdjFlav[f1, f2]/(Nf^2 - 1), fRGPi["1-Loop"]["Expression"]]];
   ntkSig = traceMeson[proj2pt[1, fRGSig["1-Loop"]["Expression"]]];

   NTFlowGate["Mesons",
    MakeNTKernelGroupDiFfRG[{ntkPi, ntkSig},
     "Name" -> "Mesons",
     "Outputs" -> {"ZPi", "ZSig"},
     (* the rescaling with the absorbed wave function: + etaPi0 Z *)
     "Constants" -> {etaPi0*ZPi[p0, p], etaPi0*ZSig[p0, p]},
     "SharedGrids" -> {{ZPi, ZSig}},
     mesonOpts]];

   (* For bench.cc only: the two flows as separate kernels, and the group without the hoisted form. They go into
      a library of their own, which the tutorial does not build by default. *)
   SetFlowName["bench_flows"];
   NTFlowGate["ZPi", MakeNTKernelDiFfRG[ntkPi, etaPi0*ZPi[p0, p], "Name" -> "ZPi", "SharedGrids" -> {{ZPi, ZSig}}, mesonOpts]];
   NTFlowGate["ZSig", MakeNTKernelDiFfRG[ntkSig, etaPi0*ZSig[p0, p], "Name" -> "ZSig", "SharedGrids" -> {{ZPi, ZSig}}, mesonOpts]];
   NTFlowGate["MesonsPlain",
    MakeNTKernelGroupDiFfRG[{ntkPi, ntkSig},
     "Name" -> "MesonsPlain",
     "Outputs" -> {"ZPi", "ZSig"},
     "Constants" -> {etaPi0*ZPi[p0, p], etaPi0*ZSig[p0, p]},
     "SharedGrids" -> {{ZPi, ZSig}},
     "Hoist" -> False,
     mesonOpts]];
   SetFlowName["flows"];
   ]];


(* ::Section:: *)
(*Effective potential*)


(* ::Text:: *)
(*The flow of V(rho) is the Wetterich equation in a constant background. The model evaluates it at all quadrature points of the field grid in one map_points() call on the GPU ("MapPoints" -> True), with rho, d1V = V', d2V = V'' different at every point. "PointArguments" names these per-point parameters.*)


(* ::Input::Initialization:: *)
traceV[] := Module[{fRG, tr},
   fRG = WetterichEquation // FTruncate // FRoute;
   tr = FTerm[1, ((fRG["1-Loop"]["Expression"] /. p1 -> 0) /. diagRules)];
   tr = FEx @@ FTerm /@ ((Times @@ # &) /@ List @@ tr);
   NumTrace[FromFunKit[ftParam[dressingRules[tr], frameV, $loops, {}, {}]], "Frame" -> frameV, "Args" -> argsV]];

potentialOpts = Sequence[
   "Integrator" -> "Integrator_fT_p2",
   "Parameters" -> kernelParameterList,
   "Coordinates" -> {},
   "CoordinateArguments" -> {},
   "IntegrationVariables" -> {"l1", "f0"},
   "Device" -> "GPU",
   "AD" -> True,
   "MapPoints" -> True];

NTFlowGate[{"V", "VAuto"}, Module[{ntk = traceV[]},
   NTFlowGate["V", MakeNTKernelDiFfRG[ntk, "Name" -> "V", "PointArguments" -> {"rhoPhi", "d1V", "d2V", "d3V"}, potentialOpts]];
   (* For bench.cc only: the default PointArguments, i.e. the AD parameters, without rhoPhi. *)
   SetFlowName["bench_flows"];
   NTFlowGate["VAuto", MakeNTKernelDiFfRG[ntk, "Name" -> "VAuto", potentialOpts]];
   SetFlowName["flows"];
   ]];


(* ::Section:: *)
(*Collect the flows*)


(* ::Text:: *)
(*UpdateNTFlows writes flows/CMakeLists.txt and flows.{hh,cc}, and adds the NumTracer build step. A run restricted with NT_FLOWS leaves this to a later full run.*)


(* ::Input::Initialization:: *)
NTFlowSelectionReport[];
If[!NTFlowSliceQ[],
  UpdateNTFlows["QMFlows"];
  SetFlowName["bench_flows"];
  UpdateNTFlows["QMBenchFlows"];
  SetFlowName["flows"]];
