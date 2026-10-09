(* ::Package:: *)

(* ::Text:: *)
(*============================================================================================ *)
(*SU(3) Yang-Mills theory -- fully momentum- and angle-dependent truncation*)
(*---------------------------------------------------------------------------------------------------------------------------------------------------*)
(*The three-point vertices are resolved over the full momentum triangle rather than evaluated on*)
(*the symmetric point; see ../SP for the symmetric-point sibling.*)
(*Run it from this directory with  wolframscript -file Yang-Mills.wl*)
(*============================================================================================ *)


(* ::Section:: *)
(*Setup*)


(* ::Subsection::Closed:: *)
(*Packages*)


Get["DiFfRG`"];
SetDirectory[GetDirectory[]];
DefineFormExecutable["/usr/bin/tform -w8"];
FSetCacheDirectory[GetDirectory[] <> "/TraceCache/"];


(* ::Subsection::Closed:: *)
(*QFT setup*)


fields = <|
  "Commuting" -> {A[p, {v, c}]},
  "Grassmann" -> {{cb[p, {c}], c[p, {c}]}}
|>;

truncation = <|
  GammaN -> {{A, A}, {A, A, A}, {A, A, A, A}, {A, cb, c}, {cb, c}},
  Propagator -> {{A, A}, {cb, c}},
  Rdot -> {{A, A}, {cb, c}},
  S -> {{A, A}, {A, A, A}, {A, A, A, A}, {cb, c}, {cb, c, A}},
  Field -> {{}}
|>;

bases = <|
  GammaN -> {{A, A} -> {"AA", 1}, {A, A, A} -> "AAAClass", {A, A, A, A} -> "AAAAClass",
     {A, cb, c} -> {"Acbc", 1}, {cb, c} -> "cbc"},
  S -> {{A, A} -> {"AA", 1}, {A, A, A} -> "AAAClass", {A, A, A, A} -> "AAAAClass",
     {A, cb, c} -> {"Acbc", 1}, {cb, c} -> "cbc"},
  Propagator -> {{A, A} -> {"AA", 1}, {cb, c} -> "cbc"},
  Rdot -> {{A, A} -> {"AA", 1}, {cb, c} -> "cbc"}
|>;

FSetTexStyles[cb -> "\\bar{c}"];

Setup = <|
  "FieldSpace" -> fields,
  "Truncation" -> truncation,
  "FeynmanRules" -> bases,
  "DiagramStyling" -> <|"Styles" -> {A -> {Orange}, c -> {Black, Dashed}}|>
|>;
FSetGlobalSetup[Setup];


(* ::Subsection::Closed:: *)
(*Diagrammatic rules*)


(* Setup-dependent but momentum-independent, so built once here rather than re-derived inside
   every trace below. *)
diagRules = FMakeDiagrammaticRules[];


(* ::Subsection::Closed:: *)
(*Parametrizations*)


(* Symmetric-point momentum variable, used only by the 4-gluon symmetric point ZA4SP. *)
SP4Patt[p1e_, p2e_, p3e_, p4e_] :=
  FullSimplify[UseLorentzLinearity[{Sqrt[(sp[p1, p1] + sp[p2, p2] + sp[p3, p3] + sp[p4, p4])/4]} /.
     {p1 :> p1e, p2 :> p2e, p3 :> p3e, p4 :> p4e}]];

(* Symmetric-triangle (S0, S1, SPhi) parametrisation of the three-point functions. S0 is the
   overall scale (log-gridded), S1 the shape (linear, 0 at the symmetric point and -> 1 at
   collinear) and SPhi an angle (linear, periodic). The transverse projector's collinear
   1/(cos^2-1) then sits at a single grid CORNER (S1 -> 1 AND SPhi -> 0) rather than on a whole
   face as with raw (|p1|, |p2|, cos), which is what makes the angle-resolved flow tunable. *)
transf3PTo = TB3PToS0S1SPhi[p1, p2, p3, l1, S0, S1, SPhi];
transf3PFrom = TB3PFromS0S1SPhi[p1, p2, p3, S0, S1, SPhi];
P3Patt[p1e_, p2e_, p3e_] :=
  FullSimplify[UseLorentzLinearity[{transf3PFrom[S0], transf3PFrom[S1], transf3PFrom[SPhi]} /.
     {p1 :> p1e, p2 :> p2e, p3 :> p3e}]];

(* Tadpole (Qk) parametrisation: the 4-gluon inside the gluon self-energy tadpole depends on one
   external (p) and one loop (q) momentum, and tadpolePatt maps that pair onto the same
   (S0, S1, SPhi) grid. The gluon self-energy must be fed this angle-resolved ZA4tadpole, NOT the
   symmetric-point ZA4SP. *)
transfTadpoleTo = TB3PToS0S1SPhiQk[p1, p2, q, S0, S1, SPhi]; (* the Qk transform only parametrises the loop when it is named q *)
transfTadpoleFrom = TB3PFromS0S1SPhiQk[p1, p2, S0, S1, SPhi];
tadpolePatt[p1e_, p2e_] :=
  FullSimplify[UseLorentzLinearity[{transfTadpoleFrom[S0], transfTadpoleFrom[S1], transfTadpoleFrom[SPhi]} /.
     {p1 :> p1e, p2 :> p2e}]];

(* Rules common to all flows. The 4-gluon rule is the one thing that differs by context, so it is
   appended separately below. *)
commonDressing = {
  (* Propagators *)
  dressing[GammaN, {cb, c}, 1, {p1_, p2_}] :> -Zc[Sqrt[sp[p2, p2]]]*sp[p2, p2],
  dressing[GammaN, {A, A}, 1, {p1_, p2_}] :> ZA[Sqrt[sp[p2, p2]]]*sp[p2, p2],
  (* Inverse propagators (these carry the regulator) *)
  dressing[InverseProp, {cb, c}, 1, {p1_, p2_}] :> -(Zc[Sqrt[sp[p2, p2]]]*sp[p2, p2] + RB[k^2, sp[p2, p2]]*Zc[k]),
  dressing[InverseProp, {A, A}, 1, {p1_, p2_}] :> ZA[Sqrt[sp[p2, p2]]]*sp[p2, p2] + RB[k^2, sp[p2, p2]]*ZA[evP],

  (* Angle-resolved 3-point vertices on the (S0, S1, SPhi) grid; the feed carries the loop
     momentum. *)
  dressing[GammaN, {A, cb, c}, 1, {p1_, p2_, p3_}] :> ZAcbc[p1, p2],
  dressing[GammaN, {A, A, A}, 1, {p1_, p2_, p3_}] :> ZA3[p1, p2],

  ZAcbc[p1_, p2_] :> ZAcbc @@ P3Patt[p1, p2, -p1 - p2],
  ZA3[p1_, p2_] :> ZA3 @@ P3Patt[p1, p2, -p1 - p2],
  ZA4SP[p1_, p2_, p3_] :> ZA4SP @@ SP4Patt[p1, p2, p3, -p1 - p2 - p3],

  (* The gluon regulator is evaluated at evP = (k^n + 1)^(1/n) rather than at k, which keeps ZA
     away from the deep IR of its own grid; devP is d(evP)/dk. *)
  nZA -> 6,
  evP :> (k^nZA + 1)^(1/nZA),
  devP :> k^(-1 + nZA)*(1 + k^nZA)^(-1 + 1/nZA),

  dressing[Rdot, {A, A}, 1, {p1_, p2_}] :> ZA[evP]*RBdot[k^2, sp[p2, p2]] +
     RB[k^2, sp[p2, p2]]*(dtZA[evP] + k*devP*((ZA[1.02*evP] - ZA[evP])/(0.02*evP))),
  dressing[Rdot, {cb, c}, 1, {p1_, p2_}] :> Zc[k]*RBdot[k^2, sp[p2, p2]] +
     RB[k^2, sp[p2, p2]]*(dtZc[k] + k*((Zc[1.02*k] - Zc[k])/(0.02*k)))
};

(* Vertex flows: the 4-gluon enters at its symmetric point. *)
dressingRules[expr_] := ReplaceRepeated[expr,
  Join[{dressing[GammaN, {A, A, A, A}, 1, {p1_, p2_, p3_, p4_}] :> ZA4SP[p1, p2, p3]}, commonDressing]];

(* Gluon self-energy: the 4-gluon tadpole is fed through the angle-resolved ZA4tadpole, taking one
   external (leg 1) and one loop (leg 3) momentum. *)
dressingRulesAA[expr_] := ReplaceRepeated[expr,
  Join[{dressing[GammaN, {A, A, A, A}, 1, {p1_, p2_, p3_, p4_}] :> ZA4tadpole @@ tadpolePatt[p1, p3]}, commonDressing]];

FSetSymmetricDressing[GammaN, {A, A}];

(* Scalar products for the 1D (propagator) kernels. *)
PropParam[expr_] := UseLorentzLinearity[expr] //. {
   lf1 -> l1, (* no frequency direction in vacuum *)
   sp[p1, p1] -> p^2, sp[l1, l1] -> l1^2,
   sp[l1, p1] -> l1*p*cos[p, l1],
   sp[p1, l1] -> l1*p*cos[p, l1],
   Sqrt[(a_)^2] :> a, ((a_)^2)^((n_)/2) :> a^n,
   cos[l1, p] :> cos1};

(* Symmetric-point reduction, for ZA4SP only. *)
SP4FormRule = FMakeSPFormRule[{l1, lf1}, p, {p1, p2, p3, p4}];
SPParam[expr_] := UseLorentzLinearity[expr] //. {
   lf1 -> l1,
   sp[p, p] -> p^2, sp[l1, l1] -> l1^2,
   sp[l1, p1] -> p*l1*cos[l1, p1], sp[l1, p2] -> p*l1*cos[l1, p2],
   sp[l1, p3] -> p*l1*cos[l1, p3], sp[l1, p4] -> p*l1*cos[l1, p4],
   Sqrt[(a_)^2] :> a, ((a_)^2)^((n_)/2) :> a^n,
   Power[Power[l1_, 2], Rational[n_, 2]] :> l1^n,
   cos[l1, p1] :> cosl1p1, cos[l1, p2] :> cosl1p2,
   cos[l1, p3] :> cosl1p3, cos[l1, p4] :> cosl1p4};

(* Angle-resolved reduction for the 3D vertices, over the external pair (p1, p2, cosp1p2). Full 4D
   products throughout (vacuum); the loop-external cosines are declared in the kernel body. *)
P3Param[expr_] := UseLorentzLinearity[expr] //. {
   lf1 -> l1,
   sp[p1, p1] -> p1^2, sp[p2, p2] -> p2^2,
   sp[p1, p2] -> p1*p2*cosp1p2, sp[p2, p1] -> p1*p2*cosp1p2,
   sp[l1, l1] -> l1^2,
   sp[l1, p1] -> l1*p1*cosl1p1, sp[p1, l1] -> l1*p1*cosl1p1,
   sp[l1, p2] -> l1*p2*cosl1p2, sp[p2, l1] -> l1*p2*cosl1p2,
   Sqrt[(a_)^2] :> a, ((a_)^2)^((n_)/2) :> a^n,
   Power[Power[l1_, 2], Rational[n_, 2]] :> l1^n,
   cos[l1, p1] :> cosl1p1, cos[l1, p2] :> cosl1p2};


(* ::Subsection::Closed:: *)
(*Code setup*)


SetNc[3];
$Assumptions = k > 0 && p > 0 && p1 > 0 && p2 > 0 && l1 > 0 && q > 0 &&
   -1 < cos1 < 1 && -1 < cos2 < 1 && -1 < cos3 < 1 && -1 < cosp1p2 < 1;

(* Both momentum axes -- the 1D propagator grid and the S0 axis of the vertex grid -- cluster
   their points around an interior scale (FocusedLogCoordinates1D in DiFfRG) so that the ~1 GeV
   structure of the dressings is resolved instead of being smeared over two or three cells. The
   types appear both as the grid a kernel is mapped over and inside the interpolator type of every
   dressing it reads, so they are named once here; all of them must agree with model.hh or the
   generated map() signatures will not match.

   Spline (C^2) interpolation on the 1D dressings is load-bearing: linear interpolation of the
   propagators injects kinks that block the deep-IR scaling trajectory. *)
coordinatesType = "FocusedLogCoordinates1D<double>";
coordinates3DType = "FocusedLogLinLinPeriodicCoordinates";
interpolatorType = "SplineInterpolator1D<double, " <> coordinatesType <> ">";
interpolator3DType = "LinearInterpolatorND<double, " <> coordinates3DType <> ">";

kernelParameterList = {
  <|"Name" -> "k", "Type" -> "double"|>,
  (* strong couplings *)
  <|"Name" -> "ZA3", "Type" -> interpolator3DType, "Const" -> True, "Reference" -> True|>,
  <|"Name" -> "ZAcbc", "Type" -> interpolator3DType, "Const" -> True, "Reference" -> True|>,
  <|"Name" -> "ZA4SP", "Type" -> interpolatorType, "Const" -> True, "Reference" -> True|>,
  <|"Name" -> "ZA4tadpole", "Type" -> interpolator3DType, "Const" -> True, "Reference" -> True|>,
  (* ghost propagator *)
  <|"Name" -> "dtZc", "Type" -> interpolatorType, "Const" -> True, "Reference" -> True|>,
  <|"Name" -> "Zc", "Type" -> interpolatorType, "Const" -> True, "Reference" -> True|>,
  (* gluon propagator *)
  <|"Name" -> "dtZA", "Type" -> interpolatorType, "Const" -> True, "Reference" -> True|>,
  <|"Name" -> "ZA", "Type" -> interpolatorType, "Const" -> True, "Reference" -> True|>
};

(* Options shared by every kernel below, so that a change of device or scalar type is a one-line
   change rather than six. *)
commonKernelOptions = {
  "d" -> 4,
  "AD" -> False,
  "ComputeType" -> "double",
  "Device" -> "GPU",
  "Type" -> "double",
  "Parameters" -> kernelParameterList
};
(* The two grids a kernel can be mapped over. *)
grid1D = {"Coordinates" -> {coordinatesType}, "CoordinateArguments" -> {"p"}};
grid3D = {"Coordinates" -> {coordinates3DType}, "CoordinateArguments" -> {"S0", "S1", "SPhi"}};

SP4Defs = DeclareSymmetricPoints4DP4[l1, p, {p1, p2, p3, p4}];

(* Kernel body for the (S0, S1, SPhi)-gridded 3-point vertices: declares the external magnitudes
   p1, p2, the external angle cosp1p2 and the loop-external cosines cosl1p1, cosl1p2 in terms of
   the grid coordinates and the loop integration angles. phi drops out, so a 2-angle integrator
   suffices. *)
S0S1SPhiDef = Module[{rp1, rp2, rcosp1p2, rcosl1p1, rcosl1p2},
  rcosl1p1 = FullSimplify[transf3PTo[sp[p1, l1]/(l1 Sqrt[sp[p1, p1]])]];
  rcosl1p2 = FullSimplify[transf3PTo[sp[p2, l1]/(l1 Sqrt[sp[p2, p2]])]];
  rcosp1p2 = FullSimplify[transf3PTo[sp[p1, p2]/Sqrt[sp[p1, p1] sp[p2, p2]]]];
  rp1 = FullSimplify[transf3PTo[Sqrt[sp[p1, p1]]]];
  rp2 = FullSimplify[transf3PTo[Sqrt[sp[p2, p2]]]];
  "const double p1 = " <> FunKit`CppForm[rp1] <> ";\n" <>
  "const double p2 = " <> FunKit`CppForm[rp2] <> ";\n" <>
  "const double cosp1p2 = " <> FunKit`CppForm[rcosp1p2] <> ";\n" <>
  "const double cosl1p1 = " <> FunKit`CppForm[rcosl1p1] <> ";\n" <>
  "const double cosl1p2 = " <> FunKit`CppForm[rcosl1p2] <> ";"];

(* The same, for the angle-resolved 4-gluon tadpole, in the Qk (one external + one loop momentum)
   parametrisation. *)
TadpoleDef = Module[{rp1, rp2, rcosp1p2, rcosl1p1, rcosl1p2},
  rcosl1p1 = FullSimplify[transfTadpoleTo[sp[p1, q]/(q Sqrt[sp[p1, p1]])]]; (* = cos(p1, loop); loop-magnitude-free *)
  rcosl1p2 = FullSimplify[transfTadpoleTo[sp[p2, q]/(q Sqrt[sp[p2, p2]])]];
  rcosp1p2 = FullSimplify[transfTadpoleTo[sp[p1, p2]/Sqrt[sp[p1, p1] sp[p2, p2]]]];
  rp1 = FullSimplify[transfTadpoleTo[Sqrt[sp[p1, p1]]]];
  rp2 = FullSimplify[transfTadpoleTo[Sqrt[sp[p2, p2]]]];
  "const double p1 = " <> FunKit`CppForm[rp1] <> ";\n" <>
  "const double p2 = " <> FunKit`CppForm[rp2] <> ";\n" <>
  "const double cosp1p2 = " <> FunKit`CppForm[rcosp1p2] <> ";\n" <>
  "const double cosl1p1 = " <> FunKit`CppForm[rcosl1p1] <> ";\n" <>
  "const double cosl1p2 = " <> FunKit`CppForm[rcosl1p2] <> ";"];

FSetRegisterSize[64];


(* ::Section:: *)
(*Propagators*)


(* ::Subsection::Closed:: *)
(*ZA*)


fRGAA = FTakeDerivatives[WetterichEquation, {A[i1], A[i2]}] // FTruncate // FPlot // FRoute // FPrint;
traceExprAA = FTerm[TBGetProjector["AA", 1, {i1, i2} /. fRGAA["1-Loop"]["ExternalIndices"]]] **
   (fRGAA["1-Loop"]["Expression"] /. diagRules);
(* dressingRulesAA feeds the 4-gluon tadpole via the angle-resolved ZA4tadpole(external, loop). *)
FlowAA = FormTrace[traceExprAA] // dressingRulesAA // PropParam // Simplify;

MakeKernel[FlowAA/p^2,
  "Name" -> "ZA",
  "Integrator" -> "Integrator_p2_1ang",
  "IntegrationVariables" -> {"l1", "cos1"},
  Sequence @@ grid1D, Sequence @@ commonKernelOptions];


(* ::Subsection::Closed:: *)
(*Zc*)


fRGcbc = FTakeDerivatives[WetterichEquation, {cb[i1], c[i2]}] // FTruncate // FPlot // FRoute // FPrint;
traceExprcbc = FTerm[TBGetProjector["cbc", 1, {i1, i2} /. fRGcbc["1-Loop"]["ExternalIndices"]]] **
   (fRGcbc["1-Loop"]["Expression"] /. diagRules);
Flowcbc = FormTrace[traceExprcbc] // dressingRules // PropParam // Simplify;

(* Note the MINUS: the ghost dressing is defined with one (dressing[GammaN,{cb,c},...] ->
   -Zc |p|^2 above), so the projection has to carry the compensating sign. Dropping it flips the
   sign of the entire ghost flow. *)
MakeKernel[-(Flowcbc/p^2),
  "Name" -> "Zc",
  "Integrator" -> "Integrator_p2_1ang",
  "IntegrationVariables" -> {"l1", "cos1"},
  Sequence @@ grid1D, Sequence @@ commonKernelOptions];


(* ::Section:: *)
(*Strong interactions*)


(* ::Subsection::Closed:: *)
(*ZAcbc (angle-resolved)*)


fRGAcbc = FTakeDerivatives[WetterichEquation, {A[i1], cb[i2], c[i3]}] // FTruncate // FSimplify // FPlot // FRoute // FPrint;
traceExprAcbc = FTerm[TBGetProjector["Acbc", 1, {i1, i2, i3} /. fRGAcbc["1-Loop"]["ExternalIndices"]]] **
   (fRGAcbc["1-Loop"]["Expression"] /. diagRules);
FlowAcbc = FormTrace["ZAcbc", traceExprAcbc] // dressingRules // P3Param // Simplify;

MakeKernel[FlowAcbc,
  "Name" -> "ZAcbc",
  "Integrator" -> "Integrator_p2_4D_2ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2"},
  "KernelBody" -> S0S1SPhiDef,
  Sequence @@ grid3D, Sequence @@ commonKernelOptions];


(* ::Subsection::Closed:: *)
(*ZA3 (angle-resolved)*)


fRGA3 = FTakeDerivatives[WetterichEquation, {A[i1], A[i2], A[i3]}] // FTruncate // FPlot // FRoute // FPrint;
traceExprA3 = FTerm[TBGetProjector["AAAClassTrans", 1, {i1, i2, i3} /. fRGA3["1-Loop"]["ExternalIndices"]]] **
   (fRGA3["1-Loop"]["Expression"] /. diagRules);
FlowA3 = FormTrace["ZA3", traceExprA3] // dressingRules // P3Param // Simplify;

MakeKernel[FlowA3,
  "Name" -> "ZA3",
  "Integrator" -> "Integrator_p2_4D_2ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2"},
  "KernelBody" -> S0S1SPhiDef,
  Sequence @@ grid3D, Sequence @@ commonKernelOptions];


(* ::Subsection::Closed:: *)
(*ZA4tadpole (angle-resolved, Qk)*)


(* The same 4-gluon diagrams as ZA4SP below, but at the tadpole leg configuration
   {p1, p2, -p1, -p2} and gridded over the (S0, S1, SPhi) of the (external, loop) pair. This is
   the dressing fed into the gluon self-energy. *)
fRGA4t = FTakeDerivatives[WetterichEquation, {A[i1], A[i2], A[i3], A[i4]}] // FTruncate // FPlot // FRoute // FPrint;
projectorA4t = FullSimplify[UseLorentzLinearity[
   FTerm[TBGetProjector["AAAAClassTrans", 1, {i1, i2, i3, i4} /. fRGA4t["1-Loop"]["ExternalIndices"]]] //.
     {p3 -> -p1, p4 -> -p2}]];
traceExprA4t = FullSimplify[UseLorentzLinearity[
   (projectorA4t ** (fRGA4t["1-Loop"]["Expression"] /. diagRules)) //. {p3 -> -p1, p4 -> -p2}]];
FlowA4t = FormTrace["ZA4tadpole", traceExprA4t] // dressingRules // P3Param // Simplify;

MakeKernel[FlowA4t,
  "Name" -> "ZA4tadpole",
  "Integrator" -> "Integrator_p2_4D_2ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2"},
  "KernelBody" -> TadpoleDef,
  Sequence @@ grid3D, Sequence @@ commonKernelOptions];


(* ::Subsection::Closed:: *)
(*ZA4SP (symmetric point)*)


fRGA4 = FTakeDerivatives[WetterichEquation, {A[i1], A[i2], A[i3], A[i4]}] // FTruncate // FPlot // FRoute // FPrint;
projectorA4 = FTerm[TBGetProjector["AAAAClassTrans", 1, {i1, i2, i3, i4} /. fRGA4["1-Loop"]["ExternalIndices"]]] //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3, p4] & // Simplify;
traceExprA4 = projectorA4 ** (fRGA4["1-Loop"]["Expression"] /. diagRules) //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3, p4] &;
FlowA4 = FormTrace["ZA4SP", traceExprA4, {}, SP4FormRule] // dressingRules //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3, p4] & // SPParam // Simplify;

MakeKernel[FlowA4,
  "Name" -> "ZA4SP",
  "Integrator" -> "Integrator_p2_4D_3ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2", "phi"},
  "KernelBody" -> SP4Defs,
  Sequence @@ grid1D, Sequence @@ commonKernelOptions];


(* ::Section:: *)
(*Finalize*)


UpdateFlows["YangMillsFlows"];
