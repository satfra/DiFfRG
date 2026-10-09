(* ::Package:: *)

(* ::Text:: *)
(*============================================================================================ *)
(*SU(3) Yang-Mills theory*)
(*---------------------------------------------------------------------------------------------------------------------------------------------------*)
(*All vertices evaluated on the symmetric point, so every dressing lives on one momentum grid.*)
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


(* The Feynman rules are setup-dependent but momentum-independent, so they are built once here
   rather than re-derived inside every trace below. *)
diagRules = FMakeDiagrammaticRules[];


(* ::Subsection::Closed:: *)
(*Parametrizations*)


(* Symmetric-point momentum variable: the single scale that a 3- or 4-point dressing is
   tabulated over. *)
SP3Patt[p1e_, p2e_, p3e_] :=
  FullSimplify[UseLorentzLinearity[{Sqrt[(sp[p1, p1] + sp[p2, p2] + sp[p3, p3])/3]} /.
     {p1 :> p1e, p2 :> p2e, p3 :> p3e}]];
SP4Patt[p1e_, p2e_, p3e_, p4e_] :=
  FullSimplify[UseLorentzLinearity[{Sqrt[(sp[p1, p1] + sp[p2, p2] + sp[p3, p3] + sp[p4, p4])/4]} /.
     {p1 :> p1e, p2 :> p2e, p3 :> p3e, p4 :> p4e}]];

(* Rule ORDER matters here (//.): the dressings must be substituted before the symmetric-point
   parametrization collapses their arguments to one scale. *)
dressingRules[expr_] := ReplaceRepeated[expr, {

  (* Propagators *)
  dressing[GammaN, {cb, c}, 1, {p1_, p2_}] :> -Zc[Sqrt[sp[p2, p2]]]*sp[p2, p2],
  dressing[GammaN, {A, A}, 1, {p1_, p2_}] :> ZA[Sqrt[sp[p2, p2]]]*sp[p2, p2],
  (* Inverse propagators (these carry the regulator) *)
  dressing[InverseProp, {cb, c}, 1, {p1_, p2_}] :> -(Zc[Sqrt[sp[p2, p2]]]*sp[p2, p2] + RB[k^2, sp[p2, p2]]*Zc[k]),
  dressing[InverseProp, {A, A}, 1, {p1_, p2_}] :> ZA[Sqrt[sp[p2, p2]]]*sp[p2, p2] + RB[k^2, sp[p2, p2]]*ZA[evP],

  (* Strong couplings *)
  dressing[GammaN, {A, cb, c}, 1, {p1_, p2_, p3_}] :> ZAcbc[p1, p2],
  dressing[GammaN, {A, A, A}, 1, {p1_, p2_, p3_}] :> ZA3[p1, p2],
  dressing[GammaN, {A, A, A, A}, 1, {p1_, p2_, p3_, p4_}] :> ZA4[p1, p2, p3],

  (* Symmetric-point parametrizations *)
  ZAcbc[p1_, p2_] :> ZAcbc @@ SP3Patt[p1, p2, -p1 - p2],
  ZA3[p1_, p2_] :> ZA3 @@ SP3Patt[p1, p2, -p1 - p2],
  ZA4[p1_, p2_, p3_] :> ZA4 @@ SP4Patt[p1, p2, p3, -p1 - p2 - p3],

  (* The gluon regulator is evaluated at evP = (k^n + 1)^(1/n) rather than at k, which keeps
     ZA away from the deep IR of its own grid; devP is d(evP)/dk. *)
  nZA -> 6,
  evP :> (k^nZA + 1)^(1/nZA),
  devP :> k^(-1 + nZA)*(1 + k^nZA)^(-1 + 1/nZA),

  (* Regulator derivatives. The finite differences supply the scale derivative of the dressing
     at the evaluation point, which is not itself a flowing variable. *)
  dressing[Rdot, {A, A}, 1, {p1_, p2_}] :> ZA[evP]*RBdot[k^2, sp[p2, p2]] +
     RB[k^2, sp[p2, p2]]*(dtZA[evP] + k*devP*((ZA[1.02*evP] - ZA[evP])/(0.02*evP))),
  dressing[Rdot, {cb, c}, 1, {p1_, p2_}] :> Zc[k]*RBdot[k^2, sp[p2, p2]] +
     RB[k^2, sp[p2, p2]]*(dtZc[k] + k*((Zc[1.02*k] - Zc[k])/(0.02*k)))
  }];

FSetSymmetricDressing[GammaN, {A, A}];

(* Vacuum 1-angle scalar reduction for the propagators: sp/cos -> {l1, p, cos1}. *)
PropParam[expr_] := UseLorentzLinearity[expr] //. {
   lf1 -> l1, (* no frequency direction in vacuum *)
   sp[p1, p1] -> p^2, sp[l1, l1] -> l1^2,
   sp[l1, p1] -> l1*p*cos[p, l1],
   sp[p1, l1] -> l1*p*cos[p, l1],
   Sqrt[(a_)^2] :> a, ((a_)^2)^((n_)/2) :> a^n,
   cos[l1, p] :> cos1};

SP3FormRule = FMakeSPFormRule[{l1, lf1}, p, {p1, p2, p3}];
SP4FormRule = FMakeSPFormRule[{l1, lf1}, p, {p1, p2, p3, p4}];

(* Vacuum multi-angle reduction for the 3-/4-point vertices: the loop-external cosines
   cosl1p{1..4} stay symbolic and are resolved by the symmetric-point definitions injected as
   the kernel body (SP3Defs/SP4Defs). *)
SPParam[expr_] := UseLorentzLinearity[expr] //. {
   lf1 -> l1,
   sp[p, p] -> p^2, sp[l1, l1] -> l1^2,
   sp[l1, p1] -> p*l1*cos[l1, p1], sp[l1, p2] -> p*l1*cos[l1, p2],
   sp[l1, p3] -> p*l1*cos[l1, p3], sp[l1, p4] -> p*l1*cos[l1, p4],
   Sqrt[(a_)^2] :> a, ((a_)^2)^((n_)/2) :> a^n,
   Power[Power[l1_, 2], Rational[n_, 2]] :> l1^n,
   cos[l1, p1] :> cosl1p1, cos[l1, p2] :> cosl1p2,
   cos[l1, p3] :> cosl1p3, cos[l1, p4] :> cosl1p4};


(* ::Subsection::Closed:: *)
(*Code setup*)


SetNc[3];
$Assumptions = k > 0 && p > 0 && l1 > 0 && -1 < cos1 < 1 && -1 < cos2 < 1 && -1 < cos3 < 1;

(* The momentum grid clusters its points around an interior scale (FocusedLogCoordinates1D in
   DiFfRG) so that the ~1 GeV structure of the dressings is resolved instead of being smeared
   over two or three cells. The type appears twice -- as the grid the kernels are mapped over and
   inside the interpolator type of every dressing they read -- so it is named once here; both
   must agree with model.hh or the generated map() signatures will not match. *)
coordinatesType = "FocusedLogCoordinates1D<double>";
interpolatorType = "SplineInterpolator1D<double, " <> coordinatesType <> ">";

dressings = {ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA};
kernelParameterList = Join[
  {<|"Name" -> "k", "Type" -> "double"|>},
  Table[<|"Name" -> ToString[d], "Type" -> interpolatorType, "Const" -> True, "Reference" -> True|>, {d, dressings}]
];

(* Options shared by every kernel below, so that a change to the device or the scalar type is a
   one-line change rather than five. *)
commonKernelOptions = {
  "d" -> 4,
  "AD" -> False,
  "ComputeType" -> "double",
  "Device" -> "GPU",
  "Type" -> "double",
  "Parameters" -> kernelParameterList,
  "Coordinates" -> {coordinatesType},
  "CoordinateArguments" -> {"p"}
};

SP3Defs = DeclareSymmetricPoints4DP3[l1, p, {p1, p2, p3}];
SP4Defs = DeclareSymmetricPoints4DP4[l1, p, {p1, p2, p3, p4}];

FSetRegisterSize[64];


(* ::Section:: *)
(*Propagators*)


(* ::Subsection::Closed:: *)
(*ZA*)


fRGAA = FTakeDerivatives[WetterichEquation, {A[i1], A[i2]}] // FTruncate // FPlot // FRoute // FPrint;
traceExprAA = FTerm[TBGetProjector["AA", 1, {i1, i2} /. fRGAA["1-Loop"]["ExternalIndices"]]] **
   (fRGAA["1-Loop"]["Expression"] /. diagRules);
FlowAA = FormTrace[traceExprAA] // dressingRules // PropParam // Simplify;

MakeKernel[FlowAA/p^2,
  "Name" -> "ZA",
  "Integrator" -> "Integrator_p2_1ang",
  "IntegrationVariables" -> {"l1", "cos1"},
  Sequence @@ commonKernelOptions];


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
  Sequence @@ commonKernelOptions];


(* ::Section:: *)
(*Strong interactions*)


(* ::Subsection::Closed:: *)
(*ZAcbc*)


fRGAcbc = FTakeDerivatives[WetterichEquation, {A[i1], cb[i2], c[i3]}] // FTruncate // FSimplify // FPlot // FRoute // FPrint;
traceExprAcbc = FTerm[TBGetProjector["Acbc", 1, {i1, i2, i3} /. fRGAcbc["1-Loop"]["ExternalIndices"]]] **
   (fRGAcbc["1-Loop"]["Expression"] /. diagRules);
FlowAcbc = FormTrace[traceExprAcbc, {}, SP3FormRule] // dressingRules //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3] & // SPParam // Simplify;

MakeKernel[FlowAcbc,
  "Name" -> "ZAcbc",
  "Integrator" -> "Integrator_p2_4D_2ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2"},
  "KernelBody" -> SP3Defs,
  Sequence @@ commonKernelOptions];


(* ::Subsection::Closed:: *)
(*ZA3*)


fRGA3 = FTakeDerivatives[WetterichEquation, {A[i1], A[i2], A[i3]}] // FTruncate // FPlot // FRoute // FPrint;
projectorA3 = FTerm[TBGetProjector["AAAClassTrans", 1, {i1, i2, i3} /. fRGA3["1-Loop"]["ExternalIndices"]]] //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3] & // Simplify;
traceExprA3 = projectorA3 ** (fRGA3["1-Loop"]["Expression"] /. diagRules);
FlowA3 = FormTrace["ZA3", traceExprA3, {}, SP3FormRule] // dressingRules //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3] & // SPParam // Simplify;

MakeKernel[FlowA3,
  "Name" -> "ZA3",
  "Integrator" -> "Integrator_p2_4D_2ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2"},
  "KernelBody" -> SP3Defs,
  Sequence @@ commonKernelOptions];


(* ::Subsection::Closed:: *)
(*ZA4*)


fRGA4 = FTakeDerivatives[WetterichEquation, {A[i1], A[i2], A[i3], A[i4]}] // FTruncate // FPlot // FRoute // FPrint;
projectorA4 = FTerm[TBGetProjector["AAAAClassTrans", 1, {i1, i2, i3, i4} /. fRGA4["1-Loop"]["ExternalIndices"]]] //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3, p4] & // Simplify;
traceExprA4 = projectorA4 ** (fRGA4["1-Loop"]["Expression"] /. diagRules) //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3, p4] &;
FlowA4 = FormTrace["ZA4", traceExprA4, {}, SP4FormRule] // dressingRules //
   TBProjectToSymmetricPoint[#, l1, p, p1, p2, p3, p4] & // SPParam // Simplify;

MakeKernel[FlowA4,
  "Name" -> "ZA4",
  "Integrator" -> "Integrator_p2_4D_3ang",
  "IntegrationVariables" -> {"l1", "cos1", "cos2", "phi"},
  "KernelBody" -> SP4Defs,
  Sequence @@ commonKernelOptions];


(* ::Section:: *)
(*Finalize*)


UpdateFlows["YangMillsFlows"];
