(* ::Package:: *)

(* ::Title:: *)

(*DiFfRG: Code Tools package*)

(* ::Chapter:: *)

(*Package Setup*)

(* ::Section:: *)

(*Setup and exports*)

BeginPackage["DiFfRG`CodeTools`", {"DiFfRG`CodeTools`MakeKernel`", "DiFfRG`CodeTools`Directory`", "DiFfRG`CodeTools`Export`"}]; (* TODO: Remove the export function from here later on *)

Unprotect["DiFfRG`CodeTools`*"];

ClearAll["DiFfRG`CodeTools`*"];

ClearAll["DiFfRG`CodeTools`Private`*"];

UpdateFlows::usage = "UpdateFlows[\"Name\"] updates the Flow Class with Name \"Name\" and the CMakeLists.txt file.";

FlowKernel::usage = "FlowKernel[expr_,name_String,NT_String:\"auto\",addprefix_String:\"\"]
Makes an equation into a lambda expression - of limited usefulness, but can be used together with LoopIntegrals::integrate and similar functions.";

CodeForm::usage = "CodeForm[expr_]
Obtain properly formatted and processed C++ code from an expression.";

SetCppNames::usage = "SetCppNames[rules___]
Set additional replacement rules used when invoking CodeForm[expr].

Example Call: SetCppNames[\"k\"->\"k_bosonic\", \"Arccos(\"->\"std::arccos(\"]";

SafeFiniteTFunctions::usage = "";

DeclareSymmetricPoints4DP4::sanityFailed = "Sanity check failed for 4D symmetric 4-point configuration: not all dot products equal -1/3.";

DeclareSymmetricPoints4DP3::sanityFailed = "Sanity check failed for 4D symmetric 3-point configuration: not all dot products equal -1/2.";

DeclareSymmetricPoints3DP4::sanityFailed = "Sanity check failed for 3D symmetric 4-point configuration: not all dot products equal -1/3.";

DeclareSymmetricPoints3DP3::sanityFailed = "Sanity check failed for 3D symmetric 3-point configuration: not all dot products equal -1/2.";

DeclareSymmetricPoints2DP3::sanityFailed = "Sanity check failed for 2D symmetric 3-point configuration: not all dot products equal -1/2.";

DeclareSymmetricPoints4DP4::usage = "DeclareSymmetricPoints4DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 4D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints4DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints4DP3::usage = "DeclareSymmetricPoints4DP3[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 4D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints4DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints3DP4::usage = "DeclareSymmetricPoints3DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 3D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints3DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints3DP3::usage = "DeclareSymmetricPoints3DP3[]
Obtain C++ code declaring angles for a three-point symmetric configuration in 3D.
The angles will have the names cosp1q, cosp2q and cosp3q.
DeclareSymmetricPoints3DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints2DP4::usage = "DeclareSymmetricPoints2DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 2D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints2DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints2DP3::usage = "DeclareSymmetricPoints2DP3[]
Obtain C++ code declaring angles for a three-point symmetric configuration in 2D.
The angles will have the names cosp1q, cosp2q and cosp3q.
DeclareSymmetricPoints2DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareAnglesP34Dpqr::usage = "DeclareAnglesP34Dpqr[q,p,r]
Obtain C++ code declaring angles for the angles in a full three-point function in 4D.
The angles will have the names cospq and cosqr, where q,p,r are replaced by the given Symbol names and ordered alphabetically.
DeclareAnglesP34Dpqr[q,p,r,computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareAnglesP33Dpqr::usage = "DeclareAnglesP33Dpqr[q,p,r]
Obtain C++ code declaring angles for the angles in a full three-point function in 3D.
The angles will have the names cospq and cosqr, where q,p,r are replaced by the given Symbol names and ordered alphabetically.
DeclareAnglesP34Dpqr[q,p,r,computeType]
Set the type of the declared C++ variables (should be double or float).";

$CodeToolsDirectory = DirectoryName[$InputFileName];

Begin["`Private`"];

Needs["DiFfRG`CodeTools`MakeKernel`"];

(* ::Chapter:: *)

(*General Definitions and structural methods*)

(* ::Section:: *)

(*Momentum Configurations*)

(* ::Subsection::Closed:: *)

(*4D*)

(* ::Input::Initialization:: *)

DeclareAnglesP34Dpqr[q_, p_, r_, cos1_ : Symbol @ "cos1", cos2_ : Symbol @ "cos2", phi_ : Symbol @ "phi", computeType_String:"double"] :=
  Module[{vec4, Vectorp, Vectorr, Vectorq, cos, Resultcospq, Resultcosqr, namecospq, namecosqr, code, file},
    vec4[\[CapitalTheta]1_, \[CapitalTheta]2_, \[Phi]_] := {Cos[\[CapitalTheta]1], Sin[\[CapitalTheta]1] Cos[\[CapitalTheta]2], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2] Cos[\[Phi]], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2] Sin[\[Phi]]};
    SetAttributes[cos, Orderless];
    Vectorp = vec4[0, 0, 0];
    Vectorr = vec4[ArcCos[cos[p, r]], 0, 0];
    Vectorq = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi];
    Resultcospq = Vectorq . Vectorp //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]] // FullSimplify;
    Resultcosqr = Vectorq . Vectorr //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]] // FullSimplify;
    namecospq = cos[p, q] //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]];
    namecosqr = cos[q, r] //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]];
    code = "const " <> computeType <> " " <> ToString[namecospq] <> " = " <> FunKit`CppForm[Resultcospq] <> ";\n" <> "const " <> computeType <> " " <> ToString[namecosqr] <> " = " <> FunKit`CppForm[Resultcosqr] <> ";";
    Return[code];
  ];

(* ::Input::Initialization:: *)

DeclareSymmetricPoints4DP4[q_, p_, {p1_, p2_, p3_, p4_}, cos1_ : Symbol @ "cos1", cos2_ : Symbol @ "cos2", phi_ : Symbol @ "phi", computeType_String:"double"] :=
  Module[{vec4, sanity, Vectors4DSP4, SymmetricPoint4DP4cosp1, SymmetricPoint4DP4cosp2, SymmetricPoint4DP4cosp3, SymmetricPoint4DP4cosp4, SymmetricPoint4DP4Code},
    vec4[\[CapitalTheta]1_, \[CapitalTheta]2_, \[Phi]_] := {Cos[\[CapitalTheta]1], Sin[\[CapitalTheta]1] Cos[\[CapitalTheta]2], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2] Cos[\[Phi]], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2] Sin[\[Phi]]};
    Vectors4DSP4 = {vec4[\[Pi] / 2, 0, 0], vec4[\[Pi] / 2, ArcCos[-(1/3)], 0], vec4[\[Pi] / 2, ArcCos[-(1/3)], 1 (2 \[Pi]) / 3], vec4[\[Pi] / 2, ArcCos[-(1/3)], 2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors4DSP4[[#[[1]]]] . Vectors4DSP4[[#[[2]]]]&, Subsets[{1, 2, 3, 4}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/3) // N)&],
      Message[DeclareSymmetricPoints4DP4::sanityFailed];
      Abort[];
    ];
    SymmetricPoint4DP4cosp1 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP4[[1]] // FullSimplify;
    SymmetricPoint4DP4cosp2 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP4[[2]] // FullSimplify;
    SymmetricPoint4DP4cosp3 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP4[[3]] // FullSimplify;
    SymmetricPoint4DP4cosp4 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP4[[4]] // FullSimplify;
    SymmetricPoint4DP4Code = "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p1}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP4cosp1] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p2}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP4cosp2] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p3}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP4cosp3] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p4}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP4cosp4] <> ";";
    Return[SymmetricPoint4DP4Code];
  ];

(* ::Input::Initialization:: *)

DeclareSymmetricPoints4DP3[q_, p_, {p1_, p2_, p3_}, cos1_ : Symbol @ "cos1", cos2_ : Symbol @ "cos2", phi_ : Symbol @ "phi", computeType_String:"double"] :=
  Module[{vec4, Vectors4DSP3, sanity, SymmetricPoint4DP3cosp1, SymmetricPoint4DP3cosp2, SymmetricPoint4DP3cosp3, SymmetricPoint4DP3Code},
    vec4[\[CapitalTheta]1_, \[CapitalTheta]2_, \[Phi]_] := {Cos[\[CapitalTheta]1], Sin[\[CapitalTheta]1] Cos[\[CapitalTheta]2], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2] Cos[\[Phi]], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2] Sin[\[Phi]]};
    Vectors4DSP3 = {vec4[0, 0, 0], vec4[(2 \[Pi]) / 3, 0, 0], vec4[(2 \[Pi]) / 3, \[Pi], 0]};
    sanity = Map[Vectors4DSP3[[#[[1]]]] . Vectors4DSP3[[#[[2]]]]&, Subsets[{1, 2, 3}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/2) // N)&],
      Message[DeclareSymmetricPoints4DP3::sanityFailed];
      Abort[];
    ];
    SymmetricPoint4DP3cosp1 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP3[[1]] // FullSimplify;
    SymmetricPoint4DP3cosp2 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP3[[2]] // FullSimplify;
    SymmetricPoint4DP3cosp3 = vec4[ArcCos[Symbol @ SymbolName @ cos1], ArcCos[Symbol @ SymbolName @ cos2], Symbol @ SymbolName @ phi] . Vectors4DSP3[[3]] // FullSimplify;
    SymmetricPoint4DP3Code = "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p1}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP3cosp1] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p2}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP3cosp2] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p3}]) <> " = " <> FunKit`CppForm[SymmetricPoint4DP3cosp3] <> ";";
    Return[SymmetricPoint4DP3Code]
  ];

(* ::Subsection::Closed:: *)

(*3D*)

(* ::Input::Initialization:: *)

DeclareAnglesP33Dpqr[q_, p_, r_, cos1_ : Symbol @ "cos1", phi_ : Symbol @ "phi", computeType_String:"double"] :=
  Module[{vec3, Vectorp, Vectorr, Vectorq, cos, Resultcospq, Resultcosqr, namecospq, namecosqr, code, file},
    vec3[\[CapitalTheta]_, \[Phi]_] := {Sin[\[CapitalTheta]] Cos[\[Phi]], Sin[\[CapitalTheta]] Sin[\[Phi]], Cos[\[CapitalTheta]]};
    SetAttributes[cos, Orderless];
    Vectorp = vec3[0, 0];
    Vectorr = vec3[ArcCos[cos[p, r]], 0];
    Vectorq = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi];
    Resultcospq = Vectorq . Vectorp //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]] // FullSimplify;
    Resultcosqr = Vectorq . Vectorr //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]] // FullSimplify;
    namecospq = cos[p, q] //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]];
    namecosqr = cos[q, r] //. cos[a_, b_] :> Symbol["cos" <> ToString[a] <> ToString[b]];
    code = "const " <> computeType <> " " <> ToString[namecospq] <> " = " <> FunKit`CppForm[Resultcospq] <> ";\n" <> "const " <> computeType <> " " <> ToString[namecosqr] <> " = " <> FunKit`CppForm[Resultcosqr] <> ";";
    Return[code];
  ];

(* ::Input::Initialization:: *)

DeclareSymmetricPoints3DP3[q_, p_, {p1_, p2_, p3_}, cos1_ : Symbol @ "cos1", phi_ : Symbol @ "phi", computeType_String:"double"] :=
  Module[{vec3, Vectors3DSP3, sanity, SymmetricPoint3DP3cosp1, SymmetricPoint3DP3cosp2, SymmetricPoint3DP3cosp3, SymmetricPoint3DP3Code},
    vec3[\[CapitalTheta]_, \[Phi]_] := {Sin[\[CapitalTheta]] Cos[\[Phi]], Sin[\[CapitalTheta]] Sin[\[Phi]], Cos[\[CapitalTheta]]};
    Vectors3DSP3 = {vec3[\[Pi] / 2, 0], vec3[\[Pi] / 2, 1 (2 \[Pi]) / 3], vec3[\[Pi] / 2, 2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors3DSP3[[#[[1]]]] . Vectors3DSP3[[#[[2]]]]&, Subsets[{1, 2, 3}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/2) // N)&],
      Message[DeclareSymmetricPoints3DP3::sanityFailed];
      Abort[];
    ];
    SymmetricPoint3DP3cosp1 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP3[[1]] // FullSimplify;
    SymmetricPoint3DP3cosp2 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP3[[2]] // FullSimplify;
    SymmetricPoint3DP3cosp3 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP3[[3]] // FullSimplify;
    SymmetricPoint3DP3Code = "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p1}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP3cosp1] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p2}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP3cosp2] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p3}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP3cosp3] <> ";";
    Return[SymmetricPoint3DP3Code];
  ];

(* ::Input::Initialization:: *)

DeclareSymmetricPoints3DP4[q_, p_, {p1_, p2_, p3_, p4_}, cos1_ : Symbol @ "cos1", phi_ : Symbol @ "phi", computeType_String:"double"] :=
  Module[{vec3, sanity, Vectors3DSP4, SymmetricPoint3DP4cosp1, SymmetricPoint3DP4cosp2, SymmetricPoint3DP4cosp3, SymmetricPoint3DP4cosp4, SymmetricPoint3DP4Code},
    vec3[\[CapitalTheta]_, \[Phi]_] := {Sin[\[CapitalTheta]] Cos[\[Phi]], Sin[\[CapitalTheta]] Sin[\[Phi]], Cos[\[CapitalTheta]]};
    Vectors3DSP4 = {vec3[0, 0], vec3[ArcCos[-(1/3)], 0], vec3[ArcCos[-(1/3)], 1 (2 \[Pi]) / 3], vec3[ArcCos[-(1/3)], 2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors3DSP4[[#[[1]]]] . Vectors3DSP4[[#[[2]]]]&, Subsets[{1, 2, 3, 4}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/3) // N)&],
      Message[DeclareSymmetricPoints3DP4::sanityFailed];
      Abort[];
    ];
    SymmetricPoint3DP4cosp1 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP4[[1]] // FullSimplify;
    SymmetricPoint3DP4cosp2 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP4[[2]] // FullSimplify;
    SymmetricPoint3DP4cosp3 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP4[[3]] // FullSimplify;
    SymmetricPoint3DP4cosp4 = vec3[ArcCos[Symbol @ SymbolName @ cos1], Symbol @ SymbolName @ phi] . Vectors3DSP4[[4]] // FullSimplify;
    SymmetricPoint3DP4Code = "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p1}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp1] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p2}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp2] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p3}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp3] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p4}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp4] <> ";";
    Return[SymmetricPoint3DP4Code];
  ];

(* ::Subsection::Closed:: *)

(*2D*)

(* ::Input::Initialization:: *)

DeclareSymmetricPoints2DP3[q_, p_, {p1_, p2_, p3_}, cos1_ : Symbol @ "cos1", computeType_String:"double"] :=
  Module[{vec2, Vectors3DSP3, sanity, SymmetricPoint3DP3cosp1, SymmetricPoint3DP3cosp2, SymmetricPoint3DP3cosp3, SymmetricPoint3DP3Code},
    vec2[\[CapitalTheta]_] := {Cos[\[CapitalTheta]], Sin[\[CapitalTheta]]};
    Vectors3DSP3 = {vec2[0], vec2[1 (2 \[Pi]) / 3], vec2[2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors3DSP3[[#[[1]]]] . Vectors3DSP3[[#[[2]]]]&, Subsets[{1, 2, 3}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/2) // N)&],
      Message[DeclareSymmetricPoints2DP3::sanityFailed];
      Abort[];
    ];
    SymmetricPoint3DP3cosp1 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP3[[1]] // FullSimplify;
    SymmetricPoint3DP3cosp2 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP3[[2]] // FullSimplify;
    SymmetricPoint3DP3cosp3 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP3[[3]] // FullSimplify;
    SymmetricPoint3DP3Code = "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p1}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP3cosp1] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p2}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP3cosp2] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p3}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP3cosp3] <> ";";
    Return[SymmetricPoint3DP3Code];
  ];

(* ::Input::Initialization:: *)

DeclareSymmetricPoints2DP4[q_, p_, {p1_, p2_, p3_, p4_}, cos1_ : Symbol @ "cos1", computeType_String:"double"] :=
  Module[{vec2, sanity, Vectors3DSP4, SymmetricPoint3DP4cosp1, SymmetricPoint3DP4cosp2, SymmetricPoint3DP4cosp3, SymmetricPoint3DP4cosp4, SymmetricPoint3DP4Code},
    vec2[\[CapitalTheta]_] := {Cos[\[CapitalTheta]], Sin[\[CapitalTheta]]};
    Vectors3DSP4 = {vec2[0], vec2[1 (2 \[Pi]) / 4], vec2[2 (2 \[Pi]) / 4], vec2[3 (2 \[Pi]) / 4]};
    sanity = Map[Vectors3DSP4[[#[[1]]]] . Vectors3DSP4[[#[[2]]]]&, Subsets[{1, 2, 3, 4}, {2}]];
    SymmetricPoint3DP4cosp1 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP4[[1]] // FullSimplify;
    SymmetricPoint3DP4cosp2 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP4[[2]] // FullSimplify;
    SymmetricPoint3DP4cosp3 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP4[[3]] // FullSimplify;
    SymmetricPoint3DP4cosp4 = vec2[ArcCos[Symbol @ SymbolName @ cos1]] . Vectors3DSP4[[4]] // FullSimplify;
    SymmetricPoint3DP4Code = "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p1}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp1] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p2}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp2] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p3}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp3] <> ";\n" <> "const " <> computeType <> " cos" <> StringJoin @ (ToString /@ Sort[{q, p4}]) <> " = " <> FunKit`CppForm[SymmetricPoint3DP4cosp4] <> ";";
    Return[SymmetricPoint3DP4Code];
  ];

(* ::Section::Closed:: *)

(*Safe replacements for finite T*)

(* ::Input::Initialization:: *)

SafeFiniteTFunctions[expr_, T_] :=
  Module[{a},
    expr //. {Tanh[a_ / (2 T)] :> Symbol["TanhFiniteT"][a, T], Tanh[a_ / T] :> Symbol["TanhFiniteT"][a, 2 T], Tanh[a_ / (2 T)] ^ n_ :> Symbol["TanhFiniteT"][a, T] ^ n, Tanh[a_ / T] ^ n_ :> Symbol["TanhFiniteT"][a, 2 T] ^ n, Coth[a_ / (2 T)] :> Symbol["CothFiniteT"][a, T], Coth[a_ / T] :> Symbol["CothFiniteT"][a, 2 T], Coth[a_ / (2 T)] ^ n_ :> Symbol["CothFiniteT"][a, T] ^ n, Coth[a_ / T] ^ n_ :> Symbol["CothFiniteT"][a, 2 T] ^ n, Csch[a_ / (2 T)] :> Symbol["CschFiniteT"][a, T], Csch[a_ / T] :> Symbol["CschFiniteT"][a, 2 T], Csch[a_ / (2 T)] ^ n_ :> Symbol["CschFiniteT"][a, T] ^ n, Csch[a_ / T] ^ n_ :> Symbol["CschFiniteT"][a, 2 T] ^ n, Sech[a_ / (2 T)] :> Symbol["SechFiniteT"][a, T], Sech[a_ / T] :> Symbol["SechFiniteT"][a, 2 T], Sech[a_ / (2 T)] ^ n_ :> Symbol["SechFiniteT"][a, T] ^ n, Sech[a_ / T] ^ n_ :> Symbol["SechFiniteT"][a, 2 T] ^ n}
  ];

(* ::Chapter:: *)

(*Flow code generation*)

(* ::Section::Closed:: *)

(*Kernel predefines*)

(* ::Section::Closed:: *)

(*Code for CMake and flow class*)

UpdateFlows[varName_:flowName] :=
  Module[{},
    updateCMake[varName];
    updateFlowClass[varName];
  ];

updateCMake[varName_:"Flows"] :=
  Module[{folders, sources, cmake, fileName = FileNameJoin[{flowDir, "CMakeLists.txt"}], flowFolderName},
    flowFolderName = StringSplit[flowDir, "/"][[-1]];
    folders = Select[FileNames["*", flowDir, 1], DirectoryQ];
    sources = Flatten @ Join[Map[Get[# <> "/sources.m"]&, folders]];
    sources = "    " <> StringRiffle[sources, "\n    "];
    cmake = "set(" <> varName <> "_SOURCES 
" <> sources <> "
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(" <> varName <> " STATIC ${" <> varName <> "_SOURCES})
add_library(" <> varName <> "_nowarn INTERFACE)
target_compile_options(" <> varName <> "_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(" <> varName <> " DiFfRG::DiFfRG " <> varName <> "_nowarn)
";
    ExportCode[fileName, cmake];
  ];

(* ::Input::Initialization:: *)

updateFlowClass[varName_:"Flows"] :=
  Module[{folders, flowFolderName, integrators, flowHeader, flowCpp, integratorInitializations},
    flowFolderName = StringSplit[flowDir, "/"][[-1]];
    folders = Select[FileNames["*", flowDir, 1], DirectoryQ];
    integrators = Map[{StringSplit[#, "/"][[-1]], StringSplit[#, "/"][[-1]] <> "_integrator", "./" <> StringSplit[#, "/"][[-1]] <> "/" <> StringSplit[#, "/"][[-1]] <> ".hh"}&, folders];
    flowHeader = FunKit`MakeCppHeader["Includes" -> Join[{"DiFfRG/common/utils.hh", "DiFfRG/physics/integration.hh"}, integrators[[All, 3]]], "Body" -> {FunKit`MakeCppClass["Name" -> varName, "MembersPublic" -> Join[{FunKit`MakeCppFunction["Name" -> varName, "Parameters" -> {<|"Type" -> "DiFfRG::ConfigTree", "Reference" -> True, "Const" -> True, "Name" -> "config"|>}, "Body" -> None, "Return" -> ""], FunKit`MakeCppFunction["Name" -> "set_k", "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "k", "Reference" -> False|>}, "Body" -> None, "Return" -> "void"], FunKit`MakeCppFunction["Name" -> "set_T", "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "T", "Reference" -> False|>}, "Body" -> None, "Return" -> "void"], FunKit`MakeCppFunction["Name" -> "set_typical_E", "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "E", "Reference" -> False|>}, "Body" -> None, "Return" -> "void"], FunKit`MakeCppFunction["Name" -> "set_x_extent", "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "x_extent", "Reference" -> False|>}, "Body" -> None, "Return" -> "void"], "DiFfRG::QuadratureProvider quadrature_provider;"}, Map[#[[2]] <> " " <> #[[1]] <> ";"&, integrators]]]}];
    integratorInitializations =
      ": quadrature_provider(config)" <>
        If[Length[integrators] > 0,
          ", " <> StringRiffle[integrators[[All, 1]], "(quadrature_provider, config), "] <> "(quadrature_provider, config)"
          ,
          ""
        ];
    flowCpp = FunKit`MakeCppBlock["Includes" -> {"./flows.hh"}, "Body" -> {FunKit`MakeCppFunction["Name" -> varName, "Class" -> varName, "Suffix" -> integratorInitializations, "Body" -> "", "Parameters" -> {<|"Type" -> "DiFfRG::ConfigTree", "Reference" -> True, "Const" -> True, "Name" -> "config"|>}, "Return" -> ""], FunKit`MakeCppFunction["Name" -> "set_k", "Class" -> varName, "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "k", "Reference" -> False|>}, "Body" -> StringJoin[Map["DiFfRG::all_set_k(" <> #[[1]] <> ", k);"&, integrators]], "Return" -> "void"], FunKit`MakeCppFunction["Name" -> "set_T", "Class" -> varName, "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "T", "Reference" -> False|>}, "Body" -> StringJoin[Map["DiFfRG::all_set_T(" <> #[[1]] <> ", T);"&, integrators]], "Return" -> "void"], FunKit`MakeCppFunction["Name" -> "set_typical_E", "Class" -> varName, "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "E", "Reference" -> False|>}, "Body" -> StringJoin[Map["DiFfRG::all_set_typical_E(" <> #[[1]] <> ", E);"&, integrators]], "Return" -> "void"], FunKit`MakeCppFunction["Name" -> "set_x_extent", "Class" -> varName, "Parameters" -> {<|"Type" -> "double", "Const" -> True, "Name" -> "x_extent", "Reference" -> False|>}, "Body" -> StringJoin[Map["DiFfRG::all_set_x_extent(" <> #[[1]] <> ", x_extent);"&, integrators]], "Return" -> "void"]}];
    ExportCode[FileNameJoin[{flowDir, "flows.hh"}], flowHeader];
    ExportCode[FileNameJoin[{flowDir, "flows.cc"}], flowCpp];
  ];

(* ::Chapter:: *)

(*Finishing package*)

Protect["DiFfRG`CodeTools`*"];

End[];

EndPackage[];
