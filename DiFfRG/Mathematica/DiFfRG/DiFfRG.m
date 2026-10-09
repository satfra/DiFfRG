(* ::Package:: *)

(* ::Title:: *)

(*DiFfRG Master package*)

(* ::Chapter:: *)

(*Package Setup*)

(* ::Section:: *)

(*Hide styling if we are in a CLI context.*)

If[$FrontEnd === Null || $Notebooks === False,
	Unprotect[Style];
	Unprotect[StyleBox];
	Unprotect[Print];
	Style[expr_, opts___] := expr;
	StyleBox[expr_, opts___] := expr;
	Print[expr__] := Write[$Output, StringJoin[ToString[#, OutputForm]& /@ {expr}]];
	Protect[Print];
	Protect[StyleBox];
	Protect[Style];
];

(*Welcome  Message*)

If[$FrontEnd=!=Null,
Print["Mathematica package \!\(\*
StyleBox[\"DiFfRG\",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\" \",\nFontWeight->\"Bold\"]\)loaded
\!\(\*
StyleBox[\"Authors\",\nFontWeight->\"Bold\"]\): Franz Richard Sattler
\!\(\*
StyleBox[\"Version\",\nFontWeight->\"Bold\"]\): 2.0
\!\(\*
StyleBox[\"Year\",\nFontWeight->\"Bold\"]\): 2025
"],
Print["Mathematica package DiFfRG loaded"];
Print["Authors: Franz Richard Sattler"];
Print["Version: 2.0"];
Print["Year:    2026"];
];

(* ::Section:: *)

(*Load FunKit*)

DiFfRG::FunKitRequired = "\!\(\*
StyleBox[\"DiFfRG\",\nFontWeight->\"Bold\"]\) requires \!\(\*
StyleBox[\"FunKit\",\nFontWeight->\"Bold\"]\) to run.";

If[Length @ PacletFind["FunKit"] === 0,
	If[ChoiceDialog["FunKit does not seem to be installed. Do you want to install it?", WindowTitle -> "Install FunKit", WindowSize -> {Medium, All}],
		If[$VersionNumber >= 11.2,
			Import["https://raw.githubusercontent.com/satfra/FunKit/main/FunKitInstaller.m"],
			Module[{zipURL, archive, installDir, data},
				zipURL = "https://github.com/satfra/FunKit/archive/refs/heads/main.zip";
				archive = FileNameJoin[{$TemporaryDirectory, "FunKit.zip"}];
				installDir = FileNameJoin[{$UserBaseDirectory, "Applications"}];
				Print["Downloading FunKit ..."];
				data = Quiet[Import[zipURL, "Byte"]];
				If[!ListQ[data] || Length[data] === 0,
					Print["Download of FunKit failed."]; Abort[]
				];
				Close[BinaryWrite[OpenWrite[archive, BinaryFormat -> True], data]];
				ExtractArchive[archive, installDir];
				RenameDirectory[FileNameJoin[{installDir, "FunKit-main"}], FileNameJoin[{installDir, "FunKit"}]];
				Quiet[DeleteFile[archive]];
				Print["FunKit installed successfully."];
			]
		]
		,
		Message[DiFfRG::FunKitRequired];
		Abort[];
	]
	,
	Get["FunKit`"]
]

Needs["DiFfRG`CodeTools`"];

ShowFlowDirectory[] // Print;

(* ::Section:: *)

(*Init*)

(* ::Input::Initialization:: *)

(*BeginPackage["DiFfRG`"];*)

Unprotect["DiFfRG`*"];

Unprotect["DiFfRG`Private`*"];

ClearAll["DiFfRG`*"];

ClearAll["DiFfRG`Private`*"];

(* ::Section:: *)

(*Exports*)

GetDirectory::usage = "GetDirectory[]
Returns the directory in which either the package file or notebook is located.";

AutoExport::usage = "AutoExport[]
Turns on automatic export of the current notebook to a .m file.";

SnapshotInfo::usage = "SnapshotInfo[file_String]
Reads a DiFfRG flow snapshot, i.e. a file <run name>_snapshot_<nnn>.h5 written by a run with /timestepping/snapshots set, and returns an Association with the RG time \"t\", the scale \"k\" (Indeterminate if the run had no /physical/Lambda), \"Lambda\", \"dim\", the flowing \"Variables\" and the \"Config\" of the run that wrote it. A later run continues from the snapshot with --restart file.";

ListSnapshots::usage = "ListSnapshots[run_String]
Returns SnapshotInfo for every snapshot of a run, sorted by RG time. run is the path of the run without extension, e.g. \"output/seed\" for the files output/seed_snapshot_*.h5.";

AutoSaveRestore::usage = "AutoSaveRestore[fileName_String,expr_]
Evaluates expr if the file fileName does not yet exist and saves it to fileName as a .m file. If fileName exists, it does not evaluate expr, but simply loads the contents.";

MatsubaraSum::usage = "MatsubaraSum[expr_,p0_Symbol,T_]
Sums the expression expr over p0, with p0 = 2\[Pi]T*n, where n\[Element]\[CapitalZeta].";

FermionMatsubaraSum::usage = "FermionMatsubaraSum[expr_,p0_Symbol,T_]
Sums the expression expr over p0, with p0 = (2*n+1)\[Pi]T, where n\[Element]\[CapitalZeta].";

ResolveChargeConjugation::usage = "ResolveChargeConjugation[expr_]
Resolves all occurences of ChargeConj[i,j] by using the effect of these in Weyl representation.";

UseSU3GellMannTrace::usage = "SU3GellMannTrace[value_?BooleanQ,groupName_String:\"color\"]
Traces the generators of the fundamental representation in the given group as SU(3) Gell-Mann matrices explicitly.";

PreTrace::usage = "PreTrace[expr_]
Resolve charge conjugation and perform possibly the explicit color trace."

FormMomentumExpansion::usage = "FormMomentumExpansion[momenta___]"

FiniteTFormMomentumExpansion[momenta__]

MakeP0FormRule::usage = "MakeP0FormRule[q_,{momenta__},{projections__}]"

GetFTSynonym::usage = "GetFTSynonym[symbol_]"

MakeSPFormRule::usage = "MakeSPFormRule[q_,p_,momenta__]"

MakeSPFormRule::usage = "MakeSPFormRule[q_,p_,momenta__]"

MakeSPFiniteTFormRule::usage = "MakeSPFiniteTFormRule[q_,p_,momenta__]"

ExtendedFormTrace::usage = "ExtendedFormTrace[expr_,disentangle:_?BooleanQ:True,preRepRules_List:{},postRepRules_List:{}]"

SeparateScalarProductsFiniteT::usage = "SeparateScalarProductsFiniteT[expr_]"

ExpandScalarProductsFiniteT::usage = "ExpandScalarProductsFiniteT[expr_]"

ExpandScalarProducts::usage = "ExpandScalarProducts[expr_]"

SimplifyAllMomenta::usage = "SimplifyAllMomenta[q_,expr_]"

ProjectToSymmetricPoint::usage = "ProjectToSymmetricPoint[expr_,q_Symbol,p_Symbol,momenta___Symbol]"

ProjectToSymmetricPointFiniteT::usage = "ProjectToSymmetricPointFiniteT[expr_,q_Symbol,p_Symbol,momenta___Symbol]"

QuickSimplify::usage = "QuickSimplify[expr_]"

SetStandardQuickSimplify::usage = "SetStandardQuickSimplify[sim_]"

SetStandardSimplify::usage = "SetStandardSimplify[sim_]"

SetDisentangle::usage = "SetDisentangle[ex_]"

SumDiagrams::usage = "SumDiagrams[nKernels_Integer,postfix_String,number_Integer:0,operation_:$StandardQuickSimplify,saveName_String:\"sum\"]";

TraceDiagrams::usage = "TraceDiagrams[simpFunc_,nKernels_Integer,postfix_String,ex_,preRepRules_List:{},postRepRules_List:{}]"

DiracTranspose::usage = "DiracTranspose[expression] switches two open dirac indices";

(* ::Section:: *)

(*Begin  Private*)

(*Begin["`Private`"]*)

(* ::Chapter:: *)

(*Utility*)

(* ::Text:: *)

(*Some logic for getting Directories etc also from scripts*)

Unprotect[GetDirectory];

GetDirectory[] :=
	If[$Notebooks,
			NotebookDirectory[]
			,
			Directory[]
		] <> "/";

(* ::Input::Initialization:: *)

AutoExport[Switch : _?BooleanQ : True] :=
	If[Switch,
		SetOptions[EvaluationNotebook[], {AutoGeneratedPackage -> Automatic, InitializationCellEvaluation -> False, InitializationCellWarning -> False, StyleDefinitions -> Notebook[{Cell[StyleData[StyleDefinitions -> "Default.nb"]], Cell[StyleData["Input"], InitializationCell -> True]}, Visible -> False, StyleDefinitions -> "PrivateStylesheetFormatting.nb"]}]
		,
		SetOptions[EvaluationNotebook[], {AutoGeneratedPackage -> False}]
	];

AutoSaveRestore[fileName_String, expr_] :=
	Module[{ret},
		If[FileExistsQ[fileName <> ".m"],
			ret = Import[fileName <> ".m"];
			Print["Imported existing file \"" <> fileName <> ".m" <> "\"..."];
			,
			ret = ReleaseHold[expr];
			Export[fileName <> ".m", ret];
			Print["Saved to file \"" <> fileName <> ".m" <> "\"..."];
		];
		ret
	];

SetAttributes[AutoSaveRestore, HoldRest];

(* ::Chapter:: *)

(*Matsubara sums*)

(* ::Text:: *)

(*Explicit summation of Matsubara sums is possible (but perhaps not recommended always.)*)

ClearAll[MatsubaraSumHelper, FermionMatsubaraSum, MatsubaraSum]

Options[FermionMatsubaraSum] = {Debug -> False};

Options[MatsubaraSum] = {Debug -> False};

MatsubaraSumHelper[expr_, p0_Symbol, T_, fun_, debug_] :=
	Module[{denom, poles, residues, n, r, SumB},
		(* term by term with the SAME weight: recursing into MatsubaraSum would sum the terms of a
		   fermionic sum on the bosonic tower *)
		If[Head[expr] == Plus,
			Return[Total[Map[MatsubaraSumHelper[#, p0, T, fun, debug]&, List @@ expr]]]
		];
		denom = Denominator[expr] // Collect[#, p0]&;
		poles = p0 /. Solve[denom == 0, p0] // FullSimplify;
		poles = Assuming[T > 0 && _Symbol \[Element] Reals, poles // FullSimplify];
		(* Solve lists a root once per multiplicity; Residue already accounts for the order of a pole,
		   so a repeated root would be counted twice *)
		poles = DeleteDuplicates[poles, Simplify[#1 - #2] === 0&];
		If[debug,
			Print["Denominator found:"];
			Print[denom];
			Print["Poles found at:"];
			Print /@ poles;
		];
		residues = {};
		For[n = 1, n <= Length[poles], n++,
			r = Simplify[Residue[expr * fun[I * p0 / (2 T)], {p0, poles[[n]]}]];
			AppendTo[residues, r];
		];
		SumB = 1 / (2 I) residues;
		Total[SumB]
	];

MatsubaraSum[expr_, p0_Symbol, T_, OptionsPattern[]] :=
	MatsubaraSumHelper[expr, p0, T, Coth, OptionValue[Debug]]

FermionMatsubaraSum[expr_, p0_Symbol, T_, OptionsPattern[]] :=
	MatsubaraSumHelper[expr, p0, T, Tanh, OptionValue[Debug]]

(* ::Chapter:: *)

(*Tracing  Algorithms*)

InsertOutputNaming[expr_] :=
	Module[{groups, outputRulesLorentzTensors, outputRulesGroupTensors, outputRules, privateOutputRulesLorentzTensors, privateOutputRulesGroupTensors, privateOutputRules, otherRules},
		outputRulesLorentzTensors = Normal[FormTracer`Private`lorentzTensorReplacementRulesOutput] /. {(a_[c___] :> b_) :> (Symbol["TB" ~~ StringSplit[ToString[a], "FTx"][[-1]]][c] :> b)};
		outputRulesGroupTensors = Normal[FormTracer`Private`groupTensorReplacementRulesOutput] /. {(a_[c___] :> b_) :> (Symbol["TB" ~~ StringSplit[ToString[a], "FTx"][[-1]]][c] :> b)};
		outputRules = Join[outputRulesLorentzTensors, outputRulesGroupTensors];
		Return[Evaluate[expr //. outputRules]];
	];

InsertInputNaming[expr_] :=
	Module[{inputRulesLorentzTensors, inputRulesGroupTensors},
		inputRulesLorentzTensors = Select[Normal[FormTracer`Private`lorentzTensorReplacementRulesInput], MatchQ[_[__] :> _[__]]] /. {(a_[d__] :> b_[c__]) :> (a[d] -> Symbol["TB" ~~ StringSplit[ToString[b], "FTx"][[-1]]][c])};
		inputRulesGroupTensors = Select[Normal[FormTracer`Private`groupTensorReplacementRulesInput], MatchQ[_[__] :> _[__]]] /. {(a_[d__] :> b_[c__]) :> (a[d] -> Symbol["TB" ~~ StringSplit[ToString[b], "FTx"][[-1]]][c])};
		Return[expr //. inputRulesLorentzTensors //. inputRulesGroupTensors /. a_Symbol :> Symbol["Global`" ~~ SymbolName[a]]];
	];

(* ::Subsection:: *)

(*Charge  Conjugators*)

(* ::Text:: *)

(*In order to be able to deal with charge conjugation operators, define some rules for that, too*)

ImplodeTerms[expr_] :=
	expr;

ImplodeTerms[expr_List] :=
	Module[{ret},
		If[Head[expr[[1]]] === List,
			Plus @@ Map[ImplodeTerms, expr]
			,
			Times @@ Map[ImplodeTerms, expr]
		]
	];

CollapseTerms[expr_] :=
	expr;

CollapseTerms[expr_List] :=
	Total[Map[#[[1]] * CollapseTerms[#[[2]]]&, expr]];

FixChargeConj[expr_] :=
	Module[{ret, restoreIdx, rule, chargeRules},
		Block[{Print},
			DisentangleLorentzStructures[False]
		];
		ret = expr;
		restoreIdx = {};
		ret =
			ret //.
				ChargeConj[a_, b_] :>
					Module[{},
						restoreIdx = Append[restoreIdx, {a, b}];
						InsertOutputNaming[TBdeltaDirac[a, b]]
					];
		rule = Map[InsertOutputNaming[TBdeltaDirac[#[[1]], #[[2]]]] :> ChargeConj[#[[1]], #[[2]]]&, restoreIdx];
		ret =
			FormTracer`ExpandTerms[ret] //. rule //
			Expand //
			InsertOutputNaming;
		chargeRules = {TBgamma[mu_, a_, c_] ChargeConj[a_, b_] :> -ChargeConj[c, a] TBgamma[mu, a, b], TBgamma[mu_, a_, b_] ChargeConj[b_, c_] :> -ChargeConj[a, b] TBgamma[mu, c, b], TBgamma5[a_, c_] ChargeConj[a_, b_] :> ChargeConj[c, a] TBgamma5[a, b], TBgamma5[a_, b_] ChargeConj[b_, c_] :> ChargeConj[a, b] TBgamma5[c, b], TBdeltaDirac[a_, b_] ChargeConj[b_, c_] :> ChargeConj[a, c], TBdeltaDirac[a_, b_] ChargeConj[c_, b_] :> ChargeConj[c, a], TBdeltaDirac[b_, a_] ChargeConj[b_, c_] :> ChargeConj[a, c], ChargeConj[a_, b_] ChargeConj[a_, c_] :> deltaDirac[b, c], ChargeConj[a_, b_] ChargeConj[c_, b_] :> deltaDirac[a, c], ChargeConj[a_, b_] ChargeConj[b_, c_] :> -deltaDirac[a, c]} // InsertOutputNaming;
		ret = ret //. chargeRules;
		(*This is chiral Weyl representation-specific.*)
		ret =
			ret //.
				{
					ChargeConj[a_, b_] :>
						Module[{dint},
							TBgamma[2, a, dint] TBgamma[0, dint, b]
						]
				};
		Return[ImplodeTerms[ret] // InsertOutputNaming];
	];

ResolveChargeConjugation[expr_] :=
	Module[{},
		If[MemberQ[expr, ChargeConj[__], Infinity],
			expr // FixChargeConj
			,
			expr
		]
	];

integrateDeltas[expr_] :=
	Module[{list1, list2, conv = InsertOutputNaming},
		list1 =
			{
				FormTracer`Private`FTxdeltaLorentz[a_, b_] FormTracer`Private`FTxdeltaLorentz[a_, c_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaLorentz[b, c]
				,
				conv @ TBdeltaLorentz[a_, b_] conv @ TBdeltaLorentz[a_, c_] /; Not[NumberQ[a]] :> conv @ TBdeltaLorentz[b, c]
				,
				FormTracer`Private`FTxdeltaLorentz[a_, b_] FormTracer`Private`FTxdeltaLorentz[c_, a_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaLorentz[b, c]
				,
				conv @ TBdeltaLorentz[a_, b_] conv @ TBdeltaLorentz[c_, a_] /; Not[NumberQ[a]] :> conv @ TBdeltaLorentz[b, c]
				,
				FormTracer`Private`FTxdeltaLorentz[b_, a_] FormTracer`Private`FTxdeltaLorentz[a_, c_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaLorentz[b, c]
				,
				conv @ TBdeltaLorentz[b_, a_] conv @ TBdeltaLorentz[a_, c_] /; Not[NumberQ[a]] :> conv @ TBdeltaLorentz[b, c]
				,
				FormTracer`Private`FTxdeltaLorentz[b_, a_] FormTracer`Private`FTxdeltaLorentz[c_, a_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaLorentz[b, c]
				,
				conv @ TBdeltaLorentz[b_, a_] conv @ TBdeltaLorentz[c_, a_] /; Not[NumberQ[a]] :> conv @ TBdeltaLorentz[b, c]
				,
				FormTracer`Private`FTxdeltaLorentz[a_, a_] :>
					If[NumberQ[a],
						1
						,
						4
					]
				,
				conv @ TBdeltaLorentz[a_, a_] :>
					If[NumberQ[a],
						1
						,
						4
					]
				,
				conv @ TBvec[p_, a_] conv @ TBdeltaLorentz[a_, b_] conv @ TBvec[q_, b_] :> conv @ TBsp[p, q]
				,
				conv @ TBvec[p_, a_] conv @ TBdeltaLorentz[b_, a_] conv @ TBvec[q_, b_] :> conv @ TBsp[p, q]
				,
				conv @ TBvecs[p_, a_] conv @ TBdeltaLorentz[a_, b_] conv @ TBvecs[q_, b_] :> conv @ TBsps[p, q]
				,
				conv @ TBvecs[p_, a_] conv @ TBdeltaLorentz[b_, a_] conv @ TBvecs[q_, b_] :> conv @ TBsps[p, q]
			};
		list2 =
			{
				FormTracer`Private`FTxdeltaDirac[a_, b_] FormTracer`Private`FTxdeltaDirac[a_, c_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaDirac[b, c]
				,
				conv @ TBdeltaDirac[a_, b_] conv @ TBdeltaDirac[a_, c_] /; Not[NumberQ[a]] :> conv @ TBdeltaDirac[b, c]
				,
				FormTracer`Private`FTxdeltaDirac[a_, b_] FormTracer`Private`FTxdeltaDirac[c_, a_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaDirac[b, c]
				,
				conv @ TBdeltaDirac[a_, b_] conv @ TBdeltaDirac[c_, a_] /; Not[NumberQ[a]] :> conv @ TBdeltaDirac[c, b]
				,
				FormTracer`Private`FTxdeltaDirac[b_, a_] FormTracer`Private`FTxdeltaDirac[a_, c_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaDirac[b, c]
				,
				conv @ TBdeltaDirac[b_, a_] conv @ TBdeltaDirac[a_, c_] /; Not[NumberQ[a]] :> conv @ TBdeltaDirac[b, c]
				,
				FormTracer`Private`FTxdeltaDirac[b_, a_] FormTracer`Private`FTxdeltaDirac[c_, a_] /; Not[NumberQ[a]] :> FormTracer`Private`FTxdeltaDirac[b, c]
				,
				conv @ TBdeltaDirac[b_, a_] conv @ TBdeltaDirac[c_, a_] /; Not[NumberQ[a]] :> conv @ TBdeltaDirac[b, c]
				,
				FormTracer`Private`FTxdeltaDirac[a_, a_] :>
					If[NumberQ[a],
						1
						,
						4
					]
				,
				conv @ TBdeltaDirac[a_, a_] :>
					If[NumberQ[a],
						1
						,
						4
					]
				,
				conv @ TBdeltaDirac[b_, a_] conv @ TBgamma[mu_, c_, a_] /; Not[NumberQ[a]] :> conv @ TBgamma[mu, c, b]
				,
				conv @ TBdeltaDirac[a_, b_] conv @ TBgamma[mu_, c_, a_] /; Not[NumberQ[a]] :> conv @ TBgamma[mu, c, b]
				,
				conv @ TBdeltaDirac[b_, a_] conv @ TBgamma[mu_, a_, c_] /; Not[NumberQ[a]] :> conv @ TBgamma[mu, b, c]
				,
				conv @ TBdeltaDirac[a_, b_] conv @ TBgamma[mu_, a_, c_] /; Not[NumberQ[a]] :> conv @ TBgamma[mu, b, c]
				,
				conv @ TBdeltaDirac[b_, a_] conv @ TBgamma5[c_, a_] /; Not[NumberQ[a]] :> conv @ TBgamma5[c, b]
				,
				conv @ TBdeltaDirac[a_, b_] conv @ TBgamma5[c_, a_] /; Not[NumberQ[a]] :> conv @ TBgamma5[c, b]
				,
				conv @ TBdeltaDirac[b_, a_] conv @ TBgamma5[a_, c_] /; Not[NumberQ[a]] :> conv @ TBgamma5[b, c]
				,
				conv @ TBdeltaDirac[a_, b_] conv @ TBgamma5[a_, c_] /; Not[NumberQ[a]] :> conv @ TBgamma5[b, c]
			};
		Return[expr //. list1 //. list2]
	];

(* ::Subsubsection:: *)

(*Transposer*)

DiracTranspose::tooManyExternalIndices = "ForTransposition, only two open indices are supported. `1` as poen indices were supplied.";

DiracTranspose[expr_] :=
	Module[{exprNew, openIndices, dummpyIndex},
		openIndices = GetOpenDiracIndices[expr];
		If[Length[openIndices] != 2,
			Message[DiracTranspose::tooManyExternalIndices, openIndices];
			Abort[]
		];
		exprNew = expr //. {openIndices[[1]] -> dummpyIndex};
		exprNew = exprNew //. {openIndices[[2]] -> openIndices[[1]]};
		exprNew = exprNew //. {dummpyIndex -> openIndices[[2]]};
		exprNew
	];

DiracTranspose[expr_List] :=
	DiracTranspose /@ expr;

(* ::Subsection::Closed:: *)

(*PreTrace helper*)

(* ::Input::Initialization:: *)

PreTrace[expr_] :=
	Module[{exprExp},
		Block[{Print},
			DisentangleLorentzStructures[False]
		];
		exprExp =
			ExpandTerms[
				expr //
				integrateDeltas //
				InsertSU3GellManns
			];
		Do[exprExp[[iDia, 2, iFlavor, 1]] = SU3GellMannTrace[ConvertInput[exprExp[[iDia, 2, iFlavor, 1]]]], {iDia, Length[exprExp]}, {iFlavor, Length[exprExp[[iDia, 2]]]}];
		Return[
			ImplodeTerms[exprExp] //
			integrateDeltas //
			QuickSimplify
		];
	];

(* ::Chapter:: *)

(*Simplification algorithms*)

(* ::Subsection:: *)

(*Momentum  simplifications*)

(* ::Text:: *)

(*Some helpers for dealing with the momentum notation.*)

SeparateScalarProductsFiniteT[expr_] :=
	Module[{},
		UseLorentzLinearity[expr] //. InsertOutputNaming @ {TBsp[q_, p_] :> TBsps[q, p] + TBvec[q, 0] TBvec[p, 0], TBvec[p_, mu_ /; mu =!= 0] :> TBvecs[p, mu] + TBdeltaLorentz[mu, 0] TBvec[p, 0]}
	]

ExpandScalarProductsFiniteT[expr_] :=
	Module[{q, p},
		UseLorentzLinearity[expr] //. InsertOutputNaming @ {TBsp[q_, p_] -> TBsps[q, p] + TBvec[q, 0] TBvec[p, 0]} //. InsertOutputNaming @ {TBsps[q_, p_] -> q p cos[q, p], cos[p_, p_] -> 1}
	]

ExpandScalarProducts[expr_] :=
	Module[{q, p},
		UseLorentzLinearity[expr] //. InsertOutputNaming @ {TBsp[q_, p_] -> q p cos[q, p], cos[p_, p_] -> 1}
	]

Unprotect[cos];

Derivative[0, 1][cos][p_, q_] = 0;

Derivative[1, 0][cos][p_, q_] = 0;

Protect[cos];

SimplifyAllMomenta[q_, expr_, qffac_ : \[Pi] T] :=
	UseLorentzLinearity[expr] //.
			InsertOutputNaming @
				{
					cos[p_, Symbol[ToString[q] <> "f"]] :> Symbol["cos" <> ToString[p] <> ToString[q]]
					,
					TBvec[q, 0] :> Symbol[ToString[q] <> "0"]
					,
					TBvec[Symbol[ToString[q] <> "f"], 0] :>
						(
							ClearAll[T];
							Symbol[ToString[q] <> "0"] + qffac
						)
				} //. InsertOutputNaming @ {cos[p_, r_] :> Symbol["cos" <> ToString[p] <> ToString[r]], TBvec[p_, 0] :> Symbol[ToString[p] <> "0"], Symbol[ToString[q] <> "f"] -> q}

(* ::Text:: *)

(*We should be also able to access symmetric points.*)

ProjectToSymmetricPoint[expr_, q_Symbol, p_Symbol, momenta___Symbol] :=
	Module[{momentaList, nMomenta, rules, qf, qfString, conv = InsertOutputNaming},
		momentaList = {momenta};
		nMomenta = Length[momentaList];
		qfString = ToString[q];
		qfString =
			If[StringMatchQ[qfString, ___ ~~ DigitCharacter..],
				StringReplace[qfString, RegularExpression["(\\d+)$"] -> "f" <> "$1"]
				,
				qfString <> "f"
			];
		qf = Symbol[qfString];
		rules = Map[conv @ TBsp[#[[1]], #[[2]]] -> -(1 / (nMomenta - 1)) conv @ TBsps[p, p]&, Subsets[momentaList, {2}]] \[Union] Map[conv @ TBsp[#, #] -> conv @ TBsps[p, p]&, momentaList] \[Union] Map[conv @ TBsp[#, q] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p q&, momentaList] \[Union] Map[conv @ TBsp[#, qf] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p qf&, momentaList] \[Union] Map[conv @ TBsp[q, #] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p q&, momentaList] \[Union] Map[conv @ TBsp[qf, #] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p qf&, momentaList]
		(*\[Union]{momentaList[[nMomenta]]->-Total[momentaList[[1;;nMomenta-1]]]}*);
		(UseLorentzLinearity[expr] //. rules // UseLorentzLinearity) //. rules
	]

ProjectToSymmetricPointFiniteT[expr_, q_Symbol, p_Symbol, momenta___Symbol] :=
	Module[{momentaList, nMomenta, rules, qf, qfString, conv = InsertOutputNaming},
		momentaList = {momenta};
		nMomenta = Length[momentaList];
		qfString = ToString[q];
		qfString =
			If[StringMatchQ[qfString, ___ ~~ DigitCharacter..],
				StringReplace[qfString, RegularExpression["(\\d+)$"] -> "f" <> "$1"]
				,
				qfString <> "f"
			];
		qf = Symbol[qfString];
		rules = Map[conv @ TBsps[#[[1]], #[[2]]] -> -(1 / (nMomenta - 1)) conv @ TBsps[p, p]&, Subsets[momentaList, {2}]] \[Union] Map[conv @ TBsps[#, #] -> conv @ TBsps[p, p]&, momentaList] \[Union] Map[conv @ TBsps[#, q] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p q&, momentaList] \[Union] Map[conv @ TBsps[#, qf] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p qf&, momentaList] \[Union] Map[conv @ TBsps[q, #] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p q&, momentaList] \[Union] Map[conv @ TBsps[qf, #] -> Symbol["cos" ~~ ToString[#] ~~ "q"] p qf&, momentaList]
		(*\[Union]{momentaList[[nMomenta]]->-Total[momentaList[[1;;nMomenta-1]]]}*);
		(SeparateScalarProductsFiniteT[UseLorentzLinearity[expr]] //. rules // UseLorentzLinearity) //. rules
	]

(* ::Subsection::Closed:: *)

(*Simplification helpers*)

(* ::Input::Initialization:: *)

QuickSimplify[expr_] :=
	Simplify[expr, TimeConstraint -> 0.1] // Quiet;

$StandardSimplify = Simplify;

SetStandardSimplify[sim_] :=
	Module[{},
		$StandardSimplify = sim;
	]

$StandardQuickSimplify = QuickSimplify;

SetStandardQuickSimplify[sim_] :=
	Module[{},
		$StandardQuickSimplify = sim;
	]

AdjustToNKernels[nKernels_Integer] :=
	Module[{},
		If[nKernels != 0,
			CloseKernels[];
			LaunchKernels[nKernels];
			DistributeDefinitions["FormTracer`"];
			DistributeDefinitions[ExtendedFormTrace, InsertChargeConjRules]
		];
	]

RestoreKernels[] :=
	Module[{},
		CloseKernels[];
		LaunchKernels[];
		DistributeDefinitions["FormTracer`"];
		DistributeDefinitions[ExtendedFormTrace, InsertChargeConjRules];
	]

DiFfRGMap[nKernels_Integer] :=
	Module[{dynMap},
		dynMap = If[$VersionNumber >= 12.0,
			ResourceFunction["DynamicMap"],
			Map
		];
		If[nKernels <= 1,
			Return[dynMap]
			,
			If[$VersionNumber >= 12.0,
				Return[ResourceFunction["DynamicMap"][#1, #2, Parallel -> nKernels]&],
				Return[ParallelMap[#1, #2]&]
			]
		];
	];

(* ::Chapter:: *)

(*Flow snapshots*)

SnapshotInfo[file_String] :=
	Module[{attributes, datasets, config, variables, k},
		If[Not[FileExistsQ[file]], Message[SnapshotInfo::nofile, file]; Return[$Failed]];
		attributes = Import[file, {"HDF5", "Attributes", "/"}];
		datasets = Import[file, {"HDF5", "Datasets"}];
		(* The configuration is written as 2E-2 etc., which RawJSON reads as the exact 1/50 *)
		config = ImportString[First[Flatten[{Import[file, {"HDF5", "Datasets", "/config_json"}]}]], "RawJSON"] /. r_Rational :> With[{x = N[r]}, x /; True];
		variables = If[MemberQ[datasets, "/state/variables"], Import[file, {"HDF5", "Datasets", "/state/variables"}], {}];
		(* k is stored as NaN when the run had no Lambda, which Import returns as $Failed *)
		k = If[NumericQ[attributes["k"]], attributes["k"], Indeterminate];
		<|"File" -> file, "t" -> attributes["t"], "k" -> k, "Lambda" -> attributes["Lambda"], "dim" -> attributes["dim"], "Variables" -> variables, "Config" -> config|>
	];

SnapshotInfo::nofile = "The snapshot file `1` does not exist.";

ListSnapshots[run_String] :=
	Module[{directory, files},
		directory = DirectoryName[run];
		If[directory === "", directory = Directory[]];
		files = FileNames[FileNameTake[run] <> "_snapshot_*.h5", directory];
		SortBy[SnapshotInfo /@ files, #["t"]&]
	];

(* ::Chapter:: *)

(*Finishing package*)

(*End[]*)

Protect["DiFfRG`*"];

(*EndPackage[];*)
