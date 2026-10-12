(* ::Package:: *)

(* ::Title:: *)
(*Quark-meson model in LPA'*)


(* ::Text:: *)
(*Derives the flows of a quark-meson model in LPA': the effective potential V(rho), the anomalous dimensions etaQ and etaPhi of quarks and mesons, and the Yukawa coupling hPhi, all at finite temperature and quark chemical potential.*)
(*Run headless with:  wolfram -script QM.wl*)


(* ::Section:: *)
(*SetUp*)


(* ::Input::Initialization:: *)
Get["DiFfRG`"]
SetDirectory[If[$InputFileName==="",Directory[],DirectoryName[$InputFileName]]];


(* ::Subsection:: *)
(*Field and Tensor bases*)


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[Sigma]\[Sigma]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[Sigma], \[Sigma]},
 "VertexStructure" -> Tensor[1, 2],
 "InnerProduct" -> Tensor1[1, 2] Tensor2[1, 2],
 "Indices" -> {{p1}, {p2}},
 "Tensors" -> {{1}}]
Print["\[Sigma]\[Sigma] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[CapitalPi]\[CapitalPi]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[CapitalPi], \[CapitalPi]},
 "VertexStructure" -> Tensor[1, 2],
 "InnerProduct" -> Tensor1[1, 2] Tensor2[1, 2],
 "Indices" -> {{p1, f1}, {p2, f2}},
 "Tensors" -> {{deltaAdjFlav[f1, f2]}}]
Print["\[CapitalPi]\[CapitalPi] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[CapitalPi]\[CapitalPi]\[Sigma]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[CapitalPi], \[CapitalPi], \[Sigma]},
 "VertexStructure" -> Tensor[1, 2, 3],
 "InnerProduct" -> Tensor1[1, 2, 3] Tensor2[1, 2, 3],
 "Indices" -> {{p1, f1}, {p2, f2}, {p3}},
 "Tensors" -> {{deltaAdjFlav[f1, f2]}}]
Print["\[CapitalPi]\[CapitalPi]\[Sigma] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[CapitalPi]\[CapitalPi]\[Sigma]\[Sigma]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[CapitalPi], \[CapitalPi], \[Sigma], \[Sigma]},
 "VertexStructure" -> Tensor[1, 2, 3, 4],
 "InnerProduct" -> Tensor1[1, 2, 3, 4] Tensor2[1, 2, 3, 4],
 "Indices" -> {{p1, f1}, {p2, f2}, {p3}, {p4}},
 "Tensors" -> {{deltaAdjFlav[f1, f2]}}]
Print["\[CapitalPi]\[CapitalPi]\[Sigma]\[Sigma] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[Sigma]\[Sigma]\[Sigma]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[Sigma], \[Sigma], \[Sigma]},
 "VertexStructure" -> Tensor[1, 2, 3],
 "InnerProduct" -> Tensor1[1, 2, 3] Tensor2[1, 2, 3],
 "Indices" -> {{p1}, {p2}, {p3}},
 "Tensors" -> {{1}}]
Print["\[Sigma]\[Sigma]\[Sigma] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[Sigma]\[Sigma]\[Sigma]\[Sigma]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[Sigma], \[Sigma], \[Sigma], \[Sigma]},
 "VertexStructure" -> Tensor[1, 2, 3, 4],
 "InnerProduct" -> Tensor1[1, 2, 3, 4] Tensor2[1, 2, 3, 4],
 "Indices" -> {{p1}, {p2}, {p3}, {p4}},
 "Tensors" -> {{1}}]
Print["\[Sigma]\[Sigma]\[Sigma]\[Sigma] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "\[CapitalPi]\[CapitalPi]\[CapitalPi]\[CapitalPi]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}},
 "Vertex" -> {\[CapitalPi], \[CapitalPi], \[CapitalPi], \[CapitalPi]},
 "VertexStructure" -> Tensor[1, 2, 3, 4],
 "InnerProduct" -> Tensor1[1, 2, 3, 4] Tensor2[1, 2, 3, 4],
 "Indices" -> {{p1, f1}, {p2, f2}, {p3, f3}, {p4, f4}},
 "Tensors" -> {{deltaAdjFlav[f1, f2] deltaAdjFlav[f3, f4]}}]
Print["\[CapitalPi]\[CapitalPi]\[CapitalPi]\[CapitalPi] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "qbq\[CapitalPi]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}, {color, SUNfund, Nc}},
 "Vertex" -> {qb, q, \[CapitalPi]},
 "VertexStructure" -> Tensor[1, 2, 3],
 "InnerProduct" -> Tensor1[1, 2, 3] Tensor2[2, 1, 3],
 "CanonicalProduct" -> Tensor1[1, 2, 3] Tensor2[2, 1, 3],
 "Indices" -> {{p1, d1, A1, F1}, {p2, d2, A2, F2}, {p3, f3}},
 "Tensors" -> {{I gamma5[d1, d2] TFlav[f3, F1, F2] deltaFundCol[A1, 
      A2]}}]
Print["qbq\[CapitalPi] constructed!"]


(* ::Input::Initialization:: *)
TBConstructBasis[
 "Name" -> "qbq\[Sigma]",
 "RequiredGroups" -> {{flavor, SUNfund, Nf}, {color, SUNfund, Nc}},
 "Vertex" -> {qb, q, \[Sigma]},
 "VertexStructure" -> Tensor[1, 2, 3],
 "InnerProduct" -> Tensor1[1, 2, 3] Tensor2[2, 1, 3],
 "CanonicalProduct" -> Tensor1[1, 2, 3] Tensor2[2, 1, 3],
 "Indices" -> {{p1, d1, A1, F1}, {p2, d2, A2, F2}, {p3}},
 "Tensors" -> {{deltaFundFlav[F1, F2] deltaDirac[d1, d2] deltaFundCol[
      A1, A2]}}]
Print["qbq\[Sigma] constructed!"]


(* ::Subsection:: *)
(*Kernel*)


(* ::Input::Initialization:: *)
kernelParameterList = {
   	<|"Name" -> "k", "Type" -> "double", "AD" -> False, 
    "Const" -> True|>,
   	<|"Name" -> "p", "Type" -> "double", "AD" -> False, 
    "Const" -> True|>,
   	<|"Name" -> "p0f", "Type" -> "double", "AD" -> False, 
    "Const" -> True|>,
   	
   	<|"Name" -> "Nf", "Type" -> "double", "AD" -> False|>,
   	<|"Name" -> "Nc", "Type" -> "double", "AD" -> False|>,
   	<|"Name" -> "T", "Type" -> "double", "AD" -> False|>,
   	<|"Name" -> "muq", "Type" -> "double", "AD" -> False|>,
   	
   	<|"Name" -> "etaQ", "Type" -> "double", "AD" -> True|>,
   	<|"Name" -> "etaPhi", "Type" -> "double", "AD" -> True|>,
   	
   	<|"Name" -> "hPhi", "Type" -> "double", "AD" -> True|>,
   	
   	<|"Name" -> "d1V", "Type" -> "double", "AD" -> True|>,
   	<|"Name" -> "d2V", "Type" -> "double", "AD" -> True|>,
   	<|"Name" -> "d3V", "Type" -> "double", "AD" -> True|>,
   	<|"Name" -> "rhoPhi", "Type" -> "double", "AD" -> False|>
   };


(* ::Input::Initialization:: *)
kerneletaQ = {
   	"Name" -> "etaQ", "Integrator" -> "Integrator_fT_p2_1ang", 
   "d" -> 4, "AD" -> True, "ComputeType" -> "double", "Device" -> "TBB", 
   "Type" -> "auto", "Parameters" -> kernelParameterList,
   	"IntegrationVariables" -> {"l1", "cos1", "l0"}
   };


(* ::Input::Initialization:: *)
kerneletaPhi = {
   	"Name" -> "etaPhi", "Integrator" -> "Integrator_fT_p2_1ang", 
   "d" -> 4, "AD" -> True, "ComputeType" -> "double", "Device" -> "TBB", 
   "Type" -> "auto", "Parameters" -> kernelParameterList,
   	"IntegrationVariables" -> {"l1", "cos1", "l0"}
   };


(* ::Input::Initialization:: *)
kernelV = {
   	"Name" -> "V", "Integrator" -> "Integrator_p2", "d" -> 3, 
   "AD" -> True, "ComputeType" -> "double", "Device" -> "TBB", 
   "Type" -> "auto", "Parameters" -> kernelParameterList,
   	"IntegrationVariables" -> {"l1"}
   };


(* ::Input::Initialization:: *)
kernelhPhi = {
   	"Name" -> "hPhi0", "Integrator" -> "Integrator_fT_p2_1ang", 
   "d" -> 4, "AD" -> True, "ComputeType" -> "double", "Device" -> "TBB", 
   "Type" -> "auto", "Parameters" -> kernelParameterList,
   	"IntegrationVariables" -> {"l1", "cos1", "l0"}
   };


(* ::Subsection:: *)
(*fRG setup and truncation*)


(* ::Input::Initialization:: *)
fields = <|"Commuting" -> {\[Sigma][p], \[CapitalPi][p, {a}]},
            "Grassmann" -> {{qb[p, {d, c, f}], q[p, {d, c, f}]}}
   |>;
truncation = <|
   	GammaN -> {{qb, 
      q}, {\[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi]},
     		{qb, q, \[Sigma]}, {qb, q, \[CapitalPi]}, (* quark-
     meson scatterings *)
     		{\[Sigma], \[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi], \
\[Sigma]}, {\[Sigma], \[Sigma], \[Sigma], \[Sigma]}, {\[CapitalPi], \
\[CapitalPi], \[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi], \
\[CapitalPi], \[CapitalPi]}(* meson scatterings *)
     	},
   	Propagator -> {{q, 
      qb}, {\[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi]}},
   	R -> {{qb, q}, {\[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi]}},
   	Rdot -> {{qb, 
      q}, {\[Sigma], \[Sigma]}, {\[CapitalPi], \[CapitalPi]}},
   	Field -> {{}}
   |>;
bases = <|
   	GammaN -> {
     		{qb, q} -> "qbqFT",
     		{\[CapitalPi], \[CapitalPi]} -> 
      "\[CapitalPi]\[CapitalPi]", {\[Sigma], \[Sigma]} -> 
      "\[Sigma]\[Sigma]",
     
     		{qb, q, \[CapitalPi]} -> "qbq\[CapitalPi]",
     		{qb, q, \[Sigma]} -> "qbq\[Sigma]",
     		{\[CapitalPi], \[CapitalPi], \[CapitalPi], \[CapitalPi]} -> 
      "\[CapitalPi]\[CapitalPi]\[CapitalPi]\[CapitalPi]",
     		{\[CapitalPi], \[CapitalPi], \[Sigma], \[Sigma]} -> 
      "\[CapitalPi]\[CapitalPi]\[Sigma]\[Sigma]",
     		{\[CapitalPi], \[CapitalPi], \[Sigma]} -> 
      "\[CapitalPi]\[CapitalPi]\[Sigma]",
     		{\[Sigma], \[Sigma], \[Sigma]} -> "\[Sigma]\[Sigma]\[Sigma]",
     		{\[Sigma], \[Sigma], \[Sigma], \[Sigma]} -> 
      "\[Sigma]\[Sigma]\[Sigma]\[Sigma]"
     	},
   	Propagator -> {{\[CapitalPi], \[CapitalPi]} -> 
      "\[CapitalPi]\[CapitalPi]", {\[Sigma], \[Sigma]} -> 
      "\[Sigma]\[Sigma]", {qb, q} -> "qbqFT"},
   	R -> {{\[CapitalPi], \[CapitalPi]} -> 
      "\[CapitalPi]\[CapitalPi]", {\[Sigma], \[Sigma]} -> 
      "\[Sigma]\[Sigma]", {qb, q} -> {"qbqFT", 2}},
   	Rdot -> {{\[CapitalPi], \[CapitalPi]} -> 
      "\[CapitalPi]\[CapitalPi]", {\[Sigma], \[Sigma]} -> 
      "\[Sigma]\[Sigma]", {qb, q} -> {"qbqFT", 2}}
   |>;
diagramStyling = <|
   	"Styles" -> { 
     q -> {Black}, \[CapitalPi] -> {Blue}, \[Sigma] -> {Blue, Dashed}}
   |>;
setup = <|
   	"FieldSpace" -> fields,
   	"Truncation" -> truncation,
   	"FeynmanRules" -> bases,
   	"DiagramStyling" -> diagramStyling
   |>;
FSetGlobalSetup[setup];
diagrammaticRules = FMakeDiagrammaticRules[setup];


(* ::Subsection:: *)
(*Dressings*)


(* ::Input::Initialization:: *)
dressingRulesMesons = {
   	(*propagators*)
   	dressing[InverseProp, {\[CapitalPi], \[CapitalPi]}, 
     1, {p1_, p2_}] :> sp[p1, p1] + m2Pi + RPi[p1],
   	dressing[InverseProp, {\[Sigma], \[Sigma]}, 1, {p1_, p2_}] :> 
    sp[p1, p1] + m2Sig + RSig[p1],
   	
   	(*GammaN mesonic*)
   	dressing[GammaN, {\[CapitalPi], \[CapitalPi]}, 1, {p1_, p2_}] :> 
    sp[p1, p1] + m2Pi,
   	dressing[GammaN, {\[Sigma], \[Sigma]}, 1, {p1_, p2_}] :> 
    sp[p1, p1] + m2Sig,
   	
   	dressing[GammaN, {\[Sigma], \[Sigma], \[Sigma]}, 
     1, {p1_, p2_, p3_}] :> 
    3 Sqrt[2 rhoPhi] d2V + Sqrt[2 rhoPhi]^3 d3V,
   	dressing[GammaN, {\[Sigma], \[CapitalPi], \[CapitalPi]}, 
     1, {p1_, p2_, p3_}] :> Sqrt[2 rhoPhi] d2V,
   	
   	dressing[GammaN, {\[Sigma], \[Sigma], \[Sigma], \[Sigma]}, 
     1, {p1_, p2_, p3_, p4_}] :> 3 d2V + 6 rhoPhi d3V,
   	dressing[GammaN, {\[Sigma], \[Sigma], \[CapitalPi], \[CapitalPi]}, 
     1, {p1_, p2_, p3_, p4_}] :> d2V + 2 rhoPhi d3V,
   	dressing[
     GammaN, {\[CapitalPi], \[CapitalPi], \[CapitalPi], \[CapitalPi]}, 
     1, {p1_, p2_, p3_, p4_}] :> 3 d2V,
   	
   	(*Regulators*)
   	dressing[R, {\[CapitalPi], \[CapitalPi]}, 1, {p1_, p2_}] :> 
    RPi[p1],
   	dressing[R, {\[Sigma], \[Sigma]}, 1, {p1_, p2_}] :> RSig[p1],
   	dressing[Rdot, {\[CapitalPi], \[CapitalPi]}, 1, {p1_, p2_}] :> 
    RPidot[p1],
   	dressing[Rdot, {\[Sigma], \[Sigma]}, 1, {p1_, p2_}] :> RSigdot[p1]
   };
dressingRulesQuarks = {
   	(*propagators*)
   	dressing[InverseProp, {qb, q}, 
     1, {p1_, p2_}] :> (vec[p1, 0] + I muq),
   	dressing[InverseProp, {qb, q}, 
     2, {p1_, p2_}] :> (1 + RQ[p1]/Sqrt[sps[p1, p1]]),
   	dressing[InverseProp, {qb, q}, 3, {p1_, p2_}] :> MQ[p1],
   
   	(*GammaN*)
   	dressing[GammaN, {qb, q}, 1, {p1_, p2_}] :> (vec[p2, 0] + I muq),
   	(* The spatial basis element is I gamma.p1, built on the momentum p1 = -p2 of the antiquark leg, while all
   	   other momentum-odd terms are written in the quark momentum p2: hence the explicit minus signs. *)
   	dressing[GammaN, {qb, q}, 2, {p1_, p2_}] :> -1,
   	dressing[GammaN, {qb, q}, 3, {p1_, p2_}] :> MQ[p2],
   
   
   	(*Regulator*)
   	dressing[R, {qb, q}, 2, {p1_, p2_}] :> -RQ[p2]/Sqrt[sps[p2, p2]],
   	dressing[Rdot, {qb, q}, 2, {p1_, p2_}] :> -RQdot[p2]/Sqrt[sps[p2, p2]]
   };
dressingRulesInteraction = {
   	(* The pion vertex is hPhi I gamma5 T^a with tr(T^a T^b) = 1/2; chiral symmetry, and the quark mass
   	   MQ = hPhi sigma/Sqrt[2 Nf], require the sigma vertex hPhi T^0 with T^0 = 1/Sqrt[2 Nf]. *)
   	dressing[GammaN, {\[Sigma], qb, q}, 1, {p1_, p2_, p3_}] :> hPhi/Sqrt[2 Nf],
   	dressing[GammaN, {\[CapitalPi], qb, q}, 1, {p1_, p2_, p3_}] :> hPhi
   };
dressingRules := 
 dressingRulesMesons \[Union] dressingRulesQuarks \[Union] 
  dressingRulesInteraction


(* ::Input::Initialization:: *)
sanityPi = 
  FTerm[FTerm[
     GammaN[{\[CapitalPi], \[CapitalPi]}, {{p1, {a1}}, {p2, {a2}}}] + 
      R[{\[CapitalPi], \[CapitalPi]}, {{p1, {a1}}, {p2, {a2}}}]], 
    Propagator[{\[CapitalPi], \[CapitalPi]}, {{p1, {a1}}, {p2, {a2}}}]] \
//. diagrammaticRules;
sanityPi = sanityPi // FormTrace;
sanityPi = sanityPi //. dressingRulesMesons // Simplify;
Print[sanityPi, 
  " should be -1+\!\(\*SuperscriptBox[\(Nf\), \(2\)]\)\
\[InvisibleSpace]"]; 


(* ::Input::Initialization:: *)
sanitySig = 
  FTerm[FTerm[
     GammaN[{\[Sigma], \[Sigma]}, {{p1}, {p2}}] + 
      R[{\[Sigma], \[Sigma]}, {{p1}, {p2}}]], 
    Propagator[{\[Sigma], \[Sigma]}, {{p1}, {p2}}]] //. 
   diagrammaticRules;
sanitySig = sanitySig // FormTrace;
sanitySig = sanitySig //. dressingRulesMesons // Simplify;
Print[sanitySig, " should be 1"];  


(* ::Input::Initialization:: *)
postTraceRulesMesons = {
   	m2Pi -> d1V,
   	m2Sig -> d1V + 2 rhoPhi d2V,
   	RPi[p_] :> RB[k^2, sps[p, p]],
   	RSig[p_] :> RB[k^2, sps[p, p]],
   	RPidot[p_] :> RBdot[k^2, sps[p, p]] - etaPhi RB[k^2, sps[p, p]],
   	RSigdot[p_] :> RBdot[k^2, sps[p, p]] - etaPhi RB[k^2, sps[p, p]]
   };


(* ::Input::Initialization:: *)
postTraceRulesQuarks = {
   		RQ[p_] :> RF[k^2, sps[p, p]],
   	RQdot[p_] :> RFdot[k^2, sps[p, p]] - etaQ RF[k^2, sps[p, p]],
   	MQ[p_] :> hPhi Sqrt[2 rhoPhi]/Sqrt[2 Nf]
   };


(* ::Input::Initialization:: *)
postTraceRules := postTraceRulesMesons \[Union] postTraceRulesQuarks
qset = TBGetIndexSet["qbqFT", 1, p];
qbset = TBGetIndexSet["qbqFT", 1, -p];
sanityQuark = 
  FTerm[FTerm[
      GammaN[{qb, q}, {qbset, qset}] + R[{qb, q}, {qbset, qset}]], 
     Propagator[{q, qb}, {qset, qbset}]] //. diagrammaticRules //. 
   dressingRulesQuarks;
sanityQuark = sanityQuark // FormTrace // Simplify;
Print[sanityQuark, " should be -4 Nc Nf"]; 


(* ::Input::Initialization:: *)
PropParam[expr_] := ExpandScalarProductsFiniteT[expr] //. {
       p2 -> -p1,
       
       vec[p1, 0] -> p0,
       vec[p, 0] -> p0,
       vec[l1, 0] -> l0,
       vec[lf1, 0] -> l0 + Pi T,
       cos[l1, p1] -> cos1,
       cos[lf1, p1] -> cos1,
       cos[l1, p] -> cos1,
       cos[lf1, p] -> cos1
       } //. {
      
      Sqrt[a_^2] :> a,
      1/Sqrt[a_^2] :> 1/a,
      (a_^2)^(n_/2) :> a^n
      } //. p1 -> p //. lf1 -> l1;


(* ::Section:: *)
(*Flows*)


(* ::Subsection:: *)
(*Quark propagator*)


(* ::Input::Initialization:: *)
fRGqbq = FTakeDerivatives[WetterichEquation, {qb[i1], q[i2]}] // 
     FTruncate // FPlot // FRoute;


(* ::Input::Initialization:: *)
qset = TBGetIndexSet["qbqFT", 1, p1];
qbset = TBGetIndexSet["qbqFT", 1, -p1];
sanityqbq = 
  FTerm[TBGetProjector["qbqFT", 
      2, {qset, qbset}] GammaN[{qb, q}, {qbset, qset}]] /. 
   diagrammaticRules;
sanityqbq = sanityqbq //. dressingRules // FormTrace // Simplify;
sanityqbq = sanityqbq //. postTraceRules


(* ::Input::Initialization:: *)
(* The projector takes the quark index first, so its indices are swapped w.r.t. the derivative order {qb, q}: the
   swap of the two Grassmann legs costs a minus sign. *)
traceExprEtaq = 
  FTerm[-TBGetProjector["qbqFT", 
     2, {i2, i1} /. 
      fRGqbq["1-Loop"]["ExternalIndices"]]] ** (fRGqbq["1-Loop"][
      "Expression"] /. diagrammaticRules);
FlowEtaq = 
  FormTrace[traceExprEtaq] /. dressingRules /. postTraceRules // 
   PropParam;
FlowEtaq = 
 FlowEtaq /. p0 -> p0f - I muq /. Im[muq] -> 0 // Total // 
  QuickSimplify


(* ::Input::Initialization:: *)
MakeKernel[Re[FlowEtaq], kerneletaQ]


(* ::Subsection:: *)
(*Meson propagator*)


(* ::Input::Initialization:: *)
fRGPiPi = 
  FTakeDerivatives[
      WetterichEquation, {\[CapitalPi][i1], \[CapitalPi][i2]}] // 
     FTruncate // FPlot // FRoute;


(* ::Input::Initialization:: *)
traceExprPiPi = 
  FTerm[TBGetProjector["\[CapitalPi]\[CapitalPi]", 
     1, {i1, i2} /. 
      fRGPiPi["1-Loop"]["ExternalIndices"]]] ** (fRGPiPi["1-Loop"][
      "Expression"] /. diagrammaticRules);
FlowPiPi = traceExprPiPi /. dressingRules // FormTrace // Simplify;
FlowPiPi = FlowPiPi //. postTraceRules // PropParam;
FlowEtaPhi = -(1/p^2) (FlowPiPi - (FlowPiPi //. p -> 0)) //. p0 -> 0 //
    Total // QuickSimplify


(* ::Input::Initialization:: *)
MakeKernel[Re[FlowEtaPhi], kerneletaPhi]


(* ::Subsection:: *)
(*Yukawa coupling*)


(* ::Input::Initialization:: *)
fRGPhiqbq = 
  FTakeDerivatives[WetterichEquation, {qb[i1], q[i2]}] // FTruncate //
     FPlot // FRoute;


(* ::Input::Initialization:: *)
qset = TBGetIndexSet["qbqFT", 1, p1];
qbset = TBGetIndexSet["qbqFT", 1, -p1];
sanityqbq = 
  FTerm[TBGetProjector["qbqFT", 
      3, {i2, i1} /. 
       fRGPhiqbq["1-Loop"]["ExternalIndices"]] GammaN[{qb, 
       q}, {i1, i2} /. fRGPhiqbq["1-Loop"]["ExternalIndices"]]] /. 
   diagrammaticRules;
sanityqbq = sanityqbq /. dressingRules // FormTrace // Simplify;
sanityqbq = sanityqbq //. postTraceRules // PropParam;
sanityqbq = sanityqbq/(Sqrt[rhoPhi]/Sqrt[Nf]);
Print[sanityqbq, " should be hPhi"]


(* ::Input::Initialization:: *)
traceExprPhiqbq = 
  FTerm[TBGetProjector["qbqFT", 
     3, {i1, i2} /. 
      fRGPhiqbq["1-Loop"]["ExternalIndices"]]] ** (fRGPhiqbq["1-Loop"][
      "Expression"] /. diagrammaticRules);
FlowPhiqbq = FormTrace[traceExprPhiqbq] /. dressingRules // Simplify;
FlowPhiqbq = FlowPhiqbq //. postTraceRules // PropParam;
FlowPhiqbq = FlowPhiqbq /. p0 -> p0f - I muq /. Im[muq] -> 0;
FlowPhiqbq = 
 FlowPhiqbq/(Sqrt[rhoPhi]/Sqrt[Nf]) // Total // QuickSimplify


(* ::Input::Initialization:: *)
(* The fields are renormalized (Z = 1), so the Yukawa coupling picks up the rescaling (etaPhi/2 + etaQ) hPhi. *)
MakeKernel[Re[FlowPhiqbq], (etaPhi/2 + etaQ) hPhi, kernelhPhi]


(* ::Subsection:: *)
(*Potential*)


(* ::Input::Initialization:: *)
fRGV = WetterichEquation // FTruncate // FPlot // FRoute // FPrint;


(* ::Input::Initialization:: *)
(*Set up the assumptions for the Matsubara Sum*)
$Assumptions = $Assumptions && muq >= 0 && l1 > 0 && 
   k > 0 && -1 < cos < 1 && -d1V - l1^2 - RB[k^2, l1^2] < 
    0 && -m2Sigma - l1^2 - RB[k^2, l1^2] < 
    0 && -m2Quark - l1^2 - 2 l1 RF[k^2, l1^2] - RF[k^2, l1^2]^2 < 0;
flowV = fRGV /. diagrammaticRules;
flowV = flowV["1-Loop"]["Expression"] // FormTrace;
(* Need to change the notation since ExpandScalarProductsFiniteT, \
need the loop momentum of fermions to have the f at the end*)
flowV = flowV //. dressingRules /. lf1 -> l1f;
flowV = ExpandScalarProductsFiniteT[flowV] //. postTraceRules // 
   UseLorentzLinearity;
flowV = flowV // SimplifyAllMomenta[l1, #] &;
flowV = flowV /. sps[l1, l1] -> l1^2 /. sp[l1, l1] -> l1^2 + l0^2 /. 
    Sqrt[a_^2] :> a /. (1/Sqrt[a_^2] :> 1/a);
(*Add change of variable for easier assumption condition*)
flowV = flowV //. 2 d2V rhoPhi -> m2Sigma - d1V //. 
   hPhi -> Sqrt[(m2Quark 2 Nf)/(2 rhoPhi)] // Simplify
flowV = Map[MatsubaraSum[#, l10, T] &, flowV];
flowV = flowV //. m2Sigma -> d1V + 2 rhoPhi d2V //. 
    m2Quark -> hPhi^2/(2 Nf) 2 rhoPhi // FullSimplify;
flowV = flowV // SafeFiniteTFunctions[#, T] & // Simplify


(* ::Input::Initialization:: *)
MakeKernel[flowV, kernelV]


(* ::Subsection:: *)
(*Collect the kernels*)


(* ::Input::Initialization:: *)
UpdateFlows["LPA_QM_flows"]
