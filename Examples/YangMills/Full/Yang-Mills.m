(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



(* ::Input::Initialization:: *)
Get["DiFfRG`"]
SetDirectory[GetDirectory[]];
$Assumptions=q>0&&k>0&&p>0&&r>0&&Sqrt[p^2]>0&&Nc>0&&p1>0&&p2>0&&-1<cos1<1&&-1<cos2<1;
DefineFormExecutable["/usr/bin/form"]

DefineFormAutoDeclareFunctions[gA,ZA,eta,m2];
AddExtraVars[
k,cos,ArcTan,cosp1p2,
(*regulators*)
RA,RAdot,Rc,Rcdot,RF,RFdot,RB,RBdot,
GAInv,GA,GcInv,Gc,m2A,dtm2A,
(*Angles for symmetric points*)
cosp1q,cosp2q,cosp3q,cosp4q,
(*wavefunction renormalizations*)
Zc,ZA,dtZA];


TBUnregister["AAA"]//Quiet
TBUnregister["AAA2"]//Quiet
TBImportBasis["../../bases/AAA.m"]
TBImportBasis["../../bases/AAA2.m"]

TBUnregister["AAAA"]//Quiet
TBUnregister["AAAA2"]//Quiet
TBImportBasis["../../bases/AAAA.m"]
TBImportBasis["../../bases/AAAA2.m"]

TBUnregister["Acbc"]//Quiet
TBImportBasis["../../bases/Acbc.m"]

TBUnregister["AA"]//Quiet
TBImportBasis["../../bases/AA.m"]


transf3PTo=TB3PToS0S1SPhi[p1,p2,p3,q,S0,S1,SPhi];
transf3PFrom=TB3PFromS0S1SPhi[p1,p2,p3,S0,S1,SPhi];


transfTadpoleTo=TB3PToS0S1SPhiQk[p1,p2,q,S0,S1,SPhi];
transfTadpoleFrom=TB3PFromS0S1SPhiQk[p1,p2,S0,S1,SPhi];


P3Patt[p1e_,p2e_,p3e_]:={transf3PFrom[S0],transf3PFrom[S1],transf3PFrom[SPhi]}/.{p1:>p1e,p2:>p2e,p3:>p3e}//UseLorentzLinearity//FullSimplify
tadpolePatt[p1e_,p2e_]:={transfTadpoleFrom[S0],transfTadpoleFrom[S1],transfTadpoleFrom[SPhi]}/.{p1:>p1e,p2:>p2e}//UseLorentzLinearity//FullSimplify


SP3Patt[p1e_,p2e_,p3e_]:={Sqrt[(sp[p1,p1]+sp[p2,p2]+sp[p3,p3])/3]}/.{p1:>p1e,p2:>p2e,p3:>p3e}//UseLorentzLinearity//FullSimplify;
SP4Patt[p1e_,p2e_,p3e_,p4e_]:={Sqrt[(sp[p1,p1]+sp[p2,p2]+sp[p3,p3]+sp[p4,p4])/4]}/.{p1:>p1e,p2:>p2e,p3:>p3e,p4:>p4e}//UseLorentzLinearity//FullSimplify;


a=(tadpolePatt[p,q]//.sp[p,q]:>0//FullSimplify)//Quiet;
a[[3]]=\[Pi]/2;
a//QCDSimp


PostTraceRulesGluons={
evP:>(k^nZA+1)^(1/nZA),
devP:>k^(-1+nZA) (1+k^nZA)^(-1+1/nZA),
RAdot[p_]:>ZA[evP]RBdot[k^2,sp[p,p]]+RB[k^2,sp[p,p]](dtZA[evP]+k*devP*(ZA[1.02evP]-ZA[evP])/(0.02*evP)),
RA[p_]:>ZA[evP]RB[k^2,sp[p,p]],
GA[p_]:>1/(ZA[Sqrt[sp[p,p]]] sp[p,p]+RA[p]),
GAInv[p_]:>ZA[Sqrt[sp[p,p]]]sp[p,p],
nZA:>6,

ZA3p[p1_,p2_]:>ZA3@@P3Patt[p1,p2,-p1-p2],
ZAcbcp[p1_,p2_]:>ZAcbc@@P3Patt[p1,p2,-p1-p2],
ZA4p[p1_,p2_,p3_]:>ZA4SP@@SP4Patt[p1,p2,p3,-p1-p2-p3],
ZA4t[p1_,p2_,p3_]:>ZA4tadpole@@tadpolePatt[p1,p3]
};


(* ::Input::Initialization:: *)
PreTraceRulesGluonsTadpole={
\[CapitalGamma]AAAA[{p1_,v1_,a1_,p2_,v2_,a2_,p3_,v3_,a3_,p4_,v4_,a4_}] :>ZA4t[p1,p2,p3]TBGetVertex["AAAA2",1,{p1,v1,a1},{p2,v2,a2},{p3,v3,a3},{p4,v4,a4}]
};


(* ::Input::Initialization:: *)
PreTraceRulesGluons={
(*Regulators*)
RAA[{p1_,v1_,a1_,p2_,v2_,a2_}]:>RA[p2]TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}],

(*Regulator Derivatives*)
RdotAA[{p1_,v1_,a1_,p2_,v2_,a2_}]:>RAdot[p2]TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}],

(*Propagators*)
\[CapitalGamma]AA[{p1_,v1_,a1_,p2_,v2_,a2_}] :> GAInv[p2]TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}],
GAA[{p1_,v1_,a1_,p2_,v2_,a2_}] :>GA[p1]TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}],

(*Pure gauge Vertices*)
\[CapitalGamma]AAA[{p1_,v1_,a1_,p2_,v2_,a2_,p3_,v3_,a3_}] :>ZA3p[p1,p2] TBGetVertex["AAA2",1,{p1,v1,a1},{p2,v2,a2},{p3,v3,a3}],
\[CapitalGamma]Acbc[{p1_,v1_,a1_,p2_,a2_,p3_,a3_}] :>ZAcbcp[p1,p2]  TBGetVertex["Acbc",1,{p1,v1,a1},{p2,a2},{p3,a3}],
\[CapitalGamma]AAAA[{p1_,v1_,a1_,p2_,v2_,a2_,p3_,v3_,a3_,p4_,v4_,a4_}] :>ZA4p[p1,p2,p3]TBGetVertex["AAAA2",1,{p1,v1,a1},{p2,v2,a2},{p3,v3,a3},{p4,v4,a4}]
};

(*Must be 3(Nc^2-1)*)
sanity=FormTrace[(\[CapitalGamma]AA[{p,v1,a1,-p,v2,a2}]+RAA[{p,v1,a1,-p,v2,a2}]) GAA[{-p,v2,a2,p,v1,a1}]//.PreTraceRulesGluons]//.PostTraceRulesGluons//Simplify;
Print["Trace is ", sanity, " should be ", 3(-1+Nc^2)];


PostTraceRulesGhosts:={
Rcdot[p_]:>Zc[k]RBdot[k^2,sp[p,p]]+RB[k^2,sp[p,p]](dtZc[k]+k (Zc[1.02*k]-Zc[k])/(0.02*k)),
Rc[p_]:>Zc[k]RB[k^2,sp[p,p]],
GcInv[p_]:>Zc[Sqrt[sp[p,p]]]sp[p,p],
Gc[p_]:>1/(Zc[Sqrt[sp[p,p]]]sp[p,p]+Rc[p])
}


(* ::Input::Initialization:: *)
PreTraceRulesGhosts:={
(*Regulators*)
Rcbc[{p1_,a1_,p2_,a2_}]:>deltaAdjCol[a1,a2] Rc[p2],

(*Regulator Derivatives*)
Rdotcbc[{p1_,a1_,p2_,a2_}]:>deltaAdjCol[a1,a2] Rcdot[p2],

(*Propagators*)
\[CapitalGamma]cbc[{p1_,a1_,p2_,a2_}]:>-deltaAdjCol[a1,a2]   GcInv[p2],
Gccb[{p1_,a1_,p2_,a2_}]:>deltaAdjCol[a1,a2]Gc[p1]
}

(*Must be 1-Nc^2*)
sanity=FormTrace[(\[CapitalGamma]cbc[{-q,a1,q,a2}]-Rcbc[{-q,a1,q,a2}])Gccb[{q,a2,-q,a1}]//.PreTraceRulesGhosts,{{PreambleFormRule,"Vector q;"}}]//.PostTraceRulesGhosts//Simplify;
Print["Trace is ", sanity, " should be ",- Nc^2+1];


(* ::Input::Initialization:: *)
PreTraceRules:=PreTraceRulesGluons\[Union]PreTraceRulesGhosts
PostTraceRules:=PostTraceRulesGhosts\[Union]PostTraceRulesGluons


(* ::Input::Initialization:: *)
fRGEq = {
"Prefactor"->{1/2},
<|"type"->"Regulatordot", "indices"->{i,j}|>,
<|"type"->"Propagator", "indices"->{i,j}|>
};
fields = <|
"bosonic"-> { 
A[p,{v, c}]
},
"fermionic"->{
{cb[p,{c}],c[p,{c}]}
}
|>;
Truncation = {
{A,A},{cb,c},(* propagators *)
{cb,c,A},{A,A,A},{A,A,A,A} (* glue sector scatterings *)
};
SetupfRG = <|
"MasterEquation"->fRGEq,
"FieldSpace"->fields,
"Truncation"->Truncation
|>;
DiagramStyle={
A->Orange,c->{Dotted,Black}
};


AddCodeOptimizeFunctions[RBdot[__],RFdot[__],RB[__],RF[__]]
ShowCodeOptimizeFunctions[]


(* ::Input::Initialization:: *)
kernelZA=<|"Path"->"AA","Name"->"ZA","Type"->"Quadrature","Angles"->1,"d"->4,"AD"->False,"ctype"->"double","Device"->"GPU"|>;
kernelm2A=<|"Path"->"AA","Name"->"m2A","Type"->"Quadrature","Angles"->1,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelZc=<|"Path"->"cbc","Name"->"Zc","Type"->"Quadrature","Angles"->1,"d"->4,"AD"->False,"ctype"->"double","Device"->"GPU"|>;
kernelZA3=<|"Path"->"AAA","Name"->"ZA3","Type"->"Quadrature","Angles"->2,"d"->4,"AD"->False,"ctype"->"double","Device"->"GPU"|>;
kernelZAcbc=<|"Path"->"Acbc","Name"->"ZAcbc","Type"->"Quadrature","Angles"->2,"d"->4,"AD"->False,"ctype"->"double","Device"->"GPU"|>;
kernelZA4tadpole=<|"Path"->"AAAA","Name"->"ZA4tadpole","Type"->"Quadrature","Angles"->2,"d"->4,"AD"->False,"ctype"->"double","Device"->"GPU"|>;
kernelZA4SP=<|"Path"->"AAAA","Name"->"ZA4SP","Type"->"Quadrature","Angles"->3,"d"->4,"AD"->False,"ctype"->"double","Device"->"GPU"|>;

kernels={
kernelZA,kernelm2A,kernelZc,
kernelZA3,kernelZAcbc,kernelZA4tadpole,kernelZA4SP
};

kernelParameterListBase={
(*strong couplings*)
<|"Name"->"ZA3","Type"->"FunctionTex3DLogLinLin","AD"->False|>,
<|"Name"->"ZAcbc","Type"->"FunctionTex3DLogLinLin","AD"->False|>,
<|"Name"->"ZA4SP","Type"->"FunctionTex1D","AD"->False|>,
<|"Name"->"ZA4tadpole","Type"->"FunctionTex3DLogLinLin","AD"->False|>,
(*ghost propagator*)
<|"Name"->"dtZc","Type"->"FunctionTex1D","AD"->False|>,
<|"Name"->"Zc","Type"->"FunctionTex1D","AD"->False|>,
(*glue propagator*)
<|"Name"->"dtZA","Type"->"FunctionTex1D","AD"->False|>,
<|"Name"->"ZA","Type"->"FunctionTex1D","AD"->False|>,
<|"Name"->"m2A","Type"->"Variable","AD"->False|>
};

kernelParameterList=Join[{
<|"Name"->"p","Type"->"Constant","AD"->False|>
},kernelParameterListBase];

kernelParameterListFull=Join[{
<|"Name"->"S0","Type"->"Constant","AD"->False|>,
<|"Name"->"S1","Type"->"Constant","AD"->False|>,
<|"Name"->"SPhi","Type"->"Constant","AD"->False|>
},kernelParameterListBase];


couplingsList={
ZA3[__], ZA4SP[__], ZAcbc[__],ZA4tadpole[__],
ZA3p[__],ZA4p[__],ZA4t[__],ZAcbcp[__]
};
otherList={
dtZA[__],ZA[__],etac[__],RA[__],Rc[__],RAdot[__],Rcdot[__]
};
QCDSimp[expr_]:=Module[{ret},
ret=expr//.Nc->3//.qf->q//UseLorentzLinearity//expandScalarProducts//simplifyAllMomenta[q,#]&;
ret=ret/.Map[(Head[#][a___]:>FullSimplify[Head[#][a]])&,couplingsList\[Union]otherList];
ret=Collect[ret,couplingsList,QuickSimplify];
ret=Collect[ret,couplingsList,Simplify];
Return[ret]
];
QCDQuickSimp[expr_]:=Module[{ret},
ret=expr//.Nc->3//.qf->q//UseLorentzLinearity//expandScalarProducts//simplifyAllMomenta[q,#]&;
ret=ret/.Map[(Head[#][a___]:>FullSimplify[Head[#][a]])&,couplingsList\[Union]otherList];
ret=Collect[ret,couplingsList,QuickSimplify];
Return[ret]
];
SetStandardSimplify[QCDSimp];
SetStandardQuickSimplify[QCDQuickSimp];


DeclareAngles[q_,p_,r_]:=Module[
{
Resultp,Resultr,
Resultcospq,Resultcosqr,Resultcospr,

namecospq,namecosqr,namecospr,

code},

Resultcospq=transf3PTo[sp[p,q]/(q Sqrt[sp[p,p]])]//FullSimplify;
Resultcosqr=transf3PTo[sp[r,q]/(q Sqrt[sp[r,r]])]//FullSimplify;
Resultcospr=transf3PTo[sp[p,r]/Sqrt[sp[p,p]sp[r,r]]]//FullSimplify;
Resultp=transf3PTo[Sqrt[sp[p,p]]]//FullSimplify;
Resultr=transf3PTo[Sqrt[sp[r,r]]]//FullSimplify;

SetAttributes[cos,Orderless];
namecospq=cos[p,q]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
namecosqr=cos[q,r]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
namecospr=cos[p,r]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
ClearAll[cos];

code=""<>
"const double "<>ToString[namecospq]<>" = "<>CodeForm[Resultcospq]<>";\n"<>
"const double "<>ToString[namecosqr]<>" = "<>CodeForm[Resultcosqr]<>";\n"<>
"const double "<>ToString[namecospr]<>" = "<>CodeForm[Resultcospr]<>";\n"<>
"const double "<>ToString[p]<>" = "<>CodeForm[Resultp]<>";\n"<>
"const double "<>ToString[r]<>" = "<>CodeForm[Resultr]<>";";

Return[code];
];


DeclareAnglesTadpole[q_,p_,r_]:=Module[
{
Resultp,Resultr,
Resultcospq,Resultcosqr,Resultcospr,

namecospq,namecosqr,namecospr,

code},

Resultcospq=transfTadpoleTo[sp[p,q]/(q Sqrt[sp[p,p]])]//FullSimplify;
Resultcosqr=transfTadpoleTo[sp[r,q]/(q Sqrt[sp[r,r]])]//FullSimplify;
Resultcospr=transfTadpoleTo[sp[p,r]/Sqrt[sp[p,p]sp[r,r]]]//FullSimplify;
Resultp=transfTadpoleTo[Sqrt[sp[p,p]]]//FullSimplify;
Resultr=transfTadpoleTo[Sqrt[sp[r,r]]]//FullSimplify;

SetAttributes[cos,Orderless];
namecospq=cos[p,q]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
namecosqr=cos[q,r]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
namecospr=cos[p,r]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
ClearAll[cos];

code=""<>
"const double "<>ToString[namecospq]<>" = "<>CodeForm[Resultcospq]<>";\n"<>
"const double "<>ToString[namecosqr]<>" = "<>CodeForm[Resultcosqr]<>";\n"<>
"const double "<>ToString[namecospr]<>" = "<>CodeForm[Resultcospr]<>";\n"<>
"const double "<>ToString[p]<>" = "<>CodeForm[Resultp]<>";\n"<>
"const double "<>ToString[r]<>" = "<>CodeForm[Resultr]<>";";

Return[code];
];


(* ::Input::Initialization:: *)
MakeFlowClass["YangMills",kernels]


(* ::Input::Initialization:: *)
SP4Defs=DeclareSymmetricPoints4DP4[];
SP3Defs=DeclareSymmetricPoints4DP3[];
Full3PDefinitions=DeclareAngles[q,p1,p2];
FullTadpoleDefinitions=DeclareAnglesTadpole[q,p1,p2];


(*Diagrams*)
DerivativeListAA= {A[-p,{v1,a1}],A[p,{v2,a2}]};
DiagramsAAsidx =DeriveFunctionalEquation[SetupfRG,DerivativeListAA,"OutputLevel"->"SuperindexDiagrams"];
DiagramsAAsidx=ReduceIdenticalFlowDiagrams[DiagramsAAsidx,DerivativeListAA];
DiagramsAA=SuperindexToFullDiagrams[DiagramsAAsidx,SetupfRG,DerivativeListAA];
Print["DiagramsAA got "<>ToString[Length[DiagramsAA]]<>" diagrams."]
PlotSuperindexDiagram[DiagramsAAsidx,SetupfRG,"EdgeStyle"->DiagramStyle]


(*Projection*)
ProjectorZA=TBGetProjector["AA",1,{-p,v1,a1},{p,v2,a2}];

(*Sanity Check*)
sanity=(FormTrace[ProjectorZA ((\[CapitalGamma]AA[{-p,v2,a2,p,v1,a1}]-\[CapitalGamma]AA[{-q,v2,a2,q,v1,a1}])/sp[p,p])//.PreTraceRules]//.PostTraceRules//expandScalarProducts)//.{cos[p,q]->1,q->0}//Simplify;
Print["Projection check is ", sanity, ", should be ZA[p]"]

(*Sanity Check*)
sanity=simplifyAllMomenta[q,FormTrace[ProjectorZA \[CapitalGamma]AA[{-p,v2,a2,p,v1,a1}]//.PreTraceRules]//.PostTraceRules//expandScalarProducts//Simplify]//.p->0;
Print["Projection check is ", sanity, ", should be m2A"]


ProjectionZA=(ProjectorZA DiagramsAA)//.PreTraceRulesGluonsTadpole//.PreTraceRules;
TraceDiagrams[4,"ZA",ProjectionZA]

ZA0Loop=SumDiagrams[4,"ZA",0,QCDSimp[QCDSimp[#//.PostTraceRules]//.cospq->cos1//.p->0]&,"sum0"];
ZALoop=SumDiagrams[4,"ZA",0,QCDSimp[#//.PostTraceRules]//.cospq->cos1&,"sum"];

MakeKernel[kernelZA,kernelParameterList,ZALoop/p^2]
MakeKernel[kernelm2A,kernelParameterList,ZA0Loop]


(*Diagrams*)
DerivativeListcbc= {cb[-p,{a1}],c[p,{a2}]};
Diagramscbcsidx =DeriveFunctionalEquation[SetupfRG,DerivativeListcbc,"OutputLevel"->"SuperindexDiagrams"];
Diagramscbcsidx=ReduceIdenticalFlowDiagrams[Diagramscbcsidx];
Diagramscbc=SuperindexToFullDiagrams[Diagramscbcsidx,SetupfRG,DerivativeListcbc];
Print["Diagramscbc got "<>ToString[Length[Diagramscbc]]<>" diagrams."]
PlotSuperindexDiagram[Diagramscbcsidx,SetupfRG,"EdgeStyle"->DiagramStyle]


(*Projection*)
NormZc=deltaAdjCol[a2,a1]  deltaAdjCol[a1,a2]  //.PreTraceRules//FormTrace;
ProjectorZc=-deltaAdjCol[a2,a1]/ NormZc ;
(*Sanity Check*)
sanity=(ProjectorZc \[CapitalGamma]cbc[{-p,a1,p,a2}]/sp[p,p]//.PreTraceRules//FormTrace)//.PostTraceRules//QCDQuickSimp;
Print["Projection check is ", sanity, ", should be Zc[p]"]


ProjectionZc=(ProjectorZc Diagramscbc/sp[p,p])//.PreTraceRules;
TraceDiagrams[2,"Zc",ProjectionZc]

ZcLoop=SumDiagrams[2,"Zc",0,QCDSimp[#//.PostTraceRules]&,"sum"]//.cospq->cos1;
MakeKernel[kernelZc,kernelParameterList,ZcLoop]


(*Diagrams*)
DerivativeListA3= {A[p1,{v1,a1}],A[p2,{v2,a2}],A[-p2-p1,{v3,a3}]};
DiagramsA3sidx =DeriveFunctionalEquation[SetupfRG,DerivativeListA3,"OutputLevel"->"SuperindexDiagrams"];
DiagramsA3sidx=ReduceIdenticalFlowDiagrams[DiagramsA3sidx];
DiagramsA3=SuperindexToFullDiagrams[DiagramsA3sidx,SetupfRG,DerivativeListA3];
Print["DiagramsA3 got "<>ToString[Length[DiagramsA3]]<>" diagrams."]
PlotSuperindexDiagram[DiagramsA3sidx,SetupfRG,"EdgeStyle"->DiagramStyle]


(*Projection*)
ProjectorZA3=TBGetProjector["AAA",1,{p1,v1,a1},{p2,v2,a2},{-p2-p1,v3,a3}]//QCDSimp;
(*Sanity check*)
sanity=Simplify[TBGetProjector["AAA",1,{p1,v1,a1},{p2,v2,a2},{-p2-p1,v3,a3}] \[CapitalGamma]AAA[{p1,v1,a1,p2,v2,a2,p3,v3,a3}]//.PreTraceRules//FormTrace]//.PostTraceRules//transf3PTo//FullSimplify;
Print["Projection check is ", sanity, ", should be ZA3[S0,S1,SPhi]"]


postRepRule=Join[FormMomentumExpansion[q,p1,p2,p3],{"id Nc=3;"}];
SetDisentangle[True];

ProjectionZA3=(ProjectorZA3 DiagramsA3)//.PreTraceRules;
TraceDiagrams[8,"ZA3",ProjectionZA3,postRepRule];

ZA3Loop=SumDiagrams[8,"ZA3",0,QCDSimp[#//.PostTraceRules]&,"sum"];
MakeKernel[kernelZA3,kernelParameterListFull, ZA3Loop,0,Full3PDefinitions]


(*Diagrams*)
DerivativeListAcbc={ A[p1,{v1,a1}],cb[p2,{a2}],c[-p2-p1,{a3}]};
DiagramsAcbcsidx =DeriveFunctionalEquation[SetupfRG,DerivativeListAcbc,"OutputLevel"->"SuperindexDiagrams"];
DiagramsAcbcsidx=ReduceIdenticalFlowDiagrams[DiagramsAcbcsidx];
DiagramsAcbc=SuperindexToFullDiagrams[DiagramsAcbcsidx,SetupfRG,DerivativeListAcbc];
Print["DiagramsAcbc got "<>ToString[Length[DiagramsAcbc]]<>" diagrams."]
PlotSuperindexDiagram[DiagramsAcbcsidx,SetupfRG,"EdgeStyle"->DiagramStyle]


ProjectorZAcbc=TBGetProjector["Acbc",1,{p1,v1,a1},{-p2-p1,a3},{p2,a2}]//UseLorentzLinearity//FullSimplify;


TBGetVertex["Acbc",1]


base=I FCol[a1,a2,a3] (ZAcbcCL* vec[p2,v1]+ZAcbcNCL*vec[p1,v1])
ProjectorZAcbc base//FormTrace//Simplify


(*Projection*)
ProjectorZAcbc=TBGetProjector["Acbc",1,{p1,v1,a1},{-p2-p1,a3},{p2,a2}]//UseLorentzLinearity//FullSimplify;
(*Sanity check*)
sanity=Simplify[ProjectorZAcbc \[CapitalGamma]Acbc[{p1,v1,a1,p2,a2,-p1-p2,a3}]//.PreTraceRules//FormTrace]//.PostTraceRules//transf3PTo//expandScalarProducts//FullSimplify;
Print["Projection check is ", sanity, ", should be ZAcbc[S0,S1,SPhi]"]


ClearTraceCache["ZAcbc"]


postRepRule=Join[FormMomentumExpansion[q,p1,p2,p3],{"id Nc=3;"}];
SetDisentangle[True];

ProjectionAcbc=(ProjectorZAcbc DiagramsAcbc)//.PreTraceRules;
TraceDiagrams[8,"ZAcbc",ProjectionAcbc,{},postRepRule];

ZAcbcLoop=SumDiagrams[8,"ZAcbc",0,QCDSimp[#//.PostTraceRules]&,"sum"];
MakeKernel[kernelZAcbc,kernelParameterListFull, ZAcbcLoop,0,Full3PDefinitions]


(*Diagrams*)
DerivativeListA4= {A[p1,{v1,a1}],A[p2,{v2,a2}],A[p3,{v3,a3}],A[-p1-p2-p3,{v4,a4}]};
DiagramsA4sidx =DeriveFunctionalEquation[SetupfRG,DerivativeListA4,"OutputLevel"->"SuperindexDiagrams"];

symmetries=Map[Join[Flatten[List@@#,1],{Plus}]&,Map[PermutationCycles,Permutations[{1,2,3,4},{4}]]][[2;;]];

DiagramsA4sidx=ReduceIdenticalFlowDiagrams[DiagramsA4sidx,DerivativeListA4,symmetries];
DiagramsA4=SuperindexToFullDiagrams[DiagramsA4sidx,SetupfRG,DerivativeListA4];
Print["DiagramsA4 got "<>ToString[Length[DiagramsA4]]<>" diagrams."]
PlotSuperindexDiagram[DiagramsA4sidx,SetupfRG,"EdgeStyle"->DiagramStyle]


(*Projection*)
ProjectorZA4=(TBGetProjector["AAAA",1,{p1,v1,a1},{p2,v2,a2},{p3,v3,a3},{-p1-p2-p3,v4,a4}]//ProjectToSymmetricPoint[#,q,p,p1,p2,p3,p4]&)//QCDSimp;
(*Sanity check*)
sanity=(Simplify[ProjectorZA4  \[CapitalGamma]AAAA[{p1,v1,a1,p2,v2,a2,p3,v3,a3,-p1-p2-p3,v4,a4}]//.PreTraceRules//FormTrace]//ProjectToSymmetricPoint[#,q,p,p1,p2,p3,p4]&)//.PostTraceRules//QCDSimp;
Print["Projection check is ", sanity, ", should be \[InvisibleSpace]ZA4[p]"]


postRepRule=Join[MakeSPFormRule[q,p,p1,p2,p3,p4],{"id Nc=3;"}];
SetDisentangle[True];
DefineFormExecutable["tform"]

ProjectionZA4=(ProjectorZA4 DiagramsA4)//.PreTraceRules;
TraceDiagrams[5,"ZA4",ProjectionZA4,{},postRepRule];

ZA4Loop=SumDiagrams[5,"ZA4",0,QCDSimp[#//.PostTraceRules//ProjectToSymmetricPoint[#,q,p,p1,p2,p3,p4]&]&,"sum"];
MakeKernel[kernelZA4SP,kernelParameterList,ZA4Loop,0,SP4Defs]


(*Diagrams*)
DerivativeListA4= {A[p1,{v1,a1}],A[p2,{v2,a2}],A[-p1,{v3,a3}],A[-p2,{v4,a4}]};
DiagramsA4sidx =DeriveFunctionalEquation[SetupfRG,DerivativeListA4,"OutputLevel"->"SuperindexDiagrams"];

DiagramsA4sidx=ReduceIdenticalFlowDiagrams[DiagramsA4sidx,DerivativeListA4];
DiagramsA4=SuperindexToFullDiagrams[DiagramsA4sidx,SetupfRG,DerivativeListA4];
Print["DiagramsA4 got "<>ToString[Length[DiagramsA4]]<>" diagrams."]
PlotSuperindexDiagram[DiagramsA4sidx,SetupfRG,"EdgeStyle"->DiagramStyle]


(*Projection*)
ProjectorZA4=(TBGetProjector["AAAA",1,{p1,v1,a1},{p2,v2,a2},{-p1,v3,a3},{-p2,v4,a4}])//UseLorentzLinearity//FullSimplify//QCDSimp;
(*Sanity check*)
sanity=Simplify[ProjectorZA4 ZA4[p1,p2,-p1,-p2]TBGetVertex["AAAA2",1,{p1,v1,a1},{p2,v2,a2},{-p1,v3,a3},{-p2,v4,a4}]//.PreTraceRules//FormTrace[#,{{PreambleFormRule,"Vector p3;"}}]&]//.PostTraceRules//QCDSimp;
Print["Projection check is ", sanity, ", should be \[InvisibleSpace]\[InvisibleSpace]ZA4[p1,p2,-p1,-p2]\[InvisibleSpace]"]


postRepRule=Join[FormMomentumExpansion[q,p1,p2,p3,p4],{"id Nc=3;"}];
SetDisentangle[True];
DefineFormExecutable["tform"]

ProjectionZA4=(ProjectorZA4 DiagramsA4)//.PreTraceRules;
TraceDiagrams[0,"ZA4_tadpole",ProjectionZA4,{},postRepRule];

ZA4Loop=SumDiagrams[5,"ZA4_tadpole",0,QCDSimp[#//.PostTraceRules]&,"sum"];
MakeKernel[kernelZA4tadpole,kernelParameterListFull,ZA4Loop,0,FullTadpoleDefinitions]



