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
$Assumptions=q>0&&k>0&&p>0&&-1<cospq<1&&T>0&&muq>=0&&qf>0&&p0f>0&&q0>0;

DefineFormAutoDeclareFunctions[ZA,lambda,gA];
AddExtraVars[k,p0f,m2A,muq,T,
(*regulators*)RQ,RQdot,RA,RAdot,
(*Angles*)cospq,
(*wavefunction renormalizations*)ZA,ZQ,
(*propagators*)GA,GAInv,GQ,MQ];


TBUnregister["AqbqDirect"]//Quiet
TBImportBasis["../../bases/AqbqDirect.m"]

TBUnregister["FierzCompleteNf2Nc34D"]//Quiet
TBImportBasis["../../bases/FierzCompleteNf2Nc34D.m"]

TBUnregister["AA"]//Quiet
TBImportBasis["../../bases/AA.m"]


basisName="FierzCompleteNf2Nc34D";
TBInfo[basisName]


PostTraceRulesQuarks={ 
RQ[p_]:>RF[k^2,sps[p,p]],
RQdot[p_]:>RFdot[k^2,sps[p,p]]-etaQ RF[k^2,sps[p,p]],
GQ[p_]:>1/((vec[p,0]+I muq)^2+sps[p,p](1+ RQ[p]/Sqrt[sps[p,p]])^2+MQ[p]^2)
};


(* ::Input::Initialization:: *)
PreTraceRulesQuarks={
(*Regulators*)
Rqbq[{p1_,d1_,c1_, f1_,p2_,d2_, c2_,f2_}]:>deltaFundCol[c1,c2]deltaFundFlav[f1,f2] (I  psdash[p2,d1,d2]) RQ[p2]/Sqrt[sps[p2,p2]],

(*Regulator Derivatives*)
Rdotqbq[{p1_,d1_,c1_, f1_,p2_,d2_, c2_,f2_}]:>deltaFundCol[c1,c2]deltaFundFlav[f1,f2] (I  psdash[p2,d1,d2]) RQdot[p2]/Sqrt[sps[p2,p2]],

(*Inverse Propagators*)
\[CapitalGamma]qbq[{p1_,d1_,c1_, f1_,p2_,d2_, c2_,f2_}] :>
-deltaFundCol[c1,c2](
deltaFundFlav[f1,f2]((I (vec[p2,0]+I muq)gamma[0,d1,d2])+ I psdash[p2,d1,d2]+  MQ[p2]deltaDirac[d1,d2])
) ,

(*Propagators*)
Gqqb[{p1_,d1_,c1_, f1_,p2_,d2_, c2_,f2_}] :>(deltaFundCol[c1,c2](deltaFundFlav[f1,f2]((-I (vec[p1,0]+I muq)gamma[0,d1,d2])+( -I psdash[p1,d1,d2])(1+RQ[p1]/Sqrt[sps[p1,p1]])+  MQ[p1]deltaDirac[d1,d2]) )
)GQ[p1],

(*Vertices*)
\[CapitalGamma]qbqbqq[{p1_,d1_, c1_,f1_,p2_,d2_,c2_, f2_,p3_,d3_,c3_, f3_,p4_,d4_,c4_, f4_}]:>
Module[{i1,i2,i3,i4},( 
  lambda1 TBGetVertex[basisName,1,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda2 TBGetVertex[basisName,2,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda3 TBGetVertex[basisName,3,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda4 TBGetVertex[basisName,4,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda5 TBGetVertex[basisName,5,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda6 TBGetVertex[basisName,6,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda7 TBGetVertex[basisName,7,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda8 TBGetVertex[basisName,8,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda9 TBGetVertex[basisName,9,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda10 TBGetVertex[basisName,10,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]
)]
};


PreTraceFullFourQuarkRule={
\[CapitalGamma]qbqbqq[{p1_,d1_, c1_,f1_,p2_,d2_,c2_, f2_,p3_,d3_,c3_, f3_,p4_,d4_,c4_, f4_}]:>
Module[{i1,i2,i3,i4},( 
  lambda1 TBGetVertex[basisName,1,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]+
  lambda2 Hold[TBGetVertex[basisName,2,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda3 Hold[TBGetVertex[basisName,3,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda4 Hold[TBGetVertex[basisName,4,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda5 Hold[TBGetVertex[basisName,5,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda6 Hold[TBGetVertex[basisName,6,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda7 Hold[TBGetVertex[basisName,7,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda8 Hold[TBGetVertex[basisName,8,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda9 Hold[TBGetVertex[basisName,9,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]+
  lambda10 Hold[TBGetVertex[basisName,10,{p1,d1,c1,f1},{p3,d3,c3,f3},{p2,d2,c2,f2},{p4,d4,c4,f4}]]
)]
};


PostTraceRulesGluons:={
RA[p_]:> RB[k^2,sps[p,p]] ,
RAdot[p_]:>RBdot[k^2,sps[p,p]]- RB[k^2,sps[p,p]]etaA,
GA[p_]:>1/(vec[p,0]^2+sps[p,p]+ RA[p]),
ZAqbq1p[p1_,p2_]:>gAqbq1

}


(* ::Input::Initialization:: *)
PreTraceRulesGluons={
(*Regulators*)
RAA[{p1_,v1_,a1_,p2_,v2_,a2_}]:> TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}]RA[p2],

(*Regulator Derivatives*)
RdotAA[{p1_,v1_,a1_,p2_,v2_,a2_}]:>TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}]RAdot[p2] ,

(*Propagators*)
\[CapitalGamma]AA[{p1_,v1_,a1_,p2_,v2_,a2_}] :>  TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}](sp[p2,p2]),
GAA[{p1_,v1_,a1_,p2_,v2_,a2_}] :>TBGetVertex["AA",1,{p1,v1,a1},{p2,v2,a2}]GA[p1],

(*Quark-Gluon Vertices*)
\[CapitalGamma]Aqbq[{p1_,v1_,a1_,p2_,d2_,c2_,f2_,p3_,d3_,c3_,f3_}] :> ( 
      ZAqbq1p[p1,p2] TBGetVertex["AqbqDirect",1,{p1,v1,a1},{p2,d2,c2,f2},{p3,d3,c3,f3}]
)
};


(* ::Input::Initialization:: *)
PreTraceRules:=PreTraceRulesQuarks\[Union]PreTraceRulesGluons
PostTraceRules:=PostTraceRulesQuarks\[Union]PostTraceRulesGluons


(* ::Input::Initialization:: *)
fRGEq = {"Prefactor"->{1/2},
<|"type"->"Regulatordot", "indices"->{i,j}|>,
<|"type"->"Propagator", "indices"->{i,j}|>};
fields= <|"bosonic"-> {A[p,{v, c}]},
                 "fermionic"->{{qb[p,{d,c,f}],q[p,{d,c,f}]}}|>;
Truncation= {{q,qb},{A,A},(* propagators *)
{qb,q,A},(* gluon sector scatterings *)
{qb,qb,q,q}(* four-quark scatterings *)
};
SetupfRG= <|"MasterEquation"->fRGEq,
"FieldSpace"->fields,
"Truncation"->Truncation|>;
DiagramStyle={
A->Orange,q->Black
};


collectList:={
ZAqbq1[__],GQ[__],GA[__],gAqbq1
};
CustomSimplify[expr_]:=Module[{ret},
ret=expr//.{Nc->3,Nf->2,Sqrt[a_^2]:>a}//UseLorentzLinearity;
ret=Collect[ret,collectList,QuickSimplify];
ret=Collect[ret,collectList,Simplify];
Return[ret];
];
CustomQuickSimplify[expr_]:=Module[{ret},
ret=expr//.{Nc->3,Nf->2,Sqrt[a_^2]:>a}//UseLorentzLinearity;
ret=Collect[ret,collectList,QuickSimplify];
Return[ret];
];
SetStandardSimplify[CustomSimplify];
SetStandardQuickSimplify[CustomQuickSimplify];

splitDiagrams[expr_List,zeroOut_List]:=Module[{ex1},
ex1=expr;
ex1=ParallelMap[Expand[#[[1]]]&,ex1//.PreTraceFullFourQuarkRule//.zeroOut];
ex1=Flatten@ParallelMap[If[Head[#]===Plus,List@@#,#]&,ex1];
ex1=Map[{ReleaseHold[#]}&,ex1];
Return[ex1]
]
Finalize[expr_]:=SimplifyAllMomenta[q,ExpandScalarProductsFiniteT[expr]]


kernelLambda1=<|"Path"->"qbqqbq","Name"->"lambda1","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda2=<|"Path"->"qbqqbq","Name"->"lambda2","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda3=<|"Path"->"qbqqbq","Name"->"lambda3","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda4=<|"Path"->"qbqqbq","Name"->"lambda4","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda5=<|"Path"->"qbqqbq","Name"->"lambda5","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda6=<|"Path"->"qbqqbq","Name"->"lambda6","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda7=<|"Path"->"qbqqbq","Name"->"lambda7","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda8=<|"Path"->"qbqqbq","Name"->"lambda8","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda9=<|"Path"->"qbqqbq","Name"->"lambda9","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;
kernelLambda10=<|"Path"->"qbqqbq","Name"->"lambda10","Type"->"Quadratureq0","Angles"->0,"d"->4,"AD"->False,"ctype"->"double","Device"->"CPU"|>;

kernels={kernelLambda1,kernelLambda2,kernelLambda3,kernelLambda4,kernelLambda5,kernelLambda6,kernelLambda7,kernelLambda8,kernelLambda9,kernelLambda10};

kernelParameterList={
<|"Name"->"p0f","Type"->"Constant","AD"->False|>,
<|"Name"->"T","Type"->"Constant","AD"->False|>,
<|"Name"->"muq","Type"->"Constant","AD"->False|>,

<|"Name"->"gAqbq1","Type"->"Variable","AD"->False|>,
<|"Name"->"etaA","Type"->"Variable","AD"->False|>,
<|"Name"->"etaQ","Type"->"Variable","AD"->False|>,
<|"Name"->"MQ","Type"->"FunctionTex1D","AD"->False|>,

<|"Name"->"lambda1","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda2","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda3","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda4","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda5","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda6","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda7","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda8","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda9","Type"->"Variable","AD"->False|>,
<|"Name"->"lambda10","Type"->"Variable","AD"->False|>
};

MakeFlowClassFiniteT["FourFermi",kernels]

AddCodeOptimizeFunctions[RB[__],RF[__],RBdot[__],RFdot[__]]
ShowCodeOptimizeFunctions[]


(*Diagrams*)
DerivativeListqbqqbq={qb[-p,{d1,c1,f1}],q[p,{d2,c2,f2}],qb[-p,{d3,c3,f3}],q[p,{d4,c4,f4}]};
Diagramsqbqqbqsidx =AutoSaveRestore["Diagramsqbqqbq",DeriveFunctionalEquation[SetupfRG,DerivativeListqbqqbq,"OutputLevel"->"SuperindexDiagrams"]];
symmetries={
{{1,3},{2,4},Plus}
};
Diagramsqbqqbqsidx=ReduceIdenticalFlowDiagrams[Diagramsqbqqbqsidx,DerivativeListqbqqbq,symmetries];
Diagramsqbqqbq=SuperindexToFullDiagrams[Diagramsqbqqbqsidx,SetupfRG,DerivativeListqbqqbq];
Diagramsqbqqbq =RerouteFermionicMomenta[Diagramsqbqqbq,SetupfRG,DerivativeListqbqqbq];
Print["Diagramsqbqqbq got "<>ToString[Length[Diagramsqbqqbq]]<>" diagrams."]

Diagramsqbqqbq=splitDiagrams[Diagramsqbqqbq,{}];
Print["Diagramsqbqqbq got "<>ToString[Length[Diagramsqbqqbq]]<>" diagrams after splitting."]

PlotSuperindexDiagram[Diagramsqbqqbqsidx,SetupfRG,"EdgeStyle"->DiagramStyle]


preTraceRule={{PreambleFormRule,"Vector q,p;"}};
postTraceRule=MakeP0FormRule[q,{p},{p0f-I muq}];
p0Projection=#//.vec[p,0]:>p0f-I muq&;
SetDisentangle[False]


Projector\[Lambda]i=Table[0,10];
\[Lambda]iLoops=Table[0,{i,1,10}];

For[ididx=1,ididx<=10,ididx++,
Print["Working on Fierz Basis Element number "<> ToString[ididx]];
Projector\[Lambda]i[[ididx]]=(-TBGetProjector[basisName,ididx,{p,d2,c2,f2},{-p,d1,c1,f1},{p,d4,c4,f4},{-p,d3,c3,f3}]);
Projection\[Lambda]i=(Projector\[Lambda]i[[ididx]] Diagramsqbqqbq//.PreTraceRules//p0Projection)//.p->0;
TraceDiagrams[16,"lambda"<>ToString[ididx],Projection\[Lambda]i,preTraceRule,postTraceRule];

\[Lambda]iLoops[[ididx]]=SumDiagrams[16,"lambda"<>ToString[ididx],0,(#//.PostTraceRules//p0Projection//Finalize//CustomSimplify)&];

MakeKernelFiniteTq0[Symbol["kernelLambda"<>ToString[ididx]],kernelParameterList,Re[\[Lambda]iLoops[[ididx]]],2*etaQ*Symbol["lambda"<>ToString[ididx]]];
];
Clear[ididx];


\[Lambda]iLoops=Table[0,{i,1,10}];

For[i=1,i<=10,i++,Print["Working on Fierz Basis Element number "<> ToString[i]];
\[Lambda]iLoops[[i]]=Import["TraceBuffer/lambda"<>ToString[i]<>"/sum.m"];
MakeKernelFiniteTq0[Symbol["kernelLambda"<>ToString[i]],kernelParameterList,Re[\[Lambda]iLoops[[i]]],2*etaQ*Symbol["lambda"<>ToString[i]]];
];
Clear[i];



