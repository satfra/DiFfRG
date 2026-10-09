(* ::Package:: *)

(* ::Title:: *)
(*Tutorial 6: O(N) model at finite temperature, batched*)


(* ::Text:: *)
(*Flow of the effective potential V(rho) in d=3+1 with 3d (spatial) regulators. Unlike Tutorial 3, the Matsubara sum is left to the finite-temperature integrator, which sums it numerically, and the kernels get map_points() for batched evaluation.*)
(*Run headless with:  wolfram -script ON.wl*)


(* ::Chapter:: *)
(*Truncation setup*)


(* ::Text:: *)
(*First thing is to import the package. Although unnecessary here, we set the working directory to be where the script resides and inform FormTracer of a few variables.*)


(* ::Input::Initialization:: *)
Get["DiFfRG`"]
SetDirectory[If[$InputFileName==="",Directory[],DirectoryName[$InputFileName]]];


(* ::Text:: *)
(*Then, we define SO(N-1), which is the remaining symmetry after SO(N) has been broken.*)


(* ::Input::Initialization:: *)
DefineGroupTensors[{{SONfund,{SONm1,N-1},deltaadjSONm1[a,b],SONm1F[a,b,c],deltaFundSONm1[a,b],SONm1T[a,l,j],adjEpsSONm1[a,b,c],epsSONm1[a,b,c]}}]


(* ::Chapter:: *)
(*Feynman rules*)


(* ::Text:: *)
(*We define in the following all Feynman rules needed to expand the QMeS expressions we obtain.*)
(*The next block is just a little helper to automatically get any n-vertex from the effective potential V((sigma^2+pi^2)/2) = V(rho).*)


(* ::Input::Initialization:: *)
FunctionalD[expr_,v:(f_[_]|{f_[_],_Integer}).., OptionsPattern[]]:=Internal`InheritedBlock[{f},
	f/:D[f[x_],f[y_],NonConstants->{f}]:=DiracDelta[x-y];
	f/:D[f,f[y_],NonConstants->{f}]:=DiracDelta[#-y]&;
	D[expr,v,NonConstants->{f}]
];

\[Lambda]m\[CapitalPi]n\[Sigma][pions_,sigmas_]:=Module[{action,sigmaDeriv},
	action=Taylor[V,((sigma^2-sigma0^2)+Module[{b},\[CapitalPi][b]^2])/2&,Length[pions]+sigmas];
	sigmaDeriv=D[action,{sigma,sigmas}];
	Simplify[
		If[Length[pions]>=1,
			FunctionalD[sigmaDeriv,(\[CapitalPi][#]&)/@pions/.List->Sequence]
				//.DiracDelta[x_-y_]->deltaMesonFlav[x,y]
				//.DiracDelta[-1+y_]->deltaMesonFlav[1,y],
			sigmaDeriv
		]//.{\[CapitalPi][a_]->0,sigma0->sigma,sigma->Sqrt[2 rhoPhi],
			Derivative[n_][V][a_]:>Symbol["d"<>ToString[n]<>"V"],V[a_]->V,d1V->m\[CapitalPi]^2}
	]
];


(* ::Text:: *)
(*Propagators, regulator insertions and the meson vertices. Only the two-point sector enters the potential flow, the higher vertices are listed for completeness.*)


(* ::Input::Initialization:: *)
FeynmanRules={
	(*regulator insertions*)
	Rdot[{\[CapitalPi],\[CapitalPi]},{{p1_,{a1_}},{p2_,{a2_}}}]:>deltaFundSONm1[a1,a2] RBdot[k^2,sps[p2,p2]],
	Rdot[{\[Sigma],\[Sigma]},{{p1_},{p2_}}]:>RBdot[k^2,sps[p2,p2]],
	(*propagators*)
	Propagator[{\[CapitalPi],\[CapitalPi]},{{p1_,{a1_}},{p2_,{a2_}}}]:>deltaFundSONm1[a1,a2] 1/(sp[p1,p1]+m2Pi+RB[k^2,sps[p1,p1]]),
	Propagator[{\[Sigma],\[Sigma]},{{p1_},{p2_}}]:>1/(sp[p1,p1]+m2Sigma+RB[k^2,sps[p1,p1]]),
	(*vertices*)
	\[CapitalGamma]\[Sigma]\[Sigma]\[Sigma][{p1_,p2_,p3_}]:>\[Lambda]m\[CapitalPi]n\[Sigma][{},3],
	\[CapitalGamma]\[CapitalPi]\[CapitalPi]\[Sigma][{p1_,a1_,p2_,a2_,p3_}]:>\[Lambda]m\[CapitalPi]n\[Sigma][{a1,a2},1],
	\[CapitalGamma]\[CapitalPi]\[CapitalPi]\[Sigma]\[Sigma][{p1_,a1_,p2_,a2_,p3_,p4_}]:>\[Lambda]m\[CapitalPi]n\[Sigma][{a1,a2},2],
	\[CapitalGamma]\[CapitalPi]\[CapitalPi]\[CapitalPi]\[CapitalPi][{p1_,a1_,p2_,a2_,p3_,a3_,p4_,a4_}]:>\[Lambda]m\[CapitalPi]n\[Sigma][{a1,a2,a3,a4},0],
	\[CapitalGamma]\[Sigma]\[Sigma]\[Sigma]\[Sigma][{p1_,p2_,p3_,p4_}]:>\[Lambda]m\[CapitalPi]n\[Sigma][{},4]
};


(* ::Chapter:: *)
(*fRG setup and truncation*)


(* ::Text:: *)
(*The fRG flow equation and the truncation for the scalar sector. We put in some higher-order scatterings, but these do not couple into the flow equation, as we only flow the potential.*)


(* ::Input::Initialization:: *)
fields=<|"Commuting"->{\[Sigma][p],\[CapitalPi][p,{a}]}|>;
truncation=<|
	GammaN->{{\[Sigma],\[Sigma]},{\[CapitalPi],\[CapitalPi]},{\[Sigma],\[Sigma],\[Sigma]},{\[Sigma],\[CapitalPi],\[CapitalPi]},{\[Sigma],\[Sigma],\[Sigma],\[Sigma]},{\[Sigma],\[Sigma],\[CapitalPi],\[CapitalPi]},{\[CapitalPi],\[CapitalPi],\[CapitalPi],\[CapitalPi]}},
	Propagator->{{\[Sigma],\[Sigma]},{\[CapitalPi],\[CapitalPi]}},
	Rdot->{{\[Sigma],\[Sigma]},{\[CapitalPi],\[CapitalPi]}}
|>;
Setup=<|"FieldSpace"->fields,"Truncation"->truncation|>;
FSetGlobalSetup[Setup];


(* ::Chapter:: *)
(*Potential flow*)


(* ::Input::Initialization:: *)
flowV=FPrint[FRoute[FTruncate[WetterichEquation]]];
flowV=flowV/.FeynmanRules;
flowV=FormTrace[flowV["1-Loop"]["Expression"]];
flowV=SimplifyAllMomenta[l1,#]&[ExpandScalarProductsFiniteT[flowV]];
flowV=FullSimplify/@flowV;


(* ::Text:: *)
(*The Kurganov-Tadmor assembler wants the flux split: the sigma loop is the only piece that sees d(u)/d(rho) and becomes the diffusive part, the pion loop depends on the field alone and becomes the advective part.*)
(*The pion loop carries the SO(N-1) multiplicity N-1 from the trace; we divide it out here because the KT models multiply by (N-1) themselves.*)


(* ::Input::Initialization:: *)
flowVSigma=Total@Select[flowV,!FreeQ[#,m2Sigma]&]//FullSimplify;
flowVPion=FullSimplify[Total@Select[flowV,FreeQ[#,m2Sigma]&]/(N-1)];


(* ::Chapter:: *)
(*Code generation*)


(* ::Text:: *)
(*The kernels depend on the spatial loop momentum l1 and the Matsubara frequency l10; Integrator_fT_p2 integrates over l1 and sums over l10. "MapPoints"->True adds map_points(), which evaluates a kernel at many points in one call. "MatsubaraEven"->True checks that the summand is even in l10 and lets the integrator evaluate it once per mode. Each kernel is generated for the CPU (TBB) and for the GPU.*)
(*V is the full flux for CG; V_pion and V_sigma are its two halves for KT (advection and diffusion). As in Tutorial 3, the pion loop is divided by its multiplicity N-1.*)


(* ::Input::Initialization:: *)
kernelParameterList={
	<|"Name"->"k"|>,
	<|"Name"->"N"|>,
	<|"Name"->"T"|>,
	<|"Name"->"m2Pi","AD"->True|>,
	<|"Name"->"m2Sigma","AD"->True|>
};

MakeV[expr_,name_,parameters_,device_]:=MakeKernel[expr,
	"Name"->name,
	"Integrator"->"Integrator_fT_p2",
	"d"->4,
	"AD"->True,
	"Device"->device,
	"MapPoints"->True,
	"MatsubaraEven"->True,
	"Parameters"->parameters,
	"IntegrationVariables"->{"l1","l10"}
];

SetFlowName["flows"];
MakeV[flowV,"V",kernelParameterList,"TBB"];
MakeV[flowV,"V_GPU",kernelParameterList,"GPU"];
MakeV[flowVPion,"V_pion",kernelParameterList[[{1,2,3,4}]],"TBB"];
MakeV[flowVPion,"V_pion_GPU",kernelParameterList[[{1,2,3,4}]],"GPU"];
MakeV[flowVSigma,"V_sigma",kernelParameterList[[{1,2,3,5}]],"TBB"];
MakeV[flowVSigma,"V_sigma_GPU",kernelParameterList[[{1,2,3,5}]],"GPU"];
UpdateFlows["ONFlows"]
