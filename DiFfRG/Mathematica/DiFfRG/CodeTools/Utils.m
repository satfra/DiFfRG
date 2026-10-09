(* ::Package:: *)

(* Exported symbols added here with SymbolName::usage *)

BeginPackage["DiFfRG`CodeTools`Utils`"]

appendDefaultAssociation::usage = "appendDefaultAssociation[config] appends default association to the given config.";
appendDefaultAssociation::wrongArgs = "appendDefaultAssociation expects a single Association argument, but got: `1`";

processParameters::usage = "processParameters[params, adReplacements] processes parameters for kernel generation.";
processParameters::wrongArgs = "processParameters expects a List of parameters and a List of AD replacements, but got: `1`";

Begin["`Private`"]

ClearAll[appendDefaultAssociation, processParameters]

appendDefaultAssociation[config_Association] :=
    Merge[{config, <|"Type" -> "double", "Reference" -> True, "Const" -> True|>}, First];

appendDefaultAssociation[x___] := (Message[appendDefaultAssociation::wrongArgs, {x}]; Abort[]);

processParameters[params_List, adReplacements_List] :=
    Module[{processedParams = params, paramsAD = {},i},
        For[i = 1, i <= Length[processedParams], i++,
            processedParams[[i]] = appendDefaultAssociation[processedParams[[i]]];
            paramsAD = Append[
                    paramsAD,
                    Module[{typeAD},
                        typeAD = If[processedParams[[i]]["AD"]===True,processedParams[[i]]["Type"] /. adReplacements,processedParams[[i]]["Type"]];
                        Merge[{processedParams[[i]], <|"Type" -> typeAD|>}, Last]
                    ]
                ];
        ];
        {processedParams, paramsAD}
    ];

processParameters[x___] := (Message[processParameters::wrongArgs, {x}]; Abort[]);

End[]

EndPackage[]
