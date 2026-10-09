BeginPackage["DiFfRG`CodeTools`Regulator`"];

Unprotect["DiFfRG`CodeTools`Regulator`*"];

ClearAll["DiFfRG`CodeTools`Regulator`*"];

ClearAll["DiFfRG`CodeTools`Regulator`Private`*"];

getRegulator::usage = "getRegulator[name, {optName, optDef}] generates a string for a C++ type alias 'Regulator'. \
'name' is the name of the regulator class. 'optName' and 'optDef' specify the template parameter and its definition; use {\"\", \"\"} for non-templated regulators.";
getRegulator::wrongArgs = "getRegulator expects a String name and a list {String, String}, but got: `1`";


Begin["`Private`"];

getRegulator[name_String, {optName_String, optDef_String}] :=
  Module[{returnValue},
    returnValue = StringTemplate["`1`\nusing Regulator = `2`<`3`>;"][If[optName =!= "", optDef, ""], name, optName];

    returnValue
  ];

getRegulator[x___] := (Message[getRegulator::wrongArgs, {x}]; Abort[]);

End[];

EndPackage[];
