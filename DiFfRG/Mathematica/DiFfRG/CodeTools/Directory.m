(* ::Package:: *)

BeginPackage["DiFfRG`CodeTools`Directory`"];

Unprotect["DiFfRG`CodeTools`Directory`*"];

ClearAll["DiFfRG`CodeTools`Directory`*"];

ClearAll["DiFfRG`CodeTools`Directory`Private`*"];

SetFlowName::usage = "SetFlowName[name_String] sets the name of the flow directory.";

SetFlowName::wrongArgs = "SetFlowName expects a single String argument, but got: `1`";

ShowFlowDirectory::usage = "ShowFlowDirectory[] shows the current flow directory.";

SetFlowDirectory::usage = "SetFlowDirectory[dir]
Set the current flow directory, i.e. where all generated files are saved. Default is ./flows/";

SetFlowDirectory::createFailed = "Failed to create directory: `1`";

SetFlowDirectory::wrongArgs = "SetFlowDirectory expects a single String argument, but got: `1`";

flowDir::usage = "flowDir is the current flow directory.";

Begin["`Private`"];

flowName = "flows";

flowDir :=
    FileNameJoin[{
        If[$Notebooks,
            NotebookDirectory[]
            ,
            Directory[]
        ]
        ,
        flowName
    }]

SetFlowName[name_String] :=
    flowName = name;

SetFlowName[x___] :=
    (
        Message[SetFlowName::wrongArgs, {x}];
        Abort[]
    );

SetFlowDirectory[name_String] :=
    Module[{dir},
        dir = CreateDirectory[name <> flowName <> "/"];
        If[dir =!= $Failed,
            flowDir := name <> flowName <> "/";
            ,
            Message[SetFlowDirectory::createFailed, name <> flowName <> "/"];
            Abort[]
        ];
    ];

SetFlowDirectory[x___] :=
    (
        Message[SetFlowDirectory::wrongArgs, {x}];
        Abort[]
    );

ShowFlowDirectory[] :=
    TemplateApply[StringTemplate["DiFfRG Flow output directory is set to `1`\n\tThis can be modified by using SetFlowName[\"YourNewName\"]"], {flowDir}];

End[];

EndPackage[];
