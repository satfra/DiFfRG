#!/usr/bin/env wolframscript

scriptDir = DirectoryName[$InputFileName];
Get[FileNameJoin[{scriptDir, "AUMP.m"}]];

args = Select[$ScriptCommandLine[[2 ;;]], # =!= "--" &];

argValue[name_, default_: Missing["NotFound"]] := Module[{pos},
    pos = FirstPosition[args, name, Missing["NotFound"]];
    If[MissingQ[pos] || First[pos] >= Length[args], default, args[[First[pos] + 1]]]
];
argValues[name_] := Module[{positions},
    positions = Flatten[Position[args, name]];
    Select[positions, # < Length[args] &] /. i_Integer :> args[[i + 1]]
];

jsonOut[assoc_] := (WriteString[$Output, ExportString[assoc, "RawJSON"] <> "\n"];);

projectRoot = argValue["--project-root", Directory[]];
testFile = argValue["--test-file"];
testIndex = ToExpression[argValue["--test-index", "0"]];
sectionArg = argValue["--section-path", ""];
keepTemp = MemberQ[args, "--keep-temp"];
wlPaths = ExpandFileName /@ argValues["--wl-path"];
initFiles = ExpandFileName /@ argValues["--init"];

If[MissingQ[testFile] || ! FileExistsQ[testFile],
    jsonOut[<|"Status" -> "error", "Error" -> "missing test file", "TestFile" -> ToString[testFile]|>];
    Exit[2];
];

AUMPTempPath[prefix_String] := FileNameJoin[{
    $TemporaryDirectory,
    prefix <> ToString[$ProcessID] <> "-" <> ToString[Round[1000000 AbsoluteTime[]]]
}];

sectionPath = If[sectionArg === "", {}, StringSplit[sectionArg, "\t"]];
tempDir = CreateDirectory[AUMPTempPath["aump-"]];
outputFile = FileNameJoin[{tempDir, "stdout.txt"}];
messageFile = FileNameJoin[{tempDir, "messages.txt"}];
out = OpenWrite[outputFile];
msg = OpenWrite[messageFile];

result = Block[{$Path = DeleteDuplicates[Join[wlPaths, {projectRoot}, $Path]], $Output = {out}, $Messages = {msg}, Print},
    Print[expr___] := WriteString[$Output, StringJoin[ToString[#, OutputForm] & /@ {expr}] <> "\n"];
    AUMPResetRegistry[];
    Scan[Get, Select[initFiles, FileExistsQ]];
    Get[testFile];
    tests = AUMPRegisteredTests[];
    If[testIndex < 1 || testIndex > Length[tests],
        <|"Status" -> "error", "Error" -> "invalid test index", "TestFile" -> testFile, "TestIndex" -> testIndex|>,
        AUMPRunLeaf[tests[[testIndex]], sectionPath, <|"ProjectRoot" -> projectRoot, "TempDirectory" -> tempDir|>]
    ]
];

Close[out];
Close[msg];

capturedOutput = If[FileExistsQ[outputFile], Import[outputFile, "Text"], ""];
capturedMessages = If[FileExistsQ[messageFile], Import[messageFile, "Text"], ""];
result = Join[result, <|"Output" -> capturedOutput, "Messages" -> capturedMessages, "TempDirectory" -> tempDir|>];

If[! keepTemp && DirectoryQ[tempDir], DeleteDirectory[tempDir, DeleteContents -> True]];

jsonOut[result];
Exit[If[MemberQ[{"passed", "skipped"}, Lookup[result, "Status", "error"]], 0, 1]];
