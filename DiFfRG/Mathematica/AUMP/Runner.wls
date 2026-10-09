#!/usr/bin/env wolframscript

scriptDir = DirectoryName[$InputFileName];
Get[FileNameJoin[{scriptDir, "AUMP.m"}]];

args = Select[$ScriptCommandLine[[2 ;;]], # =!= "--" &];

hasArg[name_] := MemberQ[args, name];
argValue[name_, default_: Missing["NotFound"]] := Module[{pos},
    pos = FirstPosition[args, name, Missing["NotFound"]];
    If[MissingQ[pos] || First[pos] >= Length[args], default, args[[First[pos] + 1]]]
];
argValues[name_] := Module[{positions},
    positions = Flatten[Position[args, name]];
    Select[positions, # < Length[args] &] /. i_Integer :> args[[i + 1]]
];

usage[] := (
    Print["usage: wolframscript -script AUMP/Runner.wls --path Tests [--tag tag] [--exclude-tag tag] [--reporter spec|json|junit] [--timeout seconds] [--keep-failed] [--wl-path dir] [--init file] [--pattern glob] [--name text] [--list-tests] [--output file]"];
    Exit[2]
);

If[hasArg["--help"], usage[]];

paths = argValues["--path"];
If[paths === {}, paths = {"Tests"}];
projectRoot = ExpandFileName[argValue["--project-root", Directory[]]];
includeTags = argValues["--tag"];
excludeTags = argValues["--exclude-tag"];
reporter = argValue["--reporter", "spec"];
timeout = ToExpression[argValue["--timeout", "30"]];
keepFailed = hasArg["--keep-failed"];
wlPaths = ExpandFileName /@ argValues["--wl-path"];
initFiles = ExpandFileName /@ argValues["--init"];
pattern = argValue["--pattern", All];
nameFilter = argValue["--name", ""];
listTests = hasArg["--list-tests"];
outputPath = argValue["--output", ""];

AUMPTempPath[prefix_String, suffix_String : ""] := FileNameJoin[{
    $TemporaryDirectory,
    prefix <> ToString[$ProcessID] <> "-" <> ToString[Round[1000000 AbsoluteTime[]]] <> suffix
}];

captureFile = AUMPTempPath["aump-discovery-", ".txt"];
capture = OpenWrite[captureFile];
leaves = Check[
    Block[{$Path = DeleteDuplicates[Join[wlPaths, {projectRoot}, $Path]], $Output = {capture}, $Messages = {capture}},
        Scan[Get, Select[initFiles, FileExistsQ]];
        AUMPDiscoverLeaves[ExpandFileName /@ paths, pattern]
    ],
    $Failed
];
Close[capture];
discoveryOutput = If[FileExistsQ[captureFile], Import[captureFile, "Text"], ""];
If[FileExistsQ[captureFile], DeleteFile[captureFile]];

If[leaves === $Failed,
    Print["AUMP discovery failed."];
    If[StringLength[StringTrim[discoveryOutput]] > 0, Print[StringTrim[discoveryOutput]]];
    Exit[2];
];

tagAllowed[leaf_] := Module[{tags = Lookup[leaf, "Tags", {}]},
    (includeTags === {} || Intersection[includeTags, tags] =!= {}) &&
    (excludeTags === {} || Intersection[excludeTags, tags] === {})
];

nameAllowed[leaf_] := nameFilter === "" || StringContainsQ[Lookup[leaf, "DisplayName", ""], nameFilter, IgnoreCase -> True];

selectedLeaves = Select[leaves, tagAllowed[#] && nameAllowed[#] &];

If[listTests,
    Scan[Print[Lookup[#, "DisplayName", Lookup[#, "Name", "unknown"]]] &, selectedLeaves];
    Exit[0];
];

wolframscriptCommand[] := Module[{entitlement = Environment["WOLFRAMSCRIPT_ENTITLEMENTID"]},
    If[StringQ[entitlement] && entitlement =!= "",
        {"wolframscript", "-entitlement", entitlement},
        {"wolframscript"}
    ]
];

runWorker[leaf_] := Module[{cmd, raw, result, status},
    cmd = Join[
        wolframscriptCommand[],
        {"-script", FileNameJoin[{scriptDir, "Worker.wls"}],
        "--project-root", projectRoot,
        "--test-file", leaf["File"],
        "--test-index", ToString[leaf["FileIndex"]]}
    ];
    If[Lookup[leaf, "SectionPath", {}] =!= {}, cmd = Join[cmd, {"--section-path", StringRiffle[leaf["SectionPath"], "\t"]}]];
    Scan[(cmd = Join[cmd, {"--wl-path", #}]) &, wlPaths];
    Scan[(cmd = Join[cmd, {"--init", #}]) &, initFiles];
    If[keepFailed, cmd = Append[cmd, "--keep-temp"]];
    raw = TimeConstrained[
        RunProcess[cmd, All],
        timeout,
        <|"ExitCode" -> 124, "StandardOutput" -> "", "StandardError" -> "timeout"|>
    ];
    If[raw["ExitCode"] === 124,
        <|"Status" -> "timeout", "DisplayName" -> leaf["DisplayName"], "File" -> leaf["File"], "SectionPath" -> leaf["SectionPath"], "Assertions" -> 0, "Failures" -> {}, "Output" -> "", "Messages" -> raw["StandardError"]|>,
        result = Quiet[Check[ImportString[raw["StandardOutput"], "RawJSON"], $Failed]];
        If[result === $Failed || ! AssociationQ[result],
            <|"Status" -> "error", "DisplayName" -> leaf["DisplayName"], "File" -> leaf["File"], "SectionPath" -> leaf["SectionPath"], "Assertions" -> 0, "Failures" -> {<|"Kind" -> "Worker", "Expression" -> "RawJSON", "Actual" -> raw["StandardOutput"], "Expected" -> "valid JSON"|>}, "Output" -> raw["StandardOutput"], "Messages" -> raw["StandardError"]|>,
            result
        ]
    ]
];

results = runWorker /@ selectedLeaves;

statusCounts = Counts[Lookup[results, "Status", "error"]];
passed = Lookup[statusCounts, "passed", 0];
skipped = Lookup[statusCounts, "skipped", 0];
failed = Length[results] - passed - skipped;
assertions = Total[Lookup[results, "Assertions", 0]];
summary = <|"Total" -> Length[results], "Passed" -> passed, "Failed" -> failed, "Skipped" -> skipped, "Assertions" -> assertions|>;

escapeXml[value_] := StringReplace[ToString[value], {"&" -> "&amp;", "<" -> "&lt;", ">" -> "&gt;", "\"" -> "&quot;", "'" -> "&apos;"}];

renderJUnit[tests_, sum_] := StringJoin[
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
    "<testsuite name=\"AUMP\" tests=\"", ToString[sum["Total"]], "\" failures=\"", ToString[sum["Failed"]], "\" skipped=\"", ToString[sum["Skipped"]], "\">\n",
    StringJoin @@ Map[
        Function[result,
            Module[{status = Lookup[result, "Status", "error"], name = escapeXml[Lookup[result, "DisplayName", Lookup[result, "Name", "unknown"]]], failures},
                failures = Lookup[result, "Failures", {}];
                StringJoin[
                    "  <testcase name=\"", name, "\" time=\"", ToString[NumberForm[Lookup[result, "Duration", 0.], {Infinity, 6}]], "\">",
                    Which[
                        status === "skipped",
                            "<skipped message=\"" <> escapeXml[Lookup[result, "SkipReason", "skipped"]] <> "\" />",
                        status =!= "passed",
                            StringJoin @@ Map["<failure message=\"" <> escapeXml[Lookup[#, "Kind", "Failure"]] <> "\">" <> escapeXml[Lookup[#, "Expression", ""]] <> "\nactual: " <> escapeXml[Lookup[#, "Actual", ""]] <> "\nexpected: " <> escapeXml[Lookup[#, "Expected", ""]] <> "</failure>" &, failures],
                        True,
                            ""
                    ],
                    "</testcase>\n"
                ]
            ]
        ],
        tests
    ],
    "</testsuite>\n"
];

renderSpec[tests_, sum_] := Module[{lines = {}},
    Scan[
        Function[result,
            Module[{status = Lookup[result, "Status", "error"], prefix},
                prefix = Switch[status, "passed", "[pass] ", "skipped", "[skip] ", _, "[fail] "];
                AppendTo[lines, prefix <> Lookup[result, "DisplayName", Lookup[result, "Name", "unknown"]]];
                If[status === "skipped",
                    AppendTo[lines, "  reason: " <> ToString[Lookup[result, "SkipReason", "skipped"]]]
                ];
                If[status =!= "passed" && status =!= "skipped",
                    Scan[
                        Function[failure,
                            AppendTo[lines, "  " <> Lookup[failure, "Kind", "Failure"] <> ": " <> Lookup[failure, "Expression", ""]];
                            AppendTo[lines, "    actual:   " <> ToString[Lookup[failure, "Actual", ""]]];
                            AppendTo[lines, "    expected: " <> ToString[Lookup[failure, "Expected", ""]]]
                        ],
                        Lookup[result, "Failures", {}]
                    ];
                    If[StringLength[Lookup[result, "Messages", ""]] > 0, AppendTo[lines, "  messages: " <> StringTrim[Lookup[result, "Messages", ""]]]]
                ]
            ]
        ],
        tests
    ];
    StringRiffle[Join[lines, {"", "AUMP: " <> ToString[sum["Passed"]] <> " passed, " <> ToString[sum["Failed"]] <> " failed, " <> ToString[sum["Skipped"]] <> " skipped, " <> ToString[sum["Assertions"]] <> " assertions."}], "\n"] <> "\n"
];

report = Switch[
    reporter,
    "json", ExportString[<|"Tests" -> results, "Summary" -> summary|>, "RawJSON"] <> "\n",
    "junit", renderJUnit[results, summary],
    _, renderSpec[results, summary]
];

If[outputPath =!= "",
    If[Export[outputPath, report, "Text"] === $Failed,
        WriteString[$Output, "AUMP: could not write report to " <> outputPath <> "\n"];
        Exit[2]
    ],
    WriteString[$Output, report]
];

Exit[If[failed === 0, 0, 1]];
