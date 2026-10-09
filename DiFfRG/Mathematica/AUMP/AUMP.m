BeginPackage["AUMP`"];

AUMPTestCase::usage = "AUMPTestCase[name, tags, body] registers a held test case.";
AUMPSection::usage = "AUMPSection[name, body] defines a Catch2-style section inside a test case.";
AUMPCHECK::usage = "AUMPCHECK[expr] records a non-fatal assertion failure unless expr is True.";
AUMPREQUIRE::usage = "AUMPREQUIRE[expr] records a fatal assertion failure unless expr is True.";
AUMPCHECKEqual::usage = "AUMPCHECKEqual[actual, expected] records a non-fatal SameQ assertion.";
AUMPREQUIREEqual::usage = "AUMPREQUIREEqual[actual, expected] records a fatal SameQ assertion.";
AUMPSKIP::usage = "AUMPSKIP[reason] skips the current test leaf.";
AUMPAssume::usage = "AUMPAssume[condition, reason] skips the current test leaf unless condition is True.";
AUMPCHECKAbort::usage = "AUMPCHECKAbort[expr] records a non-fatal assertion failure unless expr aborts.";
AUMPREQUIREAbort::usage = "AUMPREQUIREAbort[expr] records a fatal assertion failure unless expr aborts.";
AUMPCHECKMessage::usage = "AUMPCHECKMessage[expr, message] records a non-fatal assertion failure unless expr emits message.";
AUMPCHECKNoMessages::usage = "AUMPCHECKNoMessages[expr] records a non-fatal assertion failure unless expr emits no messages.";
AUMPCHECKEquivalent::usage = "AUMPCHECKEquivalent[actual, expected] records a non-fatal assertion failure unless actual and expected satisfy SameTest.";
AUMPCHECKSimplifiesToZero::usage = "AUMPCHECKSimplifiesToZero[expr] records a non-fatal assertion failure unless FullSimplify[expr] is 0.";
AUMPCHECKStringContainsAll::usage = "AUMPCHECKStringContainsAll[string, needles] records a non-fatal assertion failure unless all needles occur in string.";
AUMPCHECKStringEqualNormalized::usage = "AUMPCHECKStringEqualNormalized[actual, expected] compares strings after whitespace normalization.";
AUMPCHECKFileContent::usage = "AUMPCHECKFileContent[file, expected] compares a file's text content to expected.";
AUMPWithTempDirectory::usage = "AUMPWithTempDirectory[body] evaluates body with a fresh temporary directory and deletes it afterwards.";
AUMPProjectRoot::usage = "AUMPProjectRoot[] returns the project root for the current test leaf.";
AUMPTestFile::usage = "AUMPTestFile[] returns the file containing the current test case.";
AUMPTestTempDirectory::usage = "AUMPTestTempDirectory[] returns the private temp directory for the current test leaf.";
AUMPResetRegistry::usage = "AUMPResetRegistry[] clears registered test cases.";
AUMPRegisteredTests::usage = "AUMPRegisteredTests[] returns registered test case metadata.";
AUMPDiscoverTestFiles::usage = "AUMPDiscoverTestFiles[paths] returns Wolfram test files under paths.";
AUMPDiscoverLeaves::usage = "AUMPDiscoverLeaves[paths] loads tests and returns executable test leaves.";
AUMPRunLeaf::usage = "AUMPRunLeaf[test, sectionPath, context] runs one test leaf and returns an association.";

Begin["`Private`"];

SetAttributes[AUMPTestCase, HoldRest];
SetAttributes[AUMPSection, HoldRest];
SetAttributes[AUMPCHECK, HoldAll];
SetAttributes[AUMPREQUIRE, HoldAll];
SetAttributes[AUMPCHECKEqual, HoldAll];
SetAttributes[AUMPREQUIREEqual, HoldAll];
SetAttributes[AUMPSKIP, HoldAll];
SetAttributes[AUMPAssume, HoldAll];
SetAttributes[AUMPCHECKAbort, HoldAll];
SetAttributes[AUMPREQUIREAbort, HoldAll];
SetAttributes[AUMPCHECKMessage, HoldAll];
SetAttributes[AUMPCHECKNoMessages, HoldAll];
SetAttributes[AUMPCHECKEquivalent, HoldAll];
SetAttributes[AUMPCHECKSimplifiesToZero, HoldAll];
SetAttributes[AUMPCHECKStringContainsAll, HoldAll];
SetAttributes[AUMPCHECKStringEqualNormalized, HoldAll];
SetAttributes[AUMPCHECKFileContent, HoldAll];
SetAttributes[AUMPWithTempDirectory, HoldAll];
SetAttributes[AUMPCollectSections, HoldRest];

If[! ValueQ[$AUMPTestRegistry], $AUMPTestRegistry = {}];

$AUMPCurrentResult = <|"Assertions" -> 0, "Failures" -> {}, "Skipped" -> False, "SkipReason" -> Null|>;
$AUMPCurrentContext = <||>;
$AUMPSelectedSectionPath = {};
$AUMPSectionStack = {};
$AUMPSectionMatched = False;
$AUMPRequireFailureTag = Unique["AUMPRequireFailure"];
$AUMPSkipTag = Unique["AUMPSkip"];

Options[AUMPCHECKEquivalent] = {SameTest -> SameQ};

AUMPResetRegistry[] := ($AUMPTestRegistry = {});

AUMPNormalizeTags[tags_] := Which[
    tags === {}, {},
    ListQ[tags], ToString /@ tags,
    True, {ToString[tags]}
];

AUMPTestCase[name_String, body_] := AUMPTestCase[name, {}, body];
AUMPTestCase[name_String, tags_, body_] := Module[{file},
    file = If[StringQ[$InputFileName] && $InputFileName =!= "", $InputFileName, Missing["Unknown"]];
    AppendTo[$AUMPTestRegistry, <|
        "Name" -> name,
        "Tags" -> AUMPNormalizeTags[tags],
        "Body" -> HoldComplete[body],
        "File" -> file
    |>];
];

AUMPRegisteredTests[] := $AUMPTestRegistry;

AUMPFilesForDirectory[path_String, All] := FileNames["*.m" | "*.wl" | "*.wls", path, Infinity];
AUMPFilesForDirectory[path_String, pattern_String] := FileNames[pattern, path, Infinity];

AUMPDiscoverTestFiles[paths_List, pattern_: All] := Module[{expanded},
    expanded = Flatten[paths /. p_String :> If[DirectoryQ[p],
        AUMPFilesForDirectory[p, pattern],
        {p}
    ]];
    Sort @ DeleteDuplicates @ Select[expanded, FileExistsQ]
];
AUMPDiscoverTestFiles[path_String, pattern_: All] := AUMPDiscoverTestFiles[{path}, pattern];

AUMPDiscoverLeaves[paths_List, pattern_: All] := Module[{files, tests},
    AUMPResetRegistry[];
    files = AUMPDiscoverTestFiles[paths, pattern];
    Scan[Get, files];
    tests = AUMPAddTestIndices[AUMPRegisteredTests[]];
    Flatten[AUMPLeavesForTest /@ tests, 1]
];
AUMPDiscoverLeaves[path_String, pattern_: All] := AUMPDiscoverLeaves[{path}, pattern];

AUMPAddTestIndices[tests_List] := Module[{fileCounts = <||>},
    MapIndexed[
        Function[{test, index},
            fileCounts[test["File"]] = Lookup[fileCounts, test["File"], 0] + 1;
            Join[test, <|"Index" -> First[index], "FileIndex" -> fileCounts[test["File"]]|>]
        ],
        tests
    ]
];

AUMPLeavesForTest[test_Association] := Module[{paths},
    paths = AUMPSectionPaths[test["Body"]];
    If[paths === {},
        {AUMPLeaf[test, {}]},
        AUMPLeaf[test, #] & /@ paths
    ]
];

AUMPLeaf[test_Association, sectionPath_List] := <|
    "Name" -> test["Name"],
    "Tags" -> test["Tags"],
    "File" -> test["File"],
    "Index" -> test["Index"],
    "FileIndex" -> test["FileIndex"],
    "SectionPath" -> sectionPath,
    "DisplayName" -> If[sectionPath === {}, test["Name"], test["Name"] <> " / " <> StringRiffle[sectionPath, " / "]]
|>;

AUMPSectionPaths[HoldComplete[body_]] := Module[{paths = {}},
    AUMPCollectSections[HoldComplete[body], {}, paths];
    DeleteDuplicates[paths]
];

AUMPCollectSections[held_, prefix_, paths_Symbol] := Cases[
    held,
    HoldPattern[AUMPSection[name_String, sectionBody_]] :> (
        AppendTo[paths, Append[prefix, name]];
        AUMPCollectSections[HoldComplete[sectionBody], Append[prefix, name], paths]
    ),
    Infinity
];

AUMPProjectRoot[] := Lookup[$AUMPCurrentContext, "ProjectRoot", Directory[]];
AUMPTestFile[] := Lookup[$AUMPCurrentContext, "TestFile", Missing["NotRunning"]];
AUMPTestTempDirectory[] := Lookup[$AUMPCurrentContext, "TempDirectory", $TemporaryDirectory];

AUMPRecordFailure[kind_String, held_, actual_, expected_: Missing["NotApplicable"]] := Module[{failure},
    failure = <|
        "Kind" -> kind,
        "Expression" -> ToString[Unevaluated[held], InputForm],
        "Actual" -> ToString[actual, InputForm],
        "Expected" -> If[MissingQ[expected], Null, ToString[expected, InputForm]]
    |>;
    $AUMPCurrentResult = AssociateTo[$AUMPCurrentResult,
        "Failures" -> Append[Lookup[$AUMPCurrentResult, "Failures", {}], failure]
    ];
];

AUMPIncrementAssertions[] := ($AUMPCurrentResult = AssociateTo[$AUMPCurrentResult,
    "Assertions" -> Lookup[$AUMPCurrentResult, "Assertions", 0] + 1
]);

AUMPReasonString[reason_] := Module[{value},
    value = Quiet[Check[reason, Unevaluated[reason]]];
    If[StringQ[value], value, ToString[Unevaluated[reason], InputForm]]
];

AUMPMarkSkipped[reason_] := ($AUMPCurrentResult = AssociateTo[$AUMPCurrentResult, <|
    "Skipped" -> True,
    "SkipReason" -> AUMPReasonString[reason]
|>]);

AUMPWhitespaceNormalize[value_] := StringTrim[StringReplace[ToString[value], WhitespaceCharacter.. -> " "]];

AUMPCHECK[expr_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[Check[expr, $Failed]];
    If[! TrueQ[value], AUMPRecordFailure["CHECK", HoldForm[expr], value]];
    value
];

AUMPREQUIRE[expr_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[Check[expr, $Failed]];
    If[! TrueQ[value],
        AUMPRecordFailure["REQUIRE", HoldForm[expr], value];
        Throw[$AUMPRequireFailureTag, $AUMPRequireFailureTag]
    ];
    value
];

AUMPCHECKEqual[actual_, expected_] := Module[{a, e},
    AUMPIncrementAssertions[];
    a = Quiet[Check[actual, $Failed]];
    e = Quiet[Check[expected, $Failed]];
    If[! SameQ[a, e], AUMPRecordFailure["CHECKEqual", HoldForm[actual], a, e]];
    a
];

AUMPREQUIREEqual[actual_, expected_] := Module[{a, e},
    AUMPIncrementAssertions[];
    a = Quiet[Check[actual, $Failed]];
    e = Quiet[Check[expected, $Failed]];
    If[! SameQ[a, e],
        AUMPRecordFailure["REQUIREEqual", HoldForm[actual], a, e];
        Throw[$AUMPRequireFailureTag, $AUMPRequireFailureTag]
    ];
    a
];

AUMPSKIP[reason_: "skipped"] := (
    AUMPMarkSkipped[reason];
    Throw[AUMPReasonString[reason], $AUMPSkipTag]
);

AUMPAssume[condition_, reason_: "assumption failed"] := Module[{value},
    value = Quiet[Check[condition, False]];
    If[! TrueQ[value], AUMPSKIP[reason]];
    value
];

AUMPCHECKAbort[expr_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[CheckAbort[expr; "no-abort", "aborted"]];
    If[value =!= "aborted", AUMPRecordFailure["CHECKAbort", HoldForm[expr], value, "aborted"]];
    value
];

AUMPREQUIREAbort[expr_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[CheckAbort[expr; "no-abort", "aborted"]];
    If[value =!= "aborted",
        AUMPRecordFailure["REQUIREAbort", HoldForm[expr], value, "aborted"];
        Throw[$AUMPRequireFailureTag, $AUMPRequireFailureTag]
    ];
    value
];

AUMPCHECKMessage[expr_, message_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[Check[expr; "no-message", "message", message]];
    If[value =!= "message", AUMPRecordFailure["CHECKMessage", HoldForm[expr], value, ToString[Unevaluated[message], InputForm]]];
    value
];

AUMPCHECKNoMessages[expr_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[Check[expr; True, False]];
    If[! TrueQ[value], AUMPRecordFailure["CHECKNoMessages", HoldForm[expr], "message emitted", "no messages"]];
    value
];

AUMPCHECKEquivalent[actual_, expected_, OptionsPattern[]] := Module[{a, e, sameTest, ok},
    AUMPIncrementAssertions[];
    a = Quiet[Check[actual, $Failed]];
    e = Quiet[Check[expected, $Failed]];
    sameTest = OptionValue[SameTest];
    ok = Quiet[Check[sameTest[a, e], False]];
    If[! TrueQ[ok], AUMPRecordFailure["CHECKEquivalent", HoldForm[actual], a, e]];
    a
];

AUMPCHECKSimplifiesToZero[expr_] := Module[{value},
    AUMPIncrementAssertions[];
    value = Quiet[Check[FullSimplify[expr], $Failed]];
    If[! TrueQ[value === 0], AUMPRecordFailure["CHECKSimplifiesToZero", HoldForm[expr], value, 0]];
    value
];

AUMPCHECKStringContainsAll[string_, needles_] := Module[{s, ns, missing},
    AUMPIncrementAssertions[];
    s = Quiet[Check[ToString[string], $Failed]];
    ns = Quiet[Check[ToString /@ Flatten[{needles}], $Failed]];
    missing = If[StringQ[s] && ListQ[ns], Select[ns, ! StringContainsQ[s, #] &], ns];
    If[missing =!= {}, AUMPRecordFailure["CHECKStringContainsAll", HoldForm[string], missing, "all needles present"]];
    missing === {}
];

AUMPCHECKStringEqualNormalized[actual_, expected_] := Module[{a, e},
    AUMPIncrementAssertions[];
    a = Quiet[Check[AUMPWhitespaceNormalize[actual], $Failed]];
    e = Quiet[Check[AUMPWhitespaceNormalize[expected], $Failed]];
    If[! SameQ[a, e], AUMPRecordFailure["CHECKStringEqualNormalized", HoldForm[actual], a, e]];
    a
];

AUMPCHECKFileContent[file_, expected_] := Module[{path, content},
    AUMPIncrementAssertions[];
    path = Quiet[Check[file, $Failed]];
    content = If[StringQ[path] && FileExistsQ[path], Import[path, "Text"], $Failed];
    If[! SameQ[content, expected], AUMPRecordFailure["CHECKFileContent", HoldForm[file], content, expected]];
    content
];

AUMPWithTempDirectory[body_] := Module[{dir, result},
    dir = CreateDirectory[FileNameJoin[{$TemporaryDirectory, "aump-" <> ToString[$ProcessID] <> "-" <> ToString[Round[1000000 AbsoluteTime[]]]}]];
    result = Block[{$AUMPCurrentContext = Join[$AUMPCurrentContext, <|"TempDirectory" -> dir|>]},
        CheckAbort[body, If[DirectoryQ[dir], DeleteDirectory[dir, DeleteContents -> True]]; Abort[]]
    ];
    If[DirectoryQ[dir], DeleteDirectory[dir, DeleteContents -> True]];
    result
];

AUMPSection[name_String, body_] := Module[{path},
    path = Append[$AUMPSectionStack, name];
    If[Length[$AUMPSelectedSectionPath] >= Length[path] && Take[$AUMPSelectedSectionPath, Length[path]] === path,
        Block[{$AUMPSectionStack = path},
            If[path === $AUMPSelectedSectionPath, $AUMPSectionMatched = True];
            body
        ],
        Null
    ]
];

AUMPRunLeaf[test_Association, sectionPath_List, context_Association : <||>] := Module[
    {start, result},
    start = AbsoluteTime[];
    result = Block[
        {
            $AUMPCurrentResult = <|"Assertions" -> 0, "Failures" -> {}, "Skipped" -> False, "SkipReason" -> Null|>,
            $AUMPCurrentContext = Join[context, <|"TestFile" -> Lookup[test, "File", Missing["Unknown"]]|>],
            $AUMPSelectedSectionPath = sectionPath,
            $AUMPSectionStack = {},
            $AUMPSectionMatched = sectionPath === {}
        },
        Module[{thrown, failures, status, sectionMatched, skipped, skipReason},
        thrown = Catch[
            <|"Tag" -> None, "Value" -> ReleaseHold[test["Body"]]|>,
            _,
            Function[{value, tag}, <|"Tag" -> tag, "Value" -> value|>]
        ];
        sectionMatched = $AUMPSectionMatched;
        failures = Lookup[$AUMPCurrentResult, "Failures", {}];
        skipped = TrueQ[Lookup[$AUMPCurrentResult, "Skipped", False]] || Lookup[thrown, "Tag", None] === $AUMPSkipTag;
        skipReason = If[skipped, Lookup[$AUMPCurrentResult, "SkipReason", Lookup[thrown, "Value", "skipped"]], Null];
        status = Which[
            skipped, "skipped",
            Lookup[thrown, "Tag", None] === $AUMPRequireFailureTag, "failed",
            sectionPath =!= {} && ! TrueQ[sectionMatched], "error",
            failures === {}, "passed",
            True, "failed"
        ];
        <|
            "Name" -> test["Name"],
            "Tags" -> Lookup[test, "Tags", {}],
            "File" -> Lookup[test, "File", Missing["Unknown"]],
            "SectionPath" -> sectionPath,
            "DisplayName" -> If[sectionPath === {}, test["Name"], test["Name"] <> " / " <> StringRiffle[sectionPath, " / "]],
            "Status" -> status,
            "Assertions" -> Lookup[$AUMPCurrentResult, "Assertions", 0],
            "SkipReason" -> skipReason,
            "Failures" -> If[status === "error" && failures === {},
                {<|"Kind" -> "Section", "Expression" -> StringRiffle[sectionPath, " / "], "Actual" -> "not reached", "Expected" -> "reached"|>},
                failures
            ],
            "Duration" -> N[AbsoluteTime[] - start]
        |>
        ]
    ];
    result
];

End[];

EndPackage[];
