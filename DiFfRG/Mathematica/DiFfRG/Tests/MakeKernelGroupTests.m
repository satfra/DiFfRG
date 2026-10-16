Needs["AUMP`"];
Needs["DiFfRG`CodeTools`MakeKernel`"];
Needs["DiFfRG`CodeTools`Directory`"];

(* MakeKernelGroup and the "SharedGrids" option: the generated files are checked as text here; that they
   compile and agree with separate kernels is checked against the C++ library (tests/physics/integration). *)

containsAll[text_String, patterns_List] :=
    AllTrue[patterns, StringContainsQ[text, #] &];

$groupParameters = {
    <|"Name" -> "k", "Type" -> "double", "Const" -> True|>,
    <|"Name" -> "ZA", "Type" -> "SplineInterpolator1D<double, LinearCoordinates1D<double>>", "Const" -> True|>,
    <|"Name" -> "ZB", "Type" -> "SplineInterpolator1D<double, LinearCoordinates1D<double>>", "Const" -> True|>
};

(* Generate into a fresh flow directory and return the files of flow `name` as name -> text. *)
generateFlow[name_String, make_] :=
    Module[{tmp, dir},
        tmp = FileNameJoin[{AUMPTestTempDirectory[], "generated"}];
        CreateDirectory[tmp];
        SetFlowDirectory[tmp <> "/"];
        dir = FileNameJoin[{tmp, "flows", name}];
        CreateDirectory[FileNameJoin[{dir, "src"}], CreateIntermediateDirectories -> True];
        make[];
        Association @ Map[FileNameTake[#] -> Import[#, "Text"]&, Select[FileNames["*", {dir, FileNameJoin[{dir, "src"}]}], !DirectoryQ[#]&]]
    ];

groupOf[opts___] :=
    MakeKernelGroup[
        {{ZA[p] * ZB[l1] / (l1^2 + k^2), ZA[p], "first"}, {ZB[p] * ZA[l1] * l1 / (l1^2 + k^2), "second"}},
        (* first, so that they override the defaults below: the first occurrence of an option wins *)
        opts,
        "Name" -> "grp", "Integrator" -> "DiFfRG::Integrator_p2", "d" -> 4, "Device" -> "GPU",
        "IntegrationVariables" -> {"l1"}, "Coordinates" -> {"LinearCoordinates1D<double>"}, "CoordinateArguments" -> {"p"},
        "Parameters" -> $groupParameters];

AUMPTestCase["MakeKernelGroup emits a MultiValue integrator", {"make-kernel", "group", "funkit"},
    Module[{files = generateFlow["grp", groupOf[]&]},
        AUMPCHECK[containsAll[files["grp.hh"], {
            "DiFfRG::Integrator_p2<4, DiFfRG::MultiValue<double, 2>, grp_kernel<Regulator>, DiFfRG::GPU_exec> integrator;",
            "map(const std::array<double*, 2>& dest, const LinearCoordinates1D<double>& coordinates",
            "void get(DiFfRG::MultiValue<double, 2>& dest",
            "output_names{{\"first\", \"second\"}}"}]];
        AUMPCHECK[containsAll[files["kernel.hh"], {"DiFfRG/common/multi_value.hh", "return DiFfRG::MultiValue<double, 2>("}]];
        (* kernel() and constant() both return the pair *)
        AUMPCHECKEqual[StringCount[files["kernel.hh"], "return DiFfRG::MultiValue<double, 2>("], 2];
        AUMPCHECK[StringContainsQ[files["CT_map_1.cc"], "return integrator.map(dest, coordinates, k, ZA, ZB);"]];
    ];
];

AUMPTestCase["MakeKernelGroup with a float ComputeType still hands back double", {"make-kernel", "group", "precision", "funkit"},
    Module[{files = generateFlow["grp", groupOf["ComputeType" -> "float"]&]},
        AUMPCHECK[containsAll[files["grp.hh"], {
            "DiFfRG::MultiValue<float, 2>, grp_kernel<Regulator>",
            "map(const std::array<double*, 2>& dest",
            "void get(DiFfRG::MultiValue<double, 2>& dest"}]];
        AUMPCHECK[StringContainsQ[files["kernel.hh"], "return DiFfRG::MultiValue<float, 2>("]];
    ];
];

AUMPTestCase["MakeKernelGroup map_points takes one PointSpan per output", {"make-kernel", "group", "map-points", "funkit"},
    Module[{files = generateFlow["grp", groupOf["MapPoints" -> True]&]},
        AUMPCHECK[StringContainsQ[files["grp.hh"], "map_points(const std::array<DiFfRG::PointSpan<double>, 2>& dest"]];
        AUMPCHECK[StringContainsQ[files["CT_map_points.cc"], "integrator.map_points(dest, "]];
    ];
];

AUMPTestCase["MakeKernelGroup rejects what it does not support", {"make-kernel", "group", "abort"},
    AUMPCHECKAbort[groupOf["AD" -> True]];
    AUMPCHECKAbort[groupOf["SplitKernel" -> True]];
    AUMPCHECKAbort[groupOf["Integrator" -> "DiFfRG::IntegratorLat"]];
    AUMPCHECKAbort[MakeKernelGroup[{{ZA[p], "only"}}, "Name" -> "grp"]];
    AUMPCHECKAbort[MakeKernelGroup[{{ZA[p], "same"}, {ZB[p], "same"}}, "Name" -> "grp"]];
];

AUMPTestCase["SharedGrids shares the coordinate transform and checks the grid", {"make-kernel", "shared-grids", "funkit"},
    Module[{files},
        files = generateFlow["sg", MakeKernel[ZA[l1] * ZB[l1] + ZA[p] / (l1^2 + k^2),
            "Name" -> "sg", "Integrator" -> "DiFfRG::Integrator_p2", "d" -> 4, "Device" -> "GPU",
            "IntegrationVariables" -> {"l1"}, "Coordinates" -> {"LinearCoordinates1D<double>"}, "CoordinateArguments" -> {"p"},
            "Parameters" -> $groupParameters, "SharedGrids" -> {{ZA, ZB}}]&];
        (* l1 is transformed once for ZA and ZB; p is looked up once and keeps the plain call *)
        AUMPCHECKEqual[StringCount[files["kernel.hh"], ".index(l1)"], 1];
        AUMPCHECK[containsAll[files["kernel.hh"], {"ZA.at(_gix1)", "ZB.at(_gix1)", "ZA(p)"}]];
        AUMPCHECK[containsAll[files["CT_map_1.cc"], {
            "#if !defined(NDEBUG) || defined(DIFFRG_CHECK_SHARED_GRIDS)",
            "DiFfRG::check_shared_grid(\"ZA, ZB\", ZA, ZB);"}]];
        AUMPCHECK[StringContainsQ[files["CT_get.cc"], "DiFfRG::check_shared_grid(\"ZA, ZB\", ZA, ZB);"]];
    ];
];

AUMPTestCase["Without SharedGrids nothing is shared or checked", {"make-kernel", "shared-grids", "funkit"},
    Module[{files},
        files = generateFlow["sg", MakeKernel[ZA[l1] * ZB[l1],
            "Name" -> "sg", "Integrator" -> "DiFfRG::Integrator_p2", "d" -> 4,
            "IntegrationVariables" -> {"l1"}, "Coordinates" -> {"LinearCoordinates1D<double>"}, "CoordinateArguments" -> {"p"},
            "Parameters" -> $groupParameters]&];
        AUMPCHECK[StringFreeQ[files["kernel.hh"], "_gix"]];
        AUMPCHECK[StringFreeQ[files["CT_map_1.cc"], "check_shared_grid"]];
    ];
];
