(* CodeToolsTests.m — Tests for CodeTools.m functions *)

Needs["AUMP`"];
Needs["DiFfRG`CodeTools`"];

loadFunKitForTest[] := (
    AUMPAssume[Length @ PacletFind["FunKit"] > 0, "FunKit is not installed"];
    Block[{Print}, Get["FunKit`"]];
);

AUMPTestCase["SafeFiniteTFunctions rewrites Tanh[x/(2 T)]", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Tanh[x/(2 T)], T], TanhFiniteT[x, T]];
];

AUMPTestCase["SafeFiniteTFunctions rewrites Tanh[x/T]", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Tanh[x/T], T], TanhFiniteT[x, 2 T]];
];

AUMPTestCase["SafeFiniteTFunctions rewrites Coth[x/(2 T)]", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Coth[x/(2 T)], T], CothFiniteT[x, T]];
];

AUMPTestCase["SafeFiniteTFunctions rewrites Coth[x/T]", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Coth[x/T], T], CothFiniteT[x, 2 T]];
];

AUMPTestCase["SafeFiniteTFunctions rewrites Sech[x/(2 T)]", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Sech[x/(2 T)], T], SechFiniteT[x, T]];
];

AUMPTestCase["SafeFiniteTFunctions rewrites Csch[x/T]", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Csch[x/T], T], CschFiniteT[x, 2 T]];
];

AUMPTestCase["SafeFiniteTFunctions rewrites powers", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[Tanh[x/(2 T)]^2, T], TanhFiniteT[x, T]^2];
];

AUMPTestCase["SafeFiniteTFunctions leaves unrelated expressions unchanged", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[SafeFiniteTFunctions[x^2 + y, T], x^2 + y];
];

AUMPTestCase["SafeFiniteTFunctions rewrites compound expressions", {"codetools", "finite-temperature"},
    AUMPCHECKEqual[
        SafeFiniteTFunctions[Tanh[a/(2 T)] + Coth[b/T], T],
        TanhFiniteT[a, T] + CothFiniteT[b, 2 T]
    ];
];

AUMPTestCase["DeclareSymmetricPoints4DP4 returns a C++ declaration", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints4DP4[q, p, {p1, p2, p3, p4}]},
            StringQ[result] && StringContainsQ[result, "const double"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints4DP4 declares all angles", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints4DP4[q, p, {p1, p2, p3, p4}]},
            StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp2q"] &&
                StringContainsQ[result, "cosp3q"] && StringContainsQ[result, "cosp4q"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints4DP3 returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints4DP3[q, p, {p1, p2, p3}]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp2q"] &&
                StringContainsQ[result, "cosp3q"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints3DP4 returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints3DP4[q, p, {p1, p2, p3, p4}]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp4q"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints3DP3 returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints3DP3[q, p, {p1, p2, p3}]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp3q"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints2DP4 returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints2DP4[q, p, {p1, p2, p3, p4}]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp4q"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints2DP3 returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints2DP3[q, p, {p1, p2, p3}]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp3q"]
        ]
    ];
];

AUMPTestCase["DeclareAnglesP34Dpqr returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareAnglesP34Dpqr[q, p, r]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cospq"] && StringContainsQ[result, "cosqr"]
        ]
    ];
];

AUMPTestCase["DeclareAnglesP33Dpqr returns valid declarations", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareAnglesP33Dpqr[q, p, r]},
            StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cospq"] && StringContainsQ[result, "cosqr"]
        ]
    ];
];

AUMPTestCase["DeclareSymmetricPoints2DP3 accepts a custom compute type", {"codetools", "funkit"},
    loadFunKitForTest[];
    AUMPCHECK[
        Module[{result = DeclareSymmetricPoints2DP3[q, p, {p1, p2, p3}, Symbol@"cos1", "float"]},
            StringQ[result] && StringContainsQ[result, "const float"]
        ]
    ];
];
