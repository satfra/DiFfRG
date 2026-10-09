(* PhysicsTests.m — Tests for DiFfRG.m physics functions *)

Needs["AUMP`"];

loadDiFfRGForPhysicsTest[] := (
    AUMPAssume[Length @ PacletFind["FunKit"] > 0, "FunKit is not installed"];
    Block[{Print}, Get["DiFfRG`"]];
);

AUMPTestCase["GetDirectory returns a String", {"physics", "funkit"},
    loadDiFfRGForPhysicsTest[];
    AUMPCHECK[StringQ[GetDirectory[]]];
];

AUMPTestCase["QuickSimplify cancels equal terms", {"physics", "funkit"},
    loadDiFfRGForPhysicsTest[];
    AUMPCHECKEqual[QuickSimplify[x^2 - x^2], 0];
];

AUMPTestCase["QuickSimplify reduces a linear expression", {"physics", "funkit"},
    loadDiFfRGForPhysicsTest[];
    AUMPCHECKEqual[QuickSimplify[a + b - a], b];
];

AUMPTestCase["MatsubaraSum reproduces the bosonic thermal sum", {"physics", "funkit", "matsubara"},
    loadDiFfRGForPhysicsTest[];
    AUMPCHECK[
        FullSimplify[MatsubaraSum[1/(p0^2 + w^2), p0, T] - Coth[w/(2 T)]/(2 w)] === 0
    ];
];

AUMPTestCase["FermionMatsubaraSum reproduces the fermionic thermal sum", {"physics", "funkit", "matsubara"},
    loadDiFfRGForPhysicsTest[];
    AUMPCHECK[
        FullSimplify[FermionMatsubaraSum[1/(p0^2 + w^2), p0, T] - Tanh[w/(2 T)]/(2 w)] === 0
    ];
];
