Needs["AUMP`"];
Needs["DiFfRG`CodeTools`Directory`"];

AUMPTestCase["flowDir uses the default flow name", {"directory"},
    AUMPCHECKEqual[
        DiFfRG`CodeTools`Directory`flowDir,
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/flows"
    ];
];

AUMPTestCase["SetFlowName updates flowDir", {"directory"},
    DiFfRG`CodeTools`Directory`SetFlowName["myNewFlows"];
    AUMPCHECKEqual[
        DiFfRG`CodeTools`Directory`flowDir,
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/myNewFlows"
    ];
];

AUMPTestCase["SetFlowName rejects a non-String argument", {"directory", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Directory`SetFlowName[123]];
];

AUMPTestCase["SetFlowDirectory rejects a non-String argument", {"directory", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Directory`SetFlowDirectory[123]];
];

AUMPTestCase["SetFlowName restores the default flow name", {"directory"},
    DiFfRG`CodeTools`Directory`SetFlowName["flows"];
    AUMPCHECKEqual[
        DiFfRG`CodeTools`Directory`flowDir,
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/flows"
    ];
];
