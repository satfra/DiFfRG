Needs["AUMP`"];
Needs["DiFfRG`CodeTools`Regulator`"];

AUMPTestCase["getRegulator generates standard names", {"regulator"},
    AUMPCHECKEqual[
        DiFfRG`CodeTools`Regulator`getRegulator["MyRegulator", {"MyOptions", "default"}],
        "default\nusing Regulator = MyRegulator<MyOptions>;"
    ];
];

AUMPTestCase["getRegulator generates alternate names", {"regulator"},
    AUMPCHECKEqual[
        DiFfRG`CodeTools`Regulator`getRegulator["AnotherReg", {"OtherOptions", "default"}],
        "default\nusing Regulator = AnotherReg<OtherOptions>;"
    ];
];

AUMPTestCase["getRegulator rejects a non-String regulator name", {"regulator", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Regulator`getRegulator[123, {"a", "b"}]];
];

AUMPTestCase["getRegulator rejects a non-String options name", {"regulator", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Regulator`getRegulator["x", {1, "b"}]];
];

AUMPTestCase["getRegulator supports empty template options", {"regulator"},
    AUMPCHECKEqual[
        DiFfRG`CodeTools`Regulator`getRegulator["MyReg", {"", ""}],
        "\nusing Regulator = MyReg<>;"
    ];
];
