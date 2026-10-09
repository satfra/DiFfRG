Needs["AUMP`"];
Needs["DiFfRG`CodeTools`Export`"];

AUMPTestCase["ExportCode exports a file", {"export", "filesystem"},
    Module[{fileName, content},
        fileName = FileNameJoin[{AUMPTestTempDirectory[], "test.txt"}];
        content = "test content";
        DiFfRG`CodeTools`Export`ExportCode[fileName, content];
        AUMPCHECKFileContent[fileName, content];
    ];
];

AUMPTestCase["ExportCode rejects a non-String file name", {"export", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Export`ExportCode[123, "x"]];
];

AUMPTestCase["ExportCode rejects non-String content", {"export", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Export`ExportCode["x", 123]];
];

AUMPTestCase["ExportCode rejects missing arguments", {"export", "abort"},
    AUMPCHECKAbort[DiFfRG`CodeTools`Export`ExportCode[]];
];
