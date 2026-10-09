Needs["AUMP`"];
Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"];

AUMPTestCase["TemplateParameterGeneration supports GPU float kernels", {"template-parameters"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[<|"d" -> 2, "Name" -> "MyKernel", "ComputeType" -> "float", "Device" -> "GPU"|>],
        {"2", "float", "MyKernel_kernel<Regulator>", "DiFfRG::GPU_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration supports TBB double kernels", {"template-parameters"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[<|"d" -> 3, "Name" -> "Integrator", "ComputeType" -> "double", "Device" -> "TBB"|>],
        {"3", "double", "Integrator_kernel<Regulator>", "DiFfRG::TBB_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration applies real AD replacements", {"template-parameters", "ad"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[
            <|"d" -> 3, "Name" -> "Integrator", "ComputeType" -> "double", "Device" -> "TBB"|>,
            {"double" -> "autodiff::real"}
        ],
        {"3", "autodiff::real", "Integrator_kernel<Regulator>", "DiFfRG::TBB_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration applies second-order complex AD replacements", {"template-parameters", "ad"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[
            <|"d" -> 3, "Name" -> "pion", "ComputeType" -> "DiFfRG::complex<double>", "Device" -> "TBB"|>,
            {"DiFfRG::complex<double>" -> "cxReal<2, double>"}
        ],
        {"3", "cxReal<2, double>", "pion_kernel<Regulator>", "DiFfRG::TBB_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration supports Threads execution", {"template-parameters"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[<|"d" -> 1, "Name" -> "Test", "ComputeType" -> "float", "Device" -> "Threads"|>],
        {"1", "float", "Test_kernel<Regulator>", "DiFfRG::KokkosHost_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration defaults ComputeType to double", {"template-parameters", "defaults"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[<|"d" -> 4, "Name" -> "DefaultType", "Device" -> "TBB"|>],
        {"4", "double", "DefaultType_kernel<Regulator>", "DiFfRG::TBB_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration defaults Device to TBB", {"template-parameters", "defaults"},
    AUMPCHECKEqual[
        TemplateParameterGeneration[<|"d" -> 2, "Name" -> "DefaultDevice", "ComputeType" -> "double"|>],
        {"2", "double", "DefaultDevice_kernel<Regulator>", "DiFfRG::TBB_exec"}
    ];
];

AUMPTestCase["TemplateParameterGeneration requires d", {"template-parameters", "abort"},
    AUMPCHECKAbort[TemplateParameterGeneration[<|"Name" -> "X", "Device" -> "TBB"|>]];
];

AUMPTestCase["TemplateParameterGeneration requires Name", {"template-parameters", "abort"},
    AUMPCHECKAbort[TemplateParameterGeneration[<|"d" -> 2, "Device" -> "TBB"|>]];
];

AUMPTestCase["TemplateParameterGeneration rejects an invalid Device", {"template-parameters", "abort"},
    AUMPCHECKAbort[TemplateParameterGeneration[<|"d" -> 2, "Name" -> "X", "Device" -> "BadDevice"|>]];
];

AUMPTestCase["TemplateParameterGeneration requires an Association", {"template-parameters", "abort"},
    AUMPCHECKAbort[TemplateParameterGeneration["not an association"]];
];
