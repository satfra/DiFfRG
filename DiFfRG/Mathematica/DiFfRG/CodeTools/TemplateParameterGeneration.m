BeginPackage["DiFfRG`CodeTools`TemplateParameterGeneration`"]

TemplateParameterGeneration::usage = "";
TemplateParameterGeneration::wrongArgs = "TemplateParameterGeneration expects an Association (and optionally a List of AD replacements), but got: `1`";

DeviceExecSpace::usage = "DeviceExecSpace[device] returns the C++ execution-space type for a kernel \"Device\" specification (\"TBB\", \"GPU\" or \"Threads\").";

Begin["`Private`"]

allowedDevices = {"TBB", "GPU", "Threads"};

(* "Threads" runs on the Kokkos host backend: kokkos.hh defines TBB_exec, GPU_exec and
   KokkosHost_exec; there is no Threads_exec type on the C++ side. *)
DeviceExecSpace[device_String] := "DiFfRG::" <> (device /. "Threads" -> "KokkosHost") <> "_exec";

appendKeyDimension::missingKey = "The key `1` is missing in the template parameters.";

appendKeyDimension[templateParameter_List, params_Association] :=
    Module[{},
    (* maybe rename "d" to "spatialDimension" *)
        If[KeyExistsQ[params,"d"],
            Append[templateParameter, ToString[params["d"]]],
            Message[appendKeyDimension::missingKey, "d"];
            Abort[]
        ]
    ]

appendKeyName::missingKey = "The key `1` is missing in the template parameters.";

appendKeyName[templateParameter_List, params_Association] :=
    Module[{},
        If[KeyExistsQ[params, "Name"],
            Append[templateParameter, ToString[params["Name"]] <> "_kernel<Regulator>"],
            Message[appendKeyName::missingKey, "Name"];
            Abort[]
        ]
    ]

appendKeyType::missingKey = "The key `1` is missing in the template parameters.";

appendKeyType[templateParameter_List, params_Association, {}] :=
    Module[{},
        If[KeyExistsQ[params, "ComputeType"],
            Append[templateParameter, ToString[params["ComputeType"]]],
            Append[templateParameter, "double"]
        ]
    ]

appendKeyType[templateParameter_List, params_Association, ADReplacements_] :=
    Module[{},
        If[KeyExistsQ[params, "ComputeType"],
            Append[templateParameter, ToString[params["ComputeType"]] /. ADReplacements],
            Append[templateParameter, "autodiff::real"]
        ]
    ]

appendKeyDevice::wrongDevice = "The Device `1` is not part of known devices `2`.";

appendKeyDevice[templateParameter_List, params_Association] :=
    Module[{},
        If[KeyExistsQ[params, "Device"],
            If[MemberQ[allowedDevices, params["Device"]],
                Append[templateParameter, DeviceExecSpace[ToString[params["Device"]]]],
                Message[appendKeyDevice::wrongDevice, params["Device"], allowedDevices];
                Abort[]
            ],
            Append[templateParameter, "DiFfRG::TBB_exec"]
        ]
    ]

TemplateParameterGeneration[params_Association] := TemplateParameterGeneration[params, {}];
TemplateParameterGeneration[params_Association, ADReplacements_] :=
    Module[
        {integratorTemplateParams = {}}
        ,
        integratorTemplateParams = appendKeyDimension[integratorTemplateParams, params];
        integratorTemplateParams = appendKeyType[integratorTemplateParams, params, ADReplacements];
        integratorTemplateParams = appendKeyName[integratorTemplateParams, params];
        integratorTemplateParams = appendKeyDevice[integratorTemplateParams, params];

        integratorTemplateParams
    ];

TemplateParameterGeneration[x___] := (Message[TemplateParameterGeneration::wrongArgs, {x}]; Abort[]);

End[]

EndPackage[]
