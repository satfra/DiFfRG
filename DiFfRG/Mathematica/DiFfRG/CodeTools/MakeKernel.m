(* ::Package:: *)

(* Exported symbols added here with SymbolName::usage *)

BeginPackage["DiFfRG`CodeTools`MakeKernel`"]

GetStandardKernelDefinitions::usage = "GetStandardKernelDefinitions[] returns a list of standard kernel definitions used in DiFfRG."

MakeKernel::usage = "MakeKernel[kernel_Association, parameterList_List,integrandFlow_,constantFlow_:0., integrandDefinitions_String:\"\", constantDefinitions_String:\"\"]
Make a kernel from a given flow equation, parmeter list and kernel. The kernel must be a valid specification of an integration kernel.
This Function creates an integrator that evaluates (constantFlow + \[Integral]integrandFlow). One can prepend additional c++ definitions to the flow equation by using the integrandDefinitions and constantDefinitions parameters.
These are prepended to the respective methods of the integration kernel, allowing one to e.g. define specific angles one needs for the flow code.

The options \"KernelReturnTransform\" and \"ConstantReturnTransform\" (default Identity) accept a Mathematica function applied to the optimized expression before code generation, which lets you wrap the return value, e.g. \"KernelReturnTransform\" -> Re renders the kernel return as real(...).
The option \"ComputeType\" (default \"double\") is the value type the kernel is evaluated and integrated in: \"double\", \"float\", \"DiFfRG::complex<double>\" or \"DiFfRG::complex<float>\". With a float type the kernel, its literals and the integrator run in single precision, while map()/get() still write double results.
The option \"MapPoints\" (default False) additionally emits map_points(dest, args...), which evaluates the integral at dest.size() points in one launch (dest is a DiFfRG::PointSpan); every argument is a DiFfRG::PointArg, i.e. a single value or one value per point (a DiFfRG::PointSpan or std::vector). A model's evaluate_batch calls this. Supported by the vacuum and finite-temperature integrators, not by the lattice ones.
The lattice integrator \"Integrator\" -> \"DiFfRG::IntegratorLat\" with \"d\" -> 1..4 sums over the momenta of a periodic lattice; its \"IntegrationVariables\" are the momentum components {q0, ..., q(d-1)}, and the kernel must be even in every spatial component (and in q0 if q0_symmetric is set). It reads /integration/lattice/{N_t, N_s, a_t, a_s} and the optional q0_symmetric (default false) from the parameter file; d = 1 reads only N_t and a_t.
The option \"KernelTraits\" declares integrator traits on the emitted kernel class, as a list of names or name -> Boolean rules, e.g. \"KernelTraits\" -> {\"matsubara_finite_extent\"} or {\"matsubara_split\" -> True}. Each becomes a `static constexpr bool <name> = <value>;` member. \"MatsubaraEven\" stays a separate option because it carries a symbolic evenness check that a generic mechanism cannot.";

MakeKernel::Invalid = "The given arguments are invalid. See MakeKernel::usage";

MakeKernel::InvalidSpec = "The given kernel specification is invalid.";

MakeKernel::InvalidSpec = "The parameters given to MakeKernel are not valid.";

MakeKernel::MissingKey = "The key \"`1`\" is missing.";

MakeKernel::MissingType = "Parameter \"`1`\" has no Type specified, defaulting to double.";

MakeKernel::InvalidKey = "The key \"`1`\" is invalid: `2`";

MakeKernel::exportFailed = "Export of sources.m to `1` failed.";

MakeKernel::noMapPoints = "\"MapPoints\" -> True is not available for the lattice integrator `1`: it has no map_points().";

MakeKernel::notEven = "MatsubaraEven requested for kernel \"`1`\" but it is not even in \"`2`\"; emitting the standard kernel (the integrator keeps the explicit kernel(+f0)+kernel(-f0) form). This is expected when the loop contains fermionic dressings evaluated at f0-shifted arguments.";

MakeKernel::ctypedeprecated = "The option \"ctype\" is deprecated; use \"ComputeType\" -> `1` instead.";

MakeKernel::InvalidTrait = "KernelTraits entry `1` is not a C++ identifier optionally followed by -> True|False.";

Begin["`Private`"]

Needs["DiFfRG`CodeTools`Utils`"]

Needs["DiFfRG`CodeTools`Directory`"]

Needs["DiFfRG`CodeTools`Export`"]

Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"]

Needs["DiFfRG`CodeTools`Regulator`"]

(* "float" is the integrator value type of a single-precision kernel ("ComputeType" -> "float"); its AD
   integrators differentiate in single precision as well. *)
$ADReplacements = {"double" -> "autodiff::real", "DiFfRG::complex<double>" -> "cxreal", "float" -> "autodiff::Real<1, float>"};

$AD2Replacements = {"double" -> "autodiff::Real<2, double>", "DiFfRG::complex<double>" -> "cxReal<2, double>", "float" -> "autodiff::Real<2, float>"};

$ADSpecializations = {<|"Suffix" -> "AD", "Replacements" -> $ADReplacements|>, <|"Suffix" -> "AD2", "Replacements" -> $AD2Replacements|>};

$PredefRegFunc = {"RB", "RF", "RBdot", "RFdot", "dq2RB", "dq2RF"};

$StandardKernelDefinitions = Map[FunKit`MakeCppFunction["Name" -> #, "Body" -> "return Regulator::" <> # <> "(k2, p2);", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Suffix" -> "", "Parameters" -> {"k2", "p2"}]&, $PredefRegFunc];

CheckKey[kernel_Association, name_String, test_, msg_String] :=
    Module[{valid},
        If[Not @ KeyExistsQ[kernel, name],
            Message[MakeKernel::MissingKey, name];
            Return[False]
        ];
        If[Not @ test[kernel[name]],
            Message[MakeKernel::InvalidKey, name, msg];
            Return[False]
        ];
        Return[True];
    ];

KernelSpecQ[spec_Association] :=
    Module[{validKeys, validKeyTypes},
        validKeys = CheckKey[spec, "Name", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"] && CheckKey[spec, "Integrator", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"] && CheckKey[spec, "d", IntegerQ[#] && # >= 0&, "Must be an Integer >= 0"] && CheckKey[spec, "AD", BooleanQ, "Must be a Boolean"] && CheckKey[spec, "Device", MemberQ[{"Threads", "TBB", "GPU"}, #]&, "Must be Threads, TBB or GPU."] && CheckKey[spec, "Type", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"] && CheckKey[spec, "ComputeType", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"];
        Return[validKeys];
    ];

GetStandardKernelDefinitions[] :=
    $StandardKernelDefinitions

(* "ctype" is the old name of "ComputeType"; an explicit ctype still wins so old scripts keep working. *)
resolveComputeType[spec_Association] :=
    Module[{s = spec},
        If[s["ctype"] =!= Automatic,
            Message[MakeKernel::ctypedeprecated, ToString[s["ctype"], InputForm]];
            s["ComputeType"] = s["ctype"]
        ];
        KeyDrop[s, "ctype"]
    ];

singlePrecisionQ[spec_Association] :=
    StringContainsQ[spec["ComputeType"], "float"];

(* Real type of the kernel's arguments. *)
realType[spec_Association] :=
    If[singlePrecisionQ[spec], "float", "double"];

(* map()/get() hand results back in double even when the integrator runs in float, so the
   caller's double state vectors need no staging. *)
resultType[spec_Association] :=
    StringReplace[spec["ComputeType"], "float" -> "double"];

(* Print the kernel's literals as float (0.5f, complex<float>) in single precision; a double
   literal would silently promote the whole expression back to double. *)
SetAttributes[withComputePrecision, HoldRest];

withComputePrecision[spec_Association, body_] :=
    Block[{FunKit`Private`$codePrecision = If[singlePrecisionQ[spec], "single", FunKit`Private`$codePrecision]},
        body
    ];

(* Emit arbitrary integrator traits as class members. The traits are what the finite-T
   integrator branches on (matsubara_even, matsubara_finite_extent, matsubara_split), and a
   `requires { requires K::trait; }` detector reads FALSE for anything it cannot see -- a
   substitution failure, not an error. So a mistyped name is silently a disabled trait and a
   wrong right-hand side with no diagnostic; validate the name here instead. *)
kernelTraitMembers[traits_] :=
    Module[{list, entries},
        list =
            If[ListQ[traits],
                traits
                ,
                {traits}
            ];
        entries =
            Map[
                Function[t,
                    Module[{name, value},
                        {name, value} =
                            Switch[t,
                                _String, {t, True},
                                _Rule, {First[t], Last[t]},
                                _, {None, None}
                            ];
                        If[Not[StringQ[name]] || Not[StringMatchQ[name, RegularExpression["[A-Za-z_][A-Za-z0-9_]*"]]] || Not[BooleanQ[value]],
                            Message[MakeKernel::InvalidTrait, t];
                            Abort[]
                        ];
                        "static constexpr bool " <> name <> " = " <> If[TrueQ[value], "true", "false"] <> ";"
                    ]
                ]
                ,
                list
            ];
        DeleteDuplicates[entries]
    ];

(* Internal functions added here with Internal`*::usage *)

(* Parameters of a map_points() method: the destination array, then every argument as a PointArg,
   so the caller decides per call which arguments vary between points. *)
mapPointsParameters[returnType_String, args_List] :=
    Join[
        {<|"Name" -> "dest", "Type" -> "DiFfRG::PointSpan<" <> returnType <> ">", "Const" -> True, "Reference" -> False|>}
        ,
        Map[Merge[{#, <|"Type" -> "DiFfRG::PointArg<" <> #["Type"] <> ">", "Reference" -> True, "Const" -> True|>}, Last]&, args]
    ];

Options[MakeKernel] = {"Coordinates" -> {}, "CoordinateArguments" -> {}, "IntegrationVariables" -> {}, "KernelDefinitions" -> $StandardKernelDefinitions, "Regulator" -> "DiFfRG::PolynomialExpRegulator", "RegulatorOpts" -> {"", ""}, "KernelBody" -> "", "KernelReturnType" -> "auto", "KernelReturnTransform" -> Identity, "ConstantBody" -> "", "ConstantReturnType" -> "auto", "ConstantReturnTransform" -> Identity, "Parameters" -> {}, "Name" -> "", "d" -> -1, "Integrator" -> "", "AD" -> False, "ComputeType" -> "double", "ctype" -> Automatic, "Device" -> "TBB", "Type" -> "double", "SplitKernel" -> False, "SeparateLookups" -> False, "Decorator" -> "static KOKKOS_FUNCTION", "MatsubaraEven" -> False, "KernelTraits" -> {}, "MapPoints" -> False};

MakeKernel[__] :=
    (
        Message[MakeKernel::Invalid];
        Abort[]
    );

MakeKernel[kernelExpr_, OptionsPattern[]] :=
    MakeKernel @@ (Join[{kernelExpr, 0}, Thread[Rule @@ {#, OptionValue[MakeKernel, #]}]& @ Keys[Options[MakeKernel]]]);

MakeKernel[kernelExpr_, constExpr_, OptionsPattern[]] :=
    Module[{expr, const, exec, kernel, constant, kernelClass, kernelHeader, integratorHeader, integratorCpp, integratorTemplateParams, tparams = <|"Name" -> "...t", "Type" -> "auto&&", "Reference" -> False, "Const" -> False|>, kernelDefs = OptionValue["KernelDefinitions"], coordinates = OptionValue["Coordinates"], getArgs = OptionValue["CoordinateArguments"], intVariables = OptionValue["IntegrationVariables"], preArguments, regulator, params, adSpecs, explParamAD, arguments, outputPath, sources, returnType, returnTypePointer, spec, parameters, parametersKernel, matsubaraEvenTrait, kernelTraits},
        spec = Association @@ Thread[Rule @@ {#, OptionValue[MakeKernel, #]}]& @ Keys[Options[MakeKernel]];
        spec = resolveComputeType[spec];
        If[Not @ KernelSpecQ[spec],
            Message[MakeKernel::InvalidSpec];
            Abort[]
        ];
        If[TrueQ[spec["MapPoints"]] && StringContainsQ[spec["Integrator"], "IntegratorLat"],
            Message[MakeKernel::noMapPoints, spec["Integrator"]];
            Abort[]
        ];
        expr = kernelExpr;
        While[ListQ[expr], expr = Plus @@ expr];
(* Matsubara evenness: the finite-T integrator sums kernel(+xt)+kernel(-xt) over
   the Matsubara frequency (the last integration variable). If the kernel is
   GENUINELY even in that variable (kernel(+f0) == kernel(-f0)), the integrator
   can evaluate it ONCE and double the result -- a real saving, with NO change to
   the kernel body. We do NOT symbolically fold to the even part: for rational
   integrands the even part is a ratio-of-products that BLOATS the kernel 2-4x.
   Instead we leave the kernel untouched and only set the `matsubara_even` trait
   when evenness is verified symbolically. Note: fermionic dressings evaluated at
   f0-shifted arguments (e.g. ZQ[f0+p0, ...]) generally BREAK evenness, so most
   finite-T kernels are NOT even and keep the explicit two-call form. *)
        matsubaraEvenTrait = {};
        If[TrueQ[OptionValue["MatsubaraEven"]] && Length[OptionValue["IntegrationVariables"]] > 0,
            Module[{mVar, diff, isEven},
                mVar = Last[OptionValue["IntegrationVariables"]];
                mVar =
                    If[StringQ[mVar],
                        mVar
                        ,
                        mVar["Name"]
                    ];
                diff = expr - (expr /. s_Symbol /; SymbolName[s] === mVar :> -s);
                isEven = TrueQ[Quiet @ TimeConstrained[PossibleZeroQ[Simplify[diff]], 180, False]];
                If[isEven,
                    matsubaraEvenTrait = {"static constexpr bool matsubara_even = true;"};
                    ,
                    Message[MakeKernel::notEven, OptionValue["Name"], mVar];
                ];
            ]
        ];
        const = constExpr;
        While[ListQ[const], const = Plus @@ const];
        intVariables = FunKit`Private`prepParam /@ intVariables;
        intVariables = Map[Append[#, "Type" -> realType[spec]]&, intVariables];
        getArgs = FunKit`Private`prepParam /@ getArgs;
        getArgs = Map[Append[#, "Type" -> realType[spec]]&, getArgs];
        (********************************************************************)
        (* First, the kernel itself *)
        (********************************************************************)
(* Interpolators are `auto` in kernel() and constant(): the integrators pass a kernel their compact
   handles in its own precision (has_kernel_handle in DiFfRG/physics/interpolation/interpolator_handle.hh),
   so a single-precision kernel reads the single-precision copy of a double interpolator. *)
        parametersKernel =
            Map[
                Which[
                    #["AD"] === True,
                        Merge[{#, <|"Type" -> "auto"|>}, Last]
                    ,
                    StringQ[#["Type"]] && StringContainsQ[#["Type"], "Interpolator"],
                        Merge[{#, <|"Type" -> "auto"|>}, Last]
                    ,
                    KeyFreeQ[#, "Type"],
                        Message[MakeKernel::MissingType, #["Name"]];
                        Merge[{#, <|"Type" -> "auto"|>}, Last]
                    ,
                    True,
                        #
                ]&
                ,
                spec["Parameters"]
            ];
        kernel = withComputePrecision[spec,
            If[TrueQ[OptionValue["SplitKernel"]] || TrueQ[OptionValue["SeparateLookups"]],
                FunKit`MakeCppFunctionSplit[expr, "Name" -> "kernel", "Return" -> OptionValue["KernelReturnType"], "Suffix" -> "", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Decorator" -> OptionValue["Decorator"], "SeparateLookups" -> OptionValue["SeparateLookups"], "Parameters" -> Join[intVariables, getArgs, parametersKernel], "Body" -> StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["KernelBody"]], "ReturnTransform" -> OptionValue["KernelReturnTransform"]]
                ,
                FunKit`MakeCppFunction[expr, "Name" -> "kernel", "Return" -> OptionValue["KernelReturnType"], "Suffix" -> "", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Parameters" -> Join[intVariables, getArgs, parametersKernel], "Body" -> StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["KernelBody"]], "ReturnTransform" -> OptionValue["KernelReturnTransform"]]
            ]];
        constant = withComputePrecision[spec, FunKit`MakeCppFunction[constExpr, "Name" -> "constant", "Return" -> OptionValue["ConstantReturnType"], "Suffix" -> "", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Parameters" -> Join[getArgs, parametersKernel], "Body" -> StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["ConstantBody"]], "ReturnTransform" -> OptionValue["ConstantReturnTransform"]]];
        kernelTraits = kernelTraitMembers[OptionValue["KernelTraits"]];
        kernelClass = FunKit`MakeCppClass["TemplateTypes" -> {"_Regulator"}, "Name" -> OptionValue["Name"] <> "_kernel", "MembersPublic" -> Join[{"using Regulator = _Regulator;"}, matsubaraEvenTrait, kernelTraits, {kernel, constant}], "MembersPrivate" -> kernelDefs];
        kernelHeader = FunKit`MakeCppHeader["Includes" -> {"DiFfRG/physics/interpolation.hh", "DiFfRG/physics/physics.hh"}, "Body" -> {"namespace DiFfRG {", kernelClass, StringTemplate["} using DiFfRG::`1`_kernel;"][spec["Name"]]}];
        (********************************************************************)
        (* Next, the corresponding class holding the map and get functions *)
        (********************************************************************)
        (*We set up lists of parameters for the map/get functions, depending on their AD setting*)
        parameters = spec["Parameters"];
        params = FunKit`Private`prepParam /@ parameters;
        params = First @ processParameters[params, $ADReplacements];
        arguments = StringRiffle[Map[#["Name"]&, params], ", "];
        getArgs = FunKit`Private`prepParam /@ getArgs;
        getArgs = Map[Append[#, "Type" -> "double"]&, getArgs];
        getArgs = First @ processParameters[getArgs, $ADReplacements];
        preArguments = StringRiffle[Map[#["Name"]&, getArgs], ", "];
        If[preArguments =!= "",
            preArguments = preArguments <> ", "
        ];
        (* Choose the execution space. Default is TBB: the per-point get() of a GPU integrator must not be called
           from several threads at once, as the assemblers' per-point callbacks are. map_points() runs on any. *)
        exec =
            If[KeyFreeQ[spec, "Device"] || FreeQ[{"GPU", "Threads"}, spec["Device"]],
                "DiFfRG::TBB_exec"
                ,
                DeviceExecSpace[spec["Device"]]
            ];
        integratorTemplateParams = TemplateParameterGeneration[spec];
        integratorTemplateParams = StringRiffle[integratorTemplateParams, ", "];
        returnType = resultType[spec];
        returnTypePointer = StringTemplate["`1`*"][returnType];
        adSpecs =
            If[spec["AD"],
                Map[Merge[{#, <|"IntegratorTemplateParams" -> StringRiffle[TemplateParameterGeneration[spec, #["Replacements"]], ", "], "ReturnType" -> resultType[spec] /. #["Replacements"], "Params" -> Last @ processParameters[params, #["Replacements"]]|>}, Last]&, $ADSpecializations]
                ,
                {}
            ];
        (* Now, we create the header which holds the class with the integrators and the map/get methods *)
        integratorHeader =
            FunKit`MakeCppHeader[
(* kernel.hh, NOT a forward declaration of <Name>_kernel. The integrator member below is
   Integrator_*<..., <Name>_kernel<Regulator>, ...>, and completing that type evaluates the
   integrator's kernel-trait detectors (`requires { requires K::matsubara_even; }` and friends). On
   an INCOMPLETE K every such detector is a substitution failure, i.e. silently false -- so a TU
   that had not seen kernel.hh built a DIFFERENT integrator class than one that had. That is an ODR
   violation, and it is not benign: flows.cc instantiates set_T() -> refresh_matsubara() with the
   traits off while the CT_*.cc TUs instantiate map() with them on, and the two disagree about
   which Matsubara rule is on the axis. It produced a wrong RHS (a factor ~2 in dtZA), not a
   diagnostic. Costs compile time in every consumer of <Name>.hh; correctness first. *)
                "Includes" -> {"DiFfRG/physics/integration.hh", "DiFfRG/physics/physics.hh", "DiFfRG/physics/interpolation.hh", "kernel.hh"}
                ,
                "Body" ->
                    {
                        "namespace DiFfRG {\n"
                        ,
                        FunKit`MakeCppClass[
                            "Name" -> StringTemplate["`Name`_integrator"][spec]
                            ,
                            "MembersPublic" ->
                                Join[
                                    {FunKit`MakeCppFunction["Name" -> StringTemplate["`Name`_integrator"][spec], "Parameters" -> {<|"Type" -> "DiFfRG::QuadratureProvider", "Reference" -> True, "Const" -> False, "Name" -> "quadrature_provider"|>, <|"Type" -> "DiFfRG::ConfigTree", "Reference" -> True, "Const" -> True, "Name" -> "config"|>}, "Body" -> None, "Return" -> ""], getRegulator[OptionValue["Regulator"], OptionValue["RegulatorOpts"]], StringTemplate["`1`<`2`> integrator;"][spec["Integrator"], integratorTemplateParams]}
                                    ,
                                    Map[StringTemplate["`1`<`2`> integrator_`3`;"][spec["Integrator"], #["IntegratorTemplateParams"], #["Suffix"]]&, adSpecs]
                                    ,
                                    Map[FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnTypePointer, "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>}, params]]&, coordinates]
                                    ,
                                    If[Length[coordinates] > 0,
                                            #
                                            ,
                                            {}
                                        ]& @ {FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Body" -> "return device::apply([&](const auto...t){return map(dest, coordinates, t...);}, args);", "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> "IT*", "Reference" -> False, "Const" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> "C", "Const" -> True|>, <|"Name" -> "args", "Type" -> "device::tuple<T...>", "Reference" -> True, "Const" -> True|>}], "Templates" -> {"IT", "C", "...T"}]}
                                    ,
                                    Flatten[
                                        Map[
                                            With[{ad = #},
                                                Map[FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> StringTemplate["`1`*"][ad["ReturnType"]], "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>}, ad["Params"]]]&, coordinates]
                                            ]&
                                            ,
                                            adSpecs
                                        ]
                                    ]
                                    ,
                                    {FunKit`MakeCppFunction["Name" -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnType, "Reference" -> True, "Const" -> False|>}, getArgs, params]], FunKit`MakeCppFunction["Name" -> "get", "Return" -> "void", "Body" -> "device::apply([&](const auto...t){get(dest, " <> preArguments <> "t...);}, args);", "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> "IT", "Reference" -> True, "Const" -> False|>}, getArgs, {<|"Name" -> "args", "Type" -> "device::tuple<T...>", "Reference" -> True, "Const" -> True|>}], "Templates" -> {"IT", "...T"}]}
                                    ,
                                    Map[FunKit`MakeCppFunction["Name" -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> #["ReturnType"], "Reference" -> True, "Const" -> False|>}, getArgs, #["Params"]]]&, adSpecs]
                                    ,
                                    If[TrueQ[spec["MapPoints"]],
                                        Join[
                                            {FunKit`MakeCppFunction["Name" -> "map_points", "Return" -> "void", "Body" -> None, "Parameters" -> mapPointsParameters[returnType, Join[getArgs, params]]]}
                                            ,
                                            Map[FunKit`MakeCppFunction["Name" -> "map_points", "Return" -> "void", "Body" -> None, "Parameters" -> mapPointsParameters[#["ReturnType"], Join[getArgs, #["Params"]]]]&, adSpecs]
                                        ]
                                        ,
                                        {}
                                    ]
                                ]
                            ,
                            "MembersPrivate" -> {"DiFfRG::QuadratureProvider& quadrature_provider;"}
                        ]
                        ,
                        "}\nusing DiFfRG::" <> spec["Name"] <> "_integrator;"
                    }
            ];
        (* Finally, the code fo rall methods of the class. we will save them to different files, so they can all be compiled in separate units.*)
        integratorCpp["Constructor"] =
            FunKit`MakeCppBlock[
                "Includes" -> {"../kernel.hh"}
                ,
                "Body" ->
                    {
                        StringTemplate["#include \"../`Name`.hh\"\n"][spec]
                        ,
                        FunKit`MakeCppFunction[
                            "Name" -> StringTemplate["`Name`_integrator"][spec]
                            ,
                            "Class" -> StringTemplate["`Name`_integrator"][spec]
                            ,
                            "Suffix" ->
                                If[Length[adSpecs] > 0,
                                    ": integrator(quadrature_provider, config), " <> StringRiffle[Map[StringTemplate["integrator_`1`(quadrature_provider, config)"][#["Suffix"]]&, adSpecs], ", "] <> ", quadrature_provider(quadrature_provider)"
                                    ,
                                    ": integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)"
                                ]
                            ,
                            "Body" -> ""
                            ,
                            "Parameters" -> {<|"Type" -> "DiFfRG::QuadratureProvider", "Reference" -> True, "Const" -> False, "Name" -> "quadrature_provider"|>, <|"Type" -> "DiFfRG::ConfigTree", "Reference" -> True, "Const" -> True, "Name" -> "config"|>}
                            ,
                            "Return" -> ""
                        ]
                    }
            ];
        integratorCpp["CT", "get"] = FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> {StringTemplate["#include \"../`Name`.hh\"\n"][spec], FunKit`MakeCppFunction["Name" -> "get", "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["integrator.get(dest, `1` `2`);"][preArguments, arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnType, "Reference" -> True, "Const" -> False|>}, getArgs, params], "Return" -> "void"]}];
        integratorCpp["AD", "get"] = FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> Join[{StringTemplate["#include \"../`Name`.hh\"\n"][spec]}, Map[FunKit`MakeCppFunction["Name" -> "get", "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["integrator_`1`.get(dest, `2` `3`);"][#["Suffix"], preArguments, arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> #["ReturnType"], "Reference" -> True, "Const" -> False|>}, getArgs, #["Params"]], "Return" -> "void"]&, adSpecs]]];
        integratorCpp["CT", "map_points"] = FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> {StringTemplate["#include \"../`Name`.hh\"\n"][spec], FunKit`MakeCppFunction["Name" -> "map_points", "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["integrator.map_points(dest, `1` `2`);"][preArguments, arguments], "Parameters" -> mapPointsParameters[returnType, Join[getArgs, params]], "Return" -> "void"]}];
        integratorCpp["AD", "map_points"] = FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> Join[{StringTemplate["#include \"../`Name`.hh\"\n"][spec]}, Map[FunKit`MakeCppFunction["Name" -> "map_points", "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["integrator_`1`.map_points(dest, `2` `3`);"][#["Suffix"], preArguments, arguments], "Parameters" -> mapPointsParameters[#["ReturnType"], Join[getArgs, #["Params"]]], "Return" -> "void"]&, adSpecs]]];
        integratorCpp["CT", "map"] = Map[FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> {StringTemplate["#include \"../`Name`.hh\"\n"][spec], FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["return integrator.map(dest, coordinates, `1`);"][arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnTypePointer, "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>}, params]]}]&, coordinates];
        integratorCpp["AD", "map"] =
            Map[
                With[{coordinate = #},
                    FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> Join[{StringTemplate["#include \"../`Name`.hh\"\n"][spec]}, Map[FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Class" -> spec["Name"] <> "_integrator", "Body" -> StringTemplate["return integrator_`1`.map(dest, coordinates, `2`);"][#["Suffix"], arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> StringTemplate["`1`*"][#["ReturnType"]], "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> coordinate, "Const" -> True|>}, #["Params"]]]&, adSpecs]]]
                ]&
                ,
                coordinates
            ];
        outputPath = FileNameJoin[{flowDir, spec["Name"]}];
        ExportCode[FileNameJoin[{outputPath, spec["Name"] <> ".hh"}], integratorHeader];
        ExportCode[FileNameJoin[{outputPath, "kernel.hh"}], kernelHeader];
        sources = {FileNameJoin[{outputPath, "src", "constructor.cc"}]};
        ExportCode[sources[[-1]], integratorCpp["Constructor"]];
        AppendTo[sources, FileNameJoin[{outputPath, "src", "CT_get.cc"}]];
        ExportCode[sources[[-1]], integratorCpp["CT", "get"]];
        Do[
            AppendTo[sources, FileNameJoin[{outputPath, "src", StringTemplate["CT_map_`1`.cc"][i]}]];
            ExportCode[sources[[-1]], integratorCpp["CT", "map"][[i]]]
            ,
            {i, 1, Length[coordinates]}
        ];
        If[TrueQ[spec["MapPoints"]],
            AppendTo[sources, FileNameJoin[{outputPath, "src", "CT_map_points.cc"}]];
            ExportCode[sources[[-1]], integratorCpp["CT", "map_points"]];
            If[spec["AD"],
                AppendTo[sources, FileNameJoin[{outputPath, "src", "AD_map_points.cc"}]];
                ExportCode[sources[[-1]], integratorCpp["AD", "map_points"]]
            ]
        ];
        If[spec["AD"],
            AppendTo[sources, FileNameJoin[{outputPath, "src", "AD_get.cc"}]];
            ExportCode[sources[[-1]], integratorCpp["AD", "get"]];
            Do[
                AppendTo[sources, FileNameJoin[{outputPath, "src", StringTemplate["AD_map_`1`.cc"][i]}]];
                ExportCode[sources[[-1]], integratorCpp["AD", "map"][[i]]]
                ,
                {i, 1, Length[coordinates]}
            ];
        ];
        sources = Map[StringReplace[#, outputPath -> StringTemplate["${CMAKE_CURRENT_SOURCE_DIR}/`Name`"][spec]]&, sources];
        If[Export[FileNameJoin[{outputPath, "sources.m"}], sources] === $Failed,
            Message[MakeKernel::exportFailed, FileNameJoin[{outputPath, "sources.m"}]];
            Abort[]
        ];
        Print["Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"];
    ];

End[]

EndPackage[]
