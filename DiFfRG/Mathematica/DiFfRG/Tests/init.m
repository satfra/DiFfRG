(* init.m — Test initialization for DiFfRG Mathematica tests *)

(* Find the DiFfRG package directory (parent of Tests/) and put it at the FRONT of $Path.
   Prepending is essential: $UserBaseDirectory/Applications is searched earlier than any
   appended entry, and a DiFfRG symlink installed there (pointing at another checkout) would
   otherwise shadow the very tree we are testing -- silently running the tests against
   somebody else's generator.
   Note it is the PARENT of the DiFfRG/ directory that must be on $Path: the context
   DiFfRG`CodeTools` resolves to <entry>/DiFfRG/CodeTools.m. Putting the package directory
   itself on $Path (as this file used to do) never matches anything, which is why the tests
   silently ran against the installed copy.

   BOTH mechanisms below are required, because the two contexts resolve differently:
     - DiFfRG` and DiFfRG`CodeTools` are declared in PacletInfo.wl, so they go through the
       paclet manager, which outranks $Path entirely -> PacletDirectoryLoad.
     - the subcontexts (CodeTools`MakeKernel`, CodeTools`Directory`, ...) are NOT declared
       there and fall back to $Path -> prepend.
   Fixing only one of the two leaves the other half loading from the installed copy. *)
$diffrGPackageDir = DirectoryName[DirectoryName[$InputFileName]];
$diffrGPackageParent = ParentDirectory[$diffrGPackageDir];
PacletDirectoryLoad[$diffrGPackageParent];
$Path = Prepend[DeleteCases[$Path, $diffrGPackageParent | $diffrGPackageDir], $diffrGPackageParent];

(* Load the DiFfRG CodeTools package *)
Block[{Print},
    Needs["DiFfRG`CodeTools`"];
];

(* Report the directory actually resolved, not the one we hoped for. *)
$diffrGResolvedFile = FindFile["DiFfRG`CodeTools`"];
$diffrGResolvedDir =
    If[StringQ[$diffrGResolvedFile],
        DirectoryName[$diffrGResolvedFile]
        ,
        "<unresolved>"
    ];
Print["  DiFfRG CodeTools loaded from: " <> $diffrGResolvedDir];
If[StringQ[$diffrGResolvedFile] && !StringStartsQ[ExpandFileName[$diffrGResolvedFile], ExpandFileName[$diffrGPackageDir]],
    Print["  WARNING: tests are running against a DiFfRG package OUTSIDE this repository!"];
    Print["           expected under: " <> $diffrGPackageDir];
];
