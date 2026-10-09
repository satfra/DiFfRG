(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
   FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1 - p2 - p3, l1, 
     -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
    {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*FunKit`dressing[FunKit`Rdot, 
    {A, A}, 1, {-l1, l1}]*sp[l1, l1]*
   (3*(-160 + 18*cos[p1, l1]^4 - 9*cos[p2, l1]^4 - 27*cos[p2, l1]^3*
       cos[p3, l1] - 65*cos[p3, l1]^2 - 9*cos[p3, l1]^4 + 
      36*cos[p1, l1]^3*(cos[p2, l1] + cos[p3, l1]) + 
      cos[p1, l1]^2*(-178 + 9*cos[p2, l1]^2 + 30*cos[p2, l1]*cos[p3, l1] + 
        9*cos[p3, l1]^2) - cos[p2, l1]^2*(65 + 18*cos[p3, l1]^2) + 
      cos[p2, l1]*(48*cos[p3, l1] - 27*cos[p3, l1]^3) - 
      cos[p1, l1]*(9*cos[p2, l1]^3 + 15*cos[p2, l1]^2*cos[p3, l1] + 
        cos[p3, l1]*(178 + 9*cos[p3, l1]^2) + cos[p2, l1]*
         (178 + 15*cos[p3, l1]^2)))*sp[l1, l1]^2 + 
    (852*cos[p1, l1]^3 + 795*cos[p2, l1]^3 + 2019*cos[p2, l1]^2*cos[p3, l1] + 
      1278*cos[p1, l1]^2*(cos[p2, l1] + cos[p3, l1]) + 
      4*cos[p1, l1]*(854 + 504*cos[p2, l1]^2 + 825*cos[p2, l1]*cos[p3, l1] + 
        504*cos[p3, l1]^2) + cos[p3, l1]*(1708 + 795*cos[p3, l1]^2) + 
      cos[p2, l1]*(1708 + 2019*cos[p3, l1]^2))*sp[l1, l1]^(3/2)*
     Sqrt[sp[p, p]] - 4*(714 + 1222*cos[p1, l1]^2 + 89*cos[p2, l1]^2 + 
      85*cos[p2, l1]*cos[p3, l1] + 89*cos[p3, l1]^2 + 
      1222*cos[p1, l1]*(cos[p2, l1] + cos[p3, l1]))*sp[l1, l1]*sp[p, p] + 
    3264*(2*cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     sp[p, p]^(3/2) - 2512*sp[p, p]^2))/
  (1176*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1 - p2 - p3, -l1 + p1 + p2 + p3}]*
   (sp[l1, l1] - 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
   (sp[l1, l1] - 2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     Sqrt[sp[p, p]] + sp[p, p])) + 
 (cos[p1, l1]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
    {p1, l1 - p1, -l1}]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
    {-p1 - p2 - p3, l1, -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, 
    {A, A, A, A}, 1, {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*
   FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
   (cos[p1, l1]^2*sp[l1, l1]^(3/2)*(-2109*sp[l1, l1] + 
      2892*(cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] - 
      2764*sp[p, p])*Sqrt[sp[p, p]] + 2379*cos[p1, l1]^3*sp[l1, l1]^2*
     sp[p, p] + 4*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*(-115*sp[l1, l1]^2 + 
      230*(cos[p2, l1] + cos[p3, l1])*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] - 
      80*sp[l1, l1]*sp[p, p] + 32*(cos[p2, l1] + cos[p3, l1])*
       Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 84*sp[p, p]^2) + 
    cos[p1, l1]*sp[l1, l1]*(453*sp[l1, l1]^2 - 
      906*(cos[p2, l1] + cos[p3, l1])*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
      2170*sp[l1, l1]*sp[p, p] - 2362*(cos[p2, l1] + cos[p3, l1])*
       Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 559*sp[p, p]^2)))/
  (588*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1 - p2 - p3, -l1 + p1 + p2 + p3}]*
   (sp[l1, l1] - 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
   (sp[l1, l1] - 2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     Sqrt[sp[p, p]] + sp[p, p])) + 
 (FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
   FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1 - p2 - p3, l1, 
     -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
    {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*FunKit`dressing[FunKit`Rdot, 
    {A, A}, 1, {-l1, l1}]*sp[p, p]*
   (-163/(FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
      FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
      FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
      FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
        -l1 + p1 + p2 + p3}]) - (9*sp[p, p])/
     (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
         -l1 + p1 + p2 + p3}]*sp[l1, l1] - 2*cos[p1, l1]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
         -l1 + p1 + p2 + p3}]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
      FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
       FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
         -l1 + p1 + p2 + p3}]*sp[p, p]) + 
    (sp[p, p]*(-5/(FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
         FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
         FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
         FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
           -l1 + p1 + p2 + p3}]) - (7*sp[p, p])/
        (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
            -l1 + p1 + p2 + p3}]*sp[l1, l1] - 2*cos[p1, l1]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
            -l1 + p1 + p2 + p3}]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
         FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
          FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1 - p2 - p3, 
            -l1 + p1 + p2 + p3}]*sp[p, p])))/(sp[l1, l1] - 
      2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
       Sqrt[sp[p, p]] + sp[p, p])))/147 + 
 (cos[p2, l1]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
    {p1, l1 - p1, -l1}]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
    {-p1 - p2 - p3, l1, -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, 
    {A, A, A, A}, 1, {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*
   FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*Sqrt[sp[l1, l1]]*
   (-364*sp[l1, l1]^2*Sqrt[sp[p, p]] + 459*cos[p2, l1]^3*sp[l1, l1]^(3/2)*
     sp[p, p] + 728*cos[p3, l1]*sp[l1, l1]^(3/2)*sp[p, p] - 
    32*sp[l1, l1]*sp[p, p]^(3/2) + 520*cos[p3, l1]*Sqrt[sp[l1, l1]]*
     sp[p, p]^2 + 528*sp[p, p]^(5/2) + 3*cos[p2, l1]^2*sp[l1, l1]*
     Sqrt[sp[p, p]]*(83*sp[l1, l1] + 144*cos[p3, l1]*Sqrt[sp[l1, l1]]*
       Sqrt[sp[p, p]] + 115*sp[p, p]) + 12*cos[p1, l1]^3*
     (9*cos[p2, l1]*sp[l1, l1]^2*Sqrt[sp[p, p]] + 17*sp[l1, l1]^(3/2)*
       sp[p, p]) + cos[p1, l1]*(-228*sp[l1, l1]^(5/2) + 
      6*(189*cos[p2, l1] + 76*cos[p3, l1])*sp[l1, l1]^2*Sqrt[sp[p, p]] + 
      (-665 + 93*cos[p2, l1]^2 - 420*cos[p2, l1]*cos[p3, l1])*
       sp[l1, l1]^(3/2)*sp[p, p] + 2*(493*cos[p2, l1] - 356*cos[p3, l1])*
       sp[l1, l1]*sp[p, p]^(3/2) - 1700*Sqrt[sp[l1, l1]]*sp[p, p]^2) + 
    cos[p2, l1]*(-246*sp[l1, l1]^(5/2) + 492*cos[p3, l1]*sp[l1, l1]^2*
       Sqrt[sp[p, p]] - 19*sp[l1, l1]^(3/2)*sp[p, p] + 
      390*cos[p3, l1]*sp[l1, l1]*sp[p, p]^(3/2) - 125*Sqrt[sp[l1, l1]]*
       sp[p, p]^2) + 2*cos[p1, l1]^2*sp[l1, l1]*
     (150*sp[l1, l1]*Sqrt[sp[p, p]] + 27*cos[p2, l1]^2*sp[l1, l1]*
       Sqrt[sp[p, p]] + 546*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p] + 
      1105*sp[p, p]^(3/2) - 9*cos[p2, l1]*(3*sp[l1, l1]^(3/2) - 
        3*cos[p3, l1]*sp[l1, l1]*Sqrt[sp[p, p]] - 37*Sqrt[sp[l1, l1]]*
         sp[p, p]))))/(1176*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {-l1, l1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1 - p2 - p3, -l1 + p1 + p2 + p3}]*
   (sp[l1, l1] - 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
   (sp[l1, l1] - 2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     Sqrt[sp[p, p]] + sp[p, p])) - 
 (cos[p3, l1]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
    {p1, l1 - p1, -l1}]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
    {-p1 - p2 - p3, l1, -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, 
    {A, A, A, A}, 1, {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*
   FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*Sqrt[sp[l1, l1]]*
   (246*cos[p3, l1]*sp[l1, l1]^(5/2) + 364*sp[l1, l1]^2*Sqrt[sp[p, p]] - 
    249*cos[p3, l1]^2*sp[l1, l1]^2*Sqrt[sp[p, p]] + 
    19*cos[p3, l1]*sp[l1, l1]^(3/2)*sp[p, p] - 459*cos[p3, l1]^3*
     sp[l1, l1]^(3/2)*sp[p, p] + 32*sp[l1, l1]*sp[p, p]^(3/2) - 
    345*cos[p3, l1]^2*sp[l1, l1]*sp[p, p]^(3/2) + 
    125*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^2 - 528*sp[p, p]^(5/2) + 
    9*cos[p2, l1]^3*(12*cos[p3, l1]*sp[l1, l1]^2*Sqrt[sp[p, p]] - 
      19*sp[l1, l1]^(3/2)*sp[p, p]) - 12*cos[p1, l1]^3*
     (6*cos[p2, l1]*sp[l1, l1]^2*Sqrt[sp[p, p]] + 9*cos[p3, l1]*sp[l1, l1]^2*
       Sqrt[sp[p, p]] + 17*sp[l1, l1]^(3/2)*sp[p, p]) + 
    cos[p2, l1]*Sqrt[sp[l1, l1]]*(486*sp[l1, l1]^2 - 
      1374*cos[p3, l1]*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
      (1372 - 603*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] - 
      2172*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 1268*sp[p, p]^2) + 
    cos[p1, l1]*(-6*(-38 + 3*cos[p2, l1]^2 + 3*cos[p2, l1]*cos[p3, l1])*
       sp[l1, l1]^(5/2) + 6*(3*cos[p2, l1]^3 - 189*cos[p3, l1] + 
        42*cos[p2, l1]^2*cos[p3, l1] + cos[p2, l1]*(-460 + 3*cos[p3, l1]^2))*
       sp[l1, l1]^2*Sqrt[sp[p, p]] + (665 + 1986*cos[p2, l1]^2 + 
        2406*cos[p2, l1]*cos[p3, l1] - 93*cos[p3, l1]^2)*sp[l1, l1]^(3/2)*
       sp[p, p] - 34*(131*cos[p2, l1] + 29*cos[p3, l1])*sp[l1, l1]*
       sp[p, p]^(3/2) + 1700*Sqrt[sp[l1, l1]]*sp[p, p]^2) - 
    18*cos[p2, l1]^2*sp[l1, l1]*(-6*cos[p3, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]] + 
      2*cos[p3, l1]*Sqrt[sp[l1, l1]]*(3*sp[l1, l1] + 8*sp[p, p]) + 
      Sqrt[sp[p, p]]*(49*sp[l1, l1] + 99*sp[p, p])) + 
    2*cos[p1, l1]^2*sp[l1, l1]*(9*cos[p3, l1]*Sqrt[sp[l1, l1]]*
       (3*sp[l1, l1] - 37*sp[p, p]) - 27*cos[p3, l1]^2*sp[l1, l1]*
       Sqrt[sp[p, p]] - 5*Sqrt[sp[p, p]]*(30*sp[l1, l1] + 221*sp[p, p]) + 
      3*cos[p2, l1]*(6*sp[l1, l1]^(3/2) - 9*cos[p3, l1]*sp[l1, l1]*
         Sqrt[sp[p, p]] + 280*Sqrt[sp[l1, l1]]*sp[p, p]))))/
  (1176*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1 - p2 - p3, -l1 + p1 + p2 + p3}]*
   (sp[l1, l1] - 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
   (sp[l1, l1] - 2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     Sqrt[sp[p, p]] + sp[p, p])) - 
 (FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
   FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1 - p2 - p3, l1, 
     -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
    {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*FunKit`dressing[FunKit`Rdot, 
    {A, A}, 1, {-l1, l1}]*(-693*cos[p3, l1]^2*sp[l1, l1]^3 + 
    54*cos[p3, l1]^4*sp[l1, l1]^3 + 216*cos[p1, l1]^5*sp[l1, l1]^(5/2)*
     Sqrt[sp[p, p]] - 54*cos[p2, l1]^5*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
    96*cos[p3, l1]*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
    1215*cos[p3, l1]^3*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] - 
    54*cos[p3, l1]^5*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
    128*sp[l1, l1]^2*sp[p, p] - 2019*cos[p3, l1]^2*sp[l1, l1]^2*sp[p, p] + 
    657*cos[p3, l1]^4*sp[l1, l1]^2*sp[p, p] + 224*cos[p3, l1]*
     sp[l1, l1]^(3/2)*sp[p, p]^(3/2) + 1863*cos[p3, l1]^3*sp[l1, l1]^(3/2)*
     sp[p, p]^(3/2) + 224*sp[l1, l1]*sp[p, p]^2 - 
    1857*cos[p3, l1]^2*sp[l1, l1]*sp[p, p]^2 + 
    192*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^(5/2) + 96*sp[p, p]^3 + 
    9*cos[p2, l1]^4*sp[l1, l1]^2*(6*sp[l1, l1] - 24*cos[p3, l1]*
       Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 73*sp[p, p]) - 
    18*cos[p1, l1]^4*sp[l1, l1]^2*(6*sp[l1, l1] - 
      30*(cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
      185*sp[p, p]) + 3*cos[p2, l1]^2*sp[l1, l1]*
     (3*(-77 + 24*cos[p3, l1]^2)*sp[l1, l1]^2 - 18*cos[p3, l1]*
       (-82 + 7*cos[p3, l1]^2)*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
      (-673 + 804*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] + 
      2190*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) - 619*sp[p, p]^2) - 
    2*cos[p1, l1]^2*sp[l1, l1]*
     (18*(13 + 3*cos[p2, l1]^2 + 6*cos[p2, l1]*cos[p3, l1] + 3*cos[p3, l1]^2)*
       sp[l1, l1]^2 - 18*(241*cos[p3, l1] + 3*cos[p2, l1]^2*cos[p3, l1] + 
        cos[p2, l1]*(241 + 3*cos[p3, l1]^2))*sp[l1, l1]^(3/2)*
       Sqrt[sp[p, p]] + 9*(130 + 479*cos[p2, l1]^2 + 1042*cos[p2, l1]*
         cos[p3, l1] + 479*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] - 
      6777*(cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 
      1049*sp[p, p]^2) + 27*cos[p2, l1]^3*sp[l1, l1]^(3/2)*
     (-14*cos[p3, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]] + 
      3*cos[p3, l1]*Sqrt[sp[l1, l1]]*(2*sp[l1, l1] + 23*sp[p, p]) + 
      3*Sqrt[sp[p, p]]*(15*sp[l1, l1] + 23*sp[p, p])) + 
    18*cos[p1, l1]^3*sp[l1, l1]^(3/2)*(24*cos[p2, l1]^2*sp[l1, l1]*
       Sqrt[sp[p, p]] + 24*cos[p3, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]] - 
      2*cos[p3, l1]*Sqrt[sp[l1, l1]]*(6*sp[l1, l1] + 283*sp[p, p]) + 
      Sqrt[sp[p, p]]*(143*sp[l1, l1] + 288*sp[p, p]) - 
      2*cos[p2, l1]*(6*sp[l1, l1]^(3/2) - 24*cos[p3, l1]*sp[l1, l1]*
         Sqrt[sp[p, p]] + 283*Sqrt[sp[l1, l1]]*sp[p, p])) + 
    cos[p2, l1]*Sqrt[sp[l1, l1]]*(-216*cos[p3, l1]^4*sp[l1, l1]^2*
       Sqrt[sp[p, p]] + 81*cos[p3, l1]^3*sp[l1, l1]^(3/2)*
       (2*sp[l1, l1] + 23*sp[p, p]) + 18*cos[p3, l1]^2*sp[l1, l1]*
       Sqrt[sp[p, p]]*(246*sp[l1, l1] + 365*sp[p, p]) + 
      32*Sqrt[sp[p, p]]*(3*sp[l1, l1]^2 + 7*sp[l1, l1]*sp[p, p] + 
        6*sp[p, p]^2) - 6*cos[p3, l1]*Sqrt[sp[l1, l1]]*
       (273*sp[l1, l1]^2 + 766*sp[l1, l1]*sp[p, p] + 670*sp[p, p]^2)) - 
    cos[p1, l1]*(-27*cos[p3, l1]^3*sp[l1, l1]^2*(2*sp[l1, l1] - 
        41*sp[p, p]) - 27*cos[p2, l1]^3*sp[l1, l1]^2*
       (2*sp[l1, l1] - 18*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] - 
        41*sp[p, p]) + 162*cos[p2, l1]^4*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
      162*cos[p3, l1]^4*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
      64*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2)*(2*sp[l1, l1] + sp[p, p]) - 
      18*cos[p3, l1]^2*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
       (359*sp[l1, l1] + 521*sp[p, p]) + cos[p3, l1]*sp[l1, l1]*
       (1602*sp[l1, l1]^2 + 5505*sp[l1, l1]*sp[p, p] + 4660*sp[p, p]^2) + 
      cos[p2, l1]*sp[l1, l1]*(-18*(-89 + 6*cos[p3, l1]^2)*sp[l1, l1]^2 + 
        18*cos[p3, l1]*(-788 + 27*cos[p3, l1]^2)*sp[l1, l1]^(3/2)*
         Sqrt[sp[p, p]] + 3*(1835 + 1866*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] - 
        20214*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 
        4660*sp[p, p]^2) + 18*cos[p2, l1]^2*sp[l1, l1]^(3/2)*
       (36*cos[p3, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]] + 
        cos[p3, l1]*Sqrt[sp[l1, l1]]*(-6*sp[l1, l1] + 311*sp[p, p]) - 
        Sqrt[sp[p, p]]*(359*sp[l1, l1] + 521*sp[p, p])))))/
  (1176*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1 - p2 - p3, -l1 + p1 + p2 + p3}]*
   (sp[l1, l1] - 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
   (sp[l1, l1] - 2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     Sqrt[sp[p, p]] + sp[p, p])) + 
 (FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
   FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1 - p2 - p3, l1, 
     -l1 + p1 + p2 + p3}]*FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
    {p3, p2, l1 - p1 - p2 - p3, -l1 + p1}]*FunKit`dressing[FunKit`Rdot, 
    {A, A}, 1, {-l1, l1}]*(72*cos[p1, l1]^6*sp[l1, l1]^2*sp[p, p] - 
    8*cos[p1, l1]^5*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
     (9*sp[l1, l1] - 27*(cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
       Sqrt[sp[p, p]] + 194*sp[p, p]) + 2*cos[p1, l1]^4*sp[l1, l1]*
     (9*sp[l1, l1]^2 - 90*(cos[p2, l1] + cos[p3, l1])*sp[l1, l1]^(3/2)*
       Sqrt[sp[p, p]] + 2*(388 + 45*cos[p2, l1]^2 + 102*cos[p2, l1]*
         cos[p3, l1] + 45*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] - 
      1940*(cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 
      3352*sp[p, p]^2) - 4*cos[p1, l1]^3*Sqrt[sp[l1, l1]]*
     (cos[p3, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]]*(27*sp[l1, l1] + 
        710*sp[p, p]) + cos[p2, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]]*
       (27*sp[l1, l1] - 24*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
        710*sp[p, p]) + Sqrt[sp[p, p]]*(97*sp[l1, l1]^2 + 
        1590*sp[l1, l1]*sp[p, p] + 1544*sp[p, p]^2) - 
      cos[p3, l1]*Sqrt[sp[l1, l1]]*(9*sp[l1, l1]^2 + 776*sp[l1, l1]*
         sp[p, p] + 3352*sp[p, p]^2) - cos[p2, l1]*(9*sp[l1, l1]^(5/2) - 
        66*cos[p3, l1]*sp[l1, l1]^2*Sqrt[sp[p, p]] + 8*(97 + 3*cos[p3, l1]^2)*
         sp[l1, l1]^(3/2)*sp[p, p] - 1264*cos[p3, l1]*sp[l1, l1]*
         sp[p, p]^(3/2) + 3352*Sqrt[sp[l1, l1]]*sp[p, p]^2)) - 
    3*sp[l1, l1]*(-6*cos[p2, l1]^5*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
      cos[p2, l1]^4*sp[l1, l1]*(3*sp[l1, l1] - 24*cos[p3, l1]*
         Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 22*sp[p, p]) + 
      cos[p3, l1]^2*(-6*cos[p3, l1]^3*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
        4*sp[p, p]*(sp[l1, l1] + 3*sp[p, p]) + cos[p3, l1]^2*sp[l1, l1]*
         (3*sp[l1, l1] + 22*sp[p, p]) - cos[p3, l1]*Sqrt[sp[l1, l1]]*
         Sqrt[sp[p, p]]*(11*sp[l1, l1] + 38*sp[p, p])) + 
      cos[p2, l1]^2*(-30*cos[p3, l1]^3*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
        4*sp[p, p]*(sp[l1, l1] + 3*sp[p, p]) + 2*cos[p3, l1]^2*sp[l1, l1]*
         (3*sp[l1, l1] + 118*sp[p, p]) - cos[p3, l1]*Sqrt[sp[l1, l1]]*
         Sqrt[sp[p, p]]*(59*sp[l1, l1] + 134*sp[p, p])) + 
      cos[p2, l1]^3*(-30*cos[p3, l1]^2*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] - 
        Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*(11*sp[l1, l1] + 38*sp[p, p]) + 
        cos[p3, l1]*sp[l1, l1]*(9*sp[l1, l1] + 140*sp[p, p])) + 
      cos[p2, l1]*cos[p3, l1]*(-24*cos[p3, l1]^3*sp[l1, l1]^(3/2)*
         Sqrt[sp[p, p]] + 12*sp[p, p]*(sp[l1, l1] + 2*sp[p, p]) - 
        cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*(59*sp[l1, l1] + 
          134*sp[p, p]) + cos[p3, l1]^2*sp[l1, l1]*(9*sp[l1, l1] + 
          140*sp[p, p]))) + cos[p1, l1]*(-36*cos[p2, l1]^5*sp[l1, l1]^2*
       sp[p, p] + 6*cos[p2, l1]^4*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
       (9*sp[l1, l1] - 24*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
        22*sp[p, p]) + cos[p2, l1]^3*sp[l1, l1]*(-9*sp[l1, l1]^2 + 
        156*cos[p3, l1]*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] - 
        4*(-31 + 45*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] + 
        840*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) + 272*sp[p, p]^2) - 
      cos[p2, l1]^2*(180*cos[p3, l1]^3*sp[l1, l1]^2*sp[p, p] - 
        12*cos[p3, l1]^2*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
         (11*sp[l1, l1] + 118*sp[p, p]) + 3*cos[p3, l1]*sp[l1, l1]*
         (5*sp[l1, l1]^2 + 84*sp[l1, l1]*sp[p, p] - 208*sp[p, p]^2) + 
        8*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*(16*sp[l1, l1]^2 + 
          369*sp[l1, l1]*sp[p, p] + 386*sp[p, p]^2)) + 
      cos[p3, l1]*(-36*cos[p3, l1]^4*sp[l1, l1]^2*sp[p, p] + 
        6*cos[p3, l1]^3*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
         (9*sp[l1, l1] + 22*sp[p, p]) + 16*sp[p, p]*(94*sp[l1, l1]^2 + 
          193*sp[l1, l1]*sp[p, p] + 98*sp[p, p]^2) + cos[p3, l1]^2*sp[l1, l1]*
         (-9*sp[l1, l1]^2 + 124*sp[l1, l1]*sp[p, p] + 272*sp[p, p]^2) - 
        8*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*(16*sp[l1, l1]^2 + 
          369*sp[l1, l1]*sp[p, p] + 386*sp[p, p]^2)) + 
      cos[p2, l1]*(-144*cos[p3, l1]^4*sp[l1, l1]^2*sp[p, p] + 
        12*cos[p3, l1]^3*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
         (13*sp[l1, l1] + 70*sp[p, p]) - 3*cos[p3, l1]^2*sp[l1, l1]*
         (5*sp[l1, l1]^2 + 84*sp[l1, l1]*sp[p, p] - 208*sp[p, p]^2) + 
        16*sp[p, p]*(94*sp[l1, l1]^2 + 193*sp[l1, l1]*sp[p, p] + 
          98*sp[p, p]^2) - 4*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*
         (25*sp[l1, l1]^2 + 1446*sp[l1, l1]*sp[p, p] + 1544*sp[p, p]^2))) - 
    cos[p1, l1]^2*(72*cos[p2, l1]^4*sp[l1, l1]^2*sp[p, p] + 
      72*cos[p3, l1]^4*sp[l1, l1]^2*sp[p, p] + 2*cos[p3, l1]^3*
       sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*(-9*sp[l1, l1] + 190*sp[p, p]) + 
      2*cos[p2, l1]^3*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
       (-9*sp[l1, l1] + 102*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
        190*sp[p, p]) + cos[p2, l1]^2*sp[l1, l1]*(-9*sp[l1, l1]^2 + 
        18*cos[p3, l1]*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]] + 
        4*(-419 + 48*cos[p3, l1]^2)*sp[l1, l1]*sp[p, p] + 
        204*cos[p3, l1]*Sqrt[sp[l1, l1]]*sp[p, p]^(3/2) - 6976*sp[p, p]^2) - 
      16*sp[p, p]*(94*sp[l1, l1]^2 + 193*sp[l1, l1]*sp[p, p] + 
        98*sp[p, p]^2) + 6*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*
       (97*sp[l1, l1]^2 + 1590*sp[l1, l1]*sp[p, p] + 1544*sp[p, p]^2) - 
      cos[p3, l1]^2*sp[l1, l1]*(9*sp[l1, l1]^2 + 1676*sp[l1, l1]*sp[p, p] + 
        6976*sp[p, p]^2) + 2*cos[p2, l1]*(102*cos[p3, l1]^3*sp[l1, l1]^2*
         sp[p, p] + 3*cos[p3, l1]^2*sp[l1, l1]^(3/2)*Sqrt[sp[p, p]]*
         (3*sp[l1, l1] + 34*sp[p, p]) + 3*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]]*
         (97*sp[l1, l1]^2 + 1590*sp[l1, l1]*sp[p, p] + 1544*sp[p, p]^2) - 
        cos[p3, l1]*sp[l1, l1]*(15*sp[l1, l1]^2 + 1364*sp[l1, l1]*sp[p, p] + 
          6880*sp[p, p]^2)))))/(392*FunKit`dressing[FunKit`InverseProp, 
    {A, A}, 1, {-l1, l1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1, -l1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1, -l1 + p1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
    {l1 - p1 - p2 - p3, -l1 + p1 + p2 + p3}]*
   (sp[l1, l1] - 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
   (sp[l1, l1] - 2*(cos[p1, l1] + cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
     Sqrt[sp[p, p]] + sp[p, p]))
