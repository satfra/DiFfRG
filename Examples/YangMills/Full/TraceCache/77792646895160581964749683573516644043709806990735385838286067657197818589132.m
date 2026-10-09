(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1, -l1 - p1}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {-l1, p2, l1 - p2}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, 
   {l1 + p1, -l1 + p2, -p1 - p2}]*FunKit`dressing[FunKit`Rdot, {A, A}, 1, 
   {-l1, l1}]*Sqrt[sp[p, p]]*(-8*cos[p2, l1]^3*sp[l1, l1]^(3/2) - 
   2*cos[p1, l1]^2*(cos[p2, l1]*Sqrt[sp[l1, l1]]*(2*sp[l1, l1] - 
       5*sp[p, p]) + sp[l1, l1]*Sqrt[sp[p, p]] + 6*cos[p2, l1]^2*sp[l1, l1]*
      Sqrt[sp[p, p]]) + 2*cos[p2, l1]*Sqrt[sp[l1, l1]]*
    (2*sp[l1, l1] - sp[p, p]) - 4*cos[p1, l1]^3*cos[p2, l1]*sp[l1, l1]*
    Sqrt[sp[p, p]] - 4*cos[p2, l1]^2*sp[l1, l1]*Sqrt[sp[p, p]] + 
   3*Sqrt[sp[p, p]]*(2*sp[l1, l1] + sp[p, p]) + 
   cos[p1, l1]*((2 - 12*cos[p2, l1]^2)*sp[l1, l1]^(3/2) + 
     2*cos[p2, l1]*(3 - 4*cos[p2, l1]^2)*sp[l1, l1]*Sqrt[sp[p, p]] + 
     (5 - 4*cos[p2, l1]^2)*Sqrt[sp[l1, l1]]*sp[p, p] + 
     6*cos[p2, l1]*sp[p, p]^(3/2))))/
 (2*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 - p1, l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {-l1 + p2, l1 - p2}]*
  (sp[l1, l1] + 2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p]))
