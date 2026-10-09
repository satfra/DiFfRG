(* Created with the Wolfram Language : www.wolfram.com *)
-1/4*((-1 + 2*cos[p1, l1]*cos[p2, l1] + 2*cos[p2, l1]^2)*
   FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {-l1, l1 + p1 + p2, 
     -p1 - p2}]*FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, 
    {l1, p2, -l1 - p2}]*FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, 
    {p1, l1 + p2, -l1 - p1 - p2}]*FunKit`dressing[FunKit`Rdot, {cb, c}, 1, 
    {l1 + p2, -l1 - p2}]*(2*cos[p1, l1]*Sqrt[sp[l1, l1]] + 
    4*cos[p2, l1]*Sqrt[sp[l1, l1]] + 3*Sqrt[sp[p, p]])*Sqrt[sp[p, p]])/
  (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
   FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {l1 + p2, -l1 - p2}]^2*
   FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, 
    {l1 + p1 + p2, -l1 - p1 - p2}])
