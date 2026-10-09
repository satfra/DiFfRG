(* Created with the Wolfram Language : www.wolfram.com *)
(-9*FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {p1, -lf1, lf1 - p1}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {-p1 - p2, -lf1 + p1 + p2, 
    lf1}]*FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, 
   {p2, -lf1 + p1, lf1 - p1 - p2}]*FunKit`dressing[FunKit`Rdot, {cb, c}, 1, 
   {-lf1, lf1}]*(sp[lf1, p2]^2*sp[p1, p1] - 2*sp[lf1, p1]*sp[lf1, p2]*
    sp[p1, p2] + sp[lf1, p1]^2*sp[p2, p2] + 
   sp[lf1, lf1]*(sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2]))*
  (-(sp[p1, p1]^2*sp[p1, p2]^2) - 2*sp[p1, p1]*sp[p1, p2]^3 - 
   2*sp[p1, p2]^4 + sp[p1, p1]^3*sp[p2, p2] + 2*sp[p1, p1]^2*sp[p1, p2]*
    sp[p2, p2] + 2*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] - 
   sp[p1, p2]^3*sp[p2, p2] + sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2 + 
   sp[lf1, p2]*(2*sp[p1, p2]^2*(sp[p1, p2] + sp[p2, p2]) + 
     sp[p1, p1]^2*(3*sp[p1, p2] + sp[p2, p2]) + 
     sp[p1, p1]*(5*sp[p1, p2]^2 + 4*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2)) - 
   sp[lf1, p1]*(sp[p1, p1]^2*sp[p2, p2] + 
     sp[p1, p1]*(2*sp[p1, p2]^2 + 4*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) + 
     sp[p1, p2]*(2*sp[p1, p2]^2 + 5*sp[p1, p2]*sp[p2, p2] + 
       3*sp[p2, p2]^2))))/
 (4*FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {-lf1, lf1}]^2*
  FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {-lf1 + p1, lf1 - p1}]*
  FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, 
   {-lf1 + p1 + p2, lf1 - p1 - p2}]*(sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2])^2*
  (sp[p1, p1]^2 + 4*sp[p1, p2]^2 + sp[p1, p1]*(2*sp[p1, p2] - sp[p2, p2]) + 
   2*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2))
