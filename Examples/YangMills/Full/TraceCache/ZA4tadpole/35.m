(* Created with the Wolfram Language : www.wolfram.com *)
(-2*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1, l1, -l1 + p1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, {-p2, p2, l1, -l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
  (4*sp[l1, l1]^4*sp[p1, p1]^2*sp[p2, p2]^2 + sp[l1, l1]^3*sp[p2, p2]*
    (3*sp[l1, p2]^2*sp[p1, p1]^2 - 6*sp[l1, p2]*(sp[l1, p1] - 6*sp[p1, p1])*
      sp[p1, p1]*sp[p1, p2] + sp[l1, p1]^2*(2*sp[p1, p2]^2 - 
       3*sp[p1, p1]*sp[p2, p2]) + 2*sp[l1, p1]*sp[p1, p1]*
      (-9*sp[p1, p2]^2 + 5*sp[p1, p1]*sp[p2, p2]) + 
     sp[p1, p1]^2*(11*sp[p1, p2]^2 + 13*sp[p1, p1]*sp[p2, p2])) - 
   sp[l1, p1]^2*(sp[l1, p2]^4*sp[p1, p1]^2 - 2*sp[l1, p1]*sp[l1, p2]^3*
      sp[p1, p1]*sp[p1, p2] - 2*sp[l1, p1]*sp[l1, p2]*
      (sp[l1, p1] - sp[p1, p1])^2*sp[p1, p2]*sp[p2, p2] + 
     sp[l1, p1]^2*(sp[l1, p1] - sp[p1, p1])^2*sp[p2, p2]^2 + 
     sp[l1, p2]^2*(sp[p1, p1]^2*(sp[p1, p2]^2 - 2*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       2*sp[l1, p1]*sp[p1, p1]*(-sp[p1, p2]^2 + 2*sp[p1, p1]*sp[p2, p2]))) + 
   sp[l1, l1]*sp[l1, p1]*(2*sp[l1, p2]^4*sp[p1, p1]^2 - 
     4*sp[l1, p1]*sp[l1, p2]^3*sp[p1, p1]*sp[p1, p2] + 
     2*sp[l1, p2]*sp[p1, p2]*(sp[l1, p1]^2*sp[p1, p2]^2 - 
       2*sp[l1, p1]^3*sp[p2, p2] + sp[p1, p1]^2*(sp[p1, p2]^2 - 
         3*sp[p1, p1]*sp[p2, p2]) + 2*sp[l1, p1]*sp[p1, p1]*
        (-sp[p1, p2]^2 + 3*sp[p1, p1]*sp[p2, p2])) + 
     sp[l1, p1]*sp[p2, p2]*(2*sp[l1, p1]^3*sp[p2, p2] - 
       sp[l1, p1]^2*(sp[p1, p2]^2 - 24*sp[p1, p1]*sp[p2, p2]) + 
       2*sp[l1, p1]*sp[p1, p1]*(sp[p1, p2]^2 - 6*sp[p1, p1]*sp[p2, p2]) - 
       sp[p1, p1]^2*(sp[p1, p2]^2 + 3*sp[p1, p1]*sp[p2, p2])) + 
     sp[l1, p2]^2*(-18*sp[p1, p1]^3*sp[p2, p2] + 2*sp[l1, p1]^2*
        (sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 3*sp[l1, p1]*sp[p1, p1]*
        (-sp[p1, p2]^2 + 10*sp[p1, p1]*sp[p2, p2]))) - 
   sp[l1, l1]^2*(sp[l1, p2]^4*sp[p1, p1]^2 - 2*sp[l1, p1]*sp[l1, p2]^3*
      sp[p1, p1]*sp[p1, p2] - 2*sp[l1, p2]*(sp[l1, p1]^3 + 
       6*sp[l1, p1]^2*sp[p1, p1] - 41*sp[l1, p1]*sp[p1, p1]^2 + 
       18*sp[p1, p1]^3)*sp[p1, p2]*sp[p2, p2] + sp[l1, p1]^4*sp[p2, p2]^2 + 
     4*sp[l1, p1]^3*sp[p2, p2]*(sp[p1, p2]^2 + 3*sp[p1, p1]*sp[p2, p2]) + 
     sp[p1, p1]^2*(sp[p1, p2]^4 - 3*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] - 
       4*sp[p1, p1]^2*sp[p2, p2]^2) - 2*sp[l1, p1]*sp[p1, p1]*
      (sp[p1, p2]^4 - 12*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
       5*sp[p1, p1]^2*sp[p2, p2]^2) + sp[l1, p1]^2*
      (sp[p1, p2]^4 - 30*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
       36*sp[p1, p1]^2*sp[p2, p2]^2) + sp[l1, p2]^2*
      (24*sp[l1, p1]*sp[p1, p1]^2*sp[p2, p2] + sp[l1, p1]^2*
        (sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) - sp[p1, p1]^2*
        (2*sp[p1, p2]^2 + 11*sp[p1, p1]*sp[p2, p2])))))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
  sp[l1, l1]^2*(sp[l1, l1] - 2*sp[l1, p1] + sp[p1, p1])*
  (sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
   11*sp[p1, p1]^2*sp[p2, p2]^2))
