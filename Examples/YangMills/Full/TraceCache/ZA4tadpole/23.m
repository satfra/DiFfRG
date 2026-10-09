(* Created with the Wolfram Language : www.wolfram.com *)
(-2*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p2, l1, -l1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, l1 - p2, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, {-p1, p1, l1, -l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
  (4*sp[l1, l1]^4*sp[p1, p1]^2*sp[p2, p2]^2 + sp[l1, l1]^3*sp[p1, p1]*
    (sp[l1, p2]^2*(2*sp[p1, p2]^2 - 3*sp[p1, p1]*sp[p2, p2]) + 
     2*sp[l1, p2]*sp[p2, p2]*(-3*sp[l1, p1]*sp[p1, p2] - 9*sp[p1, p2]^2 + 
       5*sp[p1, p1]*sp[p2, p2]) + sp[p2, p2]^2*(3*sp[l1, p1]^2 + 
       36*sp[l1, p1]*sp[p1, p2] + 11*sp[p1, p2]^2 + 
       13*sp[p1, p1]*sp[p2, p2])) - sp[l1, p2]^2*(sp[l1, p2]^4*sp[p1, p1]^2 + 
     sp[l1, p1]^2*sp[p2, p2]^2*(sp[l1, p1]^2 + sp[p1, p2]^2 - 
       2*sp[p1, p1]*sp[p2, p2]) - 2*sp[l1, p2]^3*sp[p1, p1]*
      (sp[l1, p1]*sp[p1, p2] + sp[p1, p1]*sp[p2, p2]) - 
     2*sp[l1, p1]*sp[l1, p2]*sp[p2, p2]*(sp[l1, p1]^2*sp[p1, p2] + 
       sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + sp[l1, p1]*(sp[p1, p2]^2 - 
         2*sp[p1, p1]*sp[p2, p2])) + sp[l1, p2]^2*
      (4*sp[l1, p1]*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + 
       sp[p1, p1]^2*sp[p2, p2]^2 + sp[l1, p1]^2*(sp[p1, p2]^2 + 
         sp[p1, p1]*sp[p2, p2]))) - sp[l1, l1]^2*(sp[l1, p2]^4*sp[p1, p1]^2 + 
     2*sp[l1, p2]^3*sp[p1, p1]*(-(sp[l1, p1]*sp[p1, p2]) + 2*sp[p1, p2]^2 + 
       6*sp[p1, p1]*sp[p2, p2]) - 2*sp[l1, p2]*sp[p2, p2]*
      (sp[l1, p1]^3*sp[p1, p2] + sp[p1, p2]^4 - 12*sp[l1, p1]^2*sp[p1, p1]*
        sp[p2, p2] - 41*sp[l1, p1]*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] - 
       12*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 5*sp[p1, p1]^2*sp[p2, p2]^2) + 
     sp[l1, p2]^2*(sp[p1, p2]^4 - 12*sp[l1, p1]*sp[p1, p1]*sp[p1, p2]*
        sp[p2, p2] - 30*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
       36*sp[p1, p1]^2*sp[p2, p2]^2 + sp[l1, p1]^2*(sp[p1, p2]^2 + 
         sp[p1, p1]*sp[p2, p2])) + sp[p2, p2]^2*(sp[l1, p1]^4 + 
       sp[p1, p2]^4 - 36*sp[l1, p1]*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] - 
       3*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] - 4*sp[p1, p1]^2*sp[p2, p2]^2 - 
       sp[l1, p1]^2*(2*sp[p1, p2]^2 + 11*sp[p1, p1]*sp[p2, p2]))) + 
   sp[l1, l1]*sp[l1, p2]*(2*sp[l1, p2]^4*sp[p1, p1]^2 - 
     sp[l1, p2]^3*sp[p1, p1]*(4*sp[l1, p1]*sp[p1, p2] + sp[p1, p2]^2 - 
       24*sp[p1, p1]*sp[p2, p2]) + 2*sp[l1, p1]*sp[p2, p2]^2*
      (sp[l1, p1]^3 + sp[p1, p2]^3 - 9*sp[l1, p1]*sp[p1, p1]*sp[p2, p2] - 
       3*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]) + 2*sp[l1, p2]^2*
      (sp[l1, p1]*sp[p1, p2]^3 + sp[p1, p1]*sp[p2, p2]*
        (sp[p1, p2]^2 - 6*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2])) - 
     sp[l1, p2]*sp[p2, p2]*(4*sp[l1, p1]^3*sp[p1, p2] + 
       3*sp[l1, p1]^2*(sp[p1, p2]^2 - 10*sp[p1, p1]*sp[p2, p2]) + 
       sp[p1, p1]*sp[p2, p2]*(sp[p1, p2]^2 + 3*sp[p1, p1]*sp[p2, p2]) + 
       4*sp[l1, p1]*(sp[p1, p2]^3 - 3*sp[p1, p1]*sp[p1, p2]*sp[p2, p2])))))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p2, -l1 + p2}]*
  sp[l1, l1]^2*(sp[l1, l1] - 2*sp[l1, p2] + sp[p2, p2])*
  (sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
   11*sp[p1, p1]^2*sp[p2, p2]^2))
