(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, {-p1, p1, l1, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, {-p2, p2, l1, -l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*sp[p1, p1]^2*
  sp[p2, p2]^2*(11 - (sp[p1, p2]^4 - 26*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2])/
    (sp[p1, p1]^2*sp[p2, p2]^2) - (sp[l1, p2]^4*sp[p1, p1]^2 - 
     2*sp[l1, p1]*sp[l1, p2]^3*sp[p1, p1]*sp[p1, p2] + 
     sp[l1, p1]^2*sp[p2, p2]*(sp[l1, p1]^2*sp[p2, p2] + 
       sp[l1, l1]*(sp[p1, p2]^2 - 26*sp[p1, p1]*sp[p2, p2])) - 
     2*sp[l1, p1]*sp[l1, p2]*sp[p1, p2]*(sp[l1, p1]^2*sp[p2, p2] + 
       sp[l1, l1]*(sp[p1, p2]^2 - 2*sp[p1, p1]*sp[p2, p2])) + 
     sp[l1, p2]^2*(sp[l1, l1]*sp[p1, p1]*(sp[p1, p2]^2 - 
         26*sp[p1, p1]*sp[p2, p2]) + sp[l1, p1]^2*(sp[p1, p2]^2 + 
         sp[p1, p1]*sp[p2, p2])))/(sp[l1, l1]^2*sp[p1, p1]^2*sp[p2, p2]^2)))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]^2*
  (sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
   11*sp[p1, p1]^2*sp[p2, p2]^2))
