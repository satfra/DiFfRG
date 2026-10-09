(* Created with the Wolfram Language : www.wolfram.com *)
(3*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1, -l1 - p1}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {-l1, p2, l1 - p2}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, 
   {l1 + p1, -l1 + p2, -p1 - p2}]*FunKit`dressing[FunKit`Rdot, {A, A}, 1, 
   {-l1, l1}]*
  (sp[l1, l1]^2*(-(sp[l1, p2]*sp[p1, p1]*(sp[p1, p2] + sp[p2, p2])) + 
     sp[l1, p1]*sp[p1, p2]*(sp[p1, p2] + sp[p2, p2]) + 
     sp[p1, p1]*(sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2])) + 
   sp[l1, p1]*sp[l1, p2]*(-(sp[l1, p1]^2*sp[p1, p2]) + 
     sp[p1, p1]*(sp[l1, p2]^2 + sp[p1, p2]^2 + 
       sp[l1, p2]*(sp[p1, p1] + sp[p1, p2]) - sp[p1, p1]*sp[p2, p2]) + 
     sp[l1, p1]*(sp[l1, p2]*(sp[p1, p1] - sp[p1, p2]) + sp[p1, p2]^2 - 
       sp[p1, p1]*(sp[p1, p2] + 2*sp[p2, p2]))) + 
   sp[l1, l1]*(sp[l1, p1]^2*(-(sp[l1, p2]*sp[p1, p2]) + sp[p1, p2]^2 + 
       sp[p1, p1]*sp[p2, p2] + 2*sp[p1, p2]*sp[p2, p2]) + 
     sp[p1, p1]*(sp[l1, p2]^3 - sp[p1, p2]^3 + sp[l1, p2]^2*
        (sp[p1, p1] + sp[p1, p2]) - sp[l1, p2]*sp[p1, p2]*
        (sp[p1, p1] + sp[p1, p2]) + sp[p1, p1]*sp[p1, p2]*sp[p2, p2]) + 
     sp[l1, p1]*(sp[l1, p2]^2*(sp[p1, p1] - sp[p1, p2]) - sp[p1, p2]^3 + 
       sp[p1, p1]*sp[p1, p2]*(sp[p1, p2] + 2*sp[p2, p2]) - 
       sp[l1, p2]*(sp[p1, p2]^2 + sp[p1, p1]*(3*sp[p1, p2] + 
           2*sp[p2, p2]))))))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 - p1, l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {-l1 + p2, l1 - p2}]*
  sp[l1, l1]*(sp[l1, l1] + 2*sp[l1, p1] + sp[p1, p1])*
  (sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2]))
