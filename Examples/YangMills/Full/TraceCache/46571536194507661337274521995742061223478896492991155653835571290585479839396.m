(* Created with the Wolfram Language : www.wolfram.com *)
-((FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1, l1, -l1 + p1}]*
   FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
   FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
   (2*(-7*(sp[l1, l1] + sp[p1, p1]) + (sp[l1, l1] - sp[p1, p1])^2/
       (sp[l1, l1] - 2*sp[l1, p1] + sp[p1, p1])) + 
    sp[l1, p1]*(-4 + (4*sp[l1, p1]*(3*sp[l1, l1]^2 + sp[l1, p1]^2 - 
         6*sp[l1, p1]*sp[p1, p1] + 3*sp[p1, p1]^2 + 
         sp[l1, l1]*(-6*sp[l1, p1] + 5*sp[p1, p1])))/(sp[l1, l1]*sp[p1, p1]*
        (sp[l1, l1] - 2*sp[l1, p1] + sp[p1, p1])))))/
  (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]^2*
   FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 + p1, l1 - p1}]))
