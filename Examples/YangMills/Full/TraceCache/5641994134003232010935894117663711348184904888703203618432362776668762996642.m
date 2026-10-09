(* Created with the Wolfram Language : www.wolfram.com *)
(27*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
   {-p1 - p2, l1, -l1 + p1 + p2}]*FunKit`dressing[FunKit`GammaN, 
   {A, A, A, A}, 1, {p2, p1, l1 - p1 - p2, -l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
  (3*sp[l1, l1]^2*(-(sp[p1, p1]^2*sp[p1, p2]*(sp[p1, p2] - 3*sp[p2, p2])) + 
     sp[p1, p1]^3*sp[p2, p2] - sp[p1, p2]^2*(4*sp[p1, p2]^2 + 
       3*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) + 
     sp[p1, p1]*(-3*sp[p1, p2]^3 + 4*sp[p1, p2]^2*sp[p2, p2] + 
       3*sp[p1, p2]*sp[p2, p2]^2 + sp[p2, p2]^3)) + 
   (sp[l1, p1] + sp[l1, p2])*
    (-(sp[l1, p2]^2*(sp[p1, p1]^3 + 3*sp[p1, p1]^2*sp[p1, p2] + 
        4*sp[p1, p2]^2*(2*sp[p1, p2] + sp[p2, p2]) + 
        sp[p1, p1]*(4*sp[p1, p2]^2 - 5*sp[p1, p2]*sp[p2, p2] - 
          3*sp[p2, p2]^2))) + sp[l1, p1]*
      (-(sp[l1, p1]*(8*sp[p1, p2]^3 + sp[p1, p1]*sp[p1, p2]*
           (4*sp[p1, p2] - 5*sp[p2, p2]) - 3*sp[p1, p1]^2*sp[p2, p2] + 
          4*sp[p1, p2]^2*sp[p2, p2] + 3*sp[p1, p2]*sp[p2, p2]^2 + 
          sp[p2, p2]^3)) - 2*(sp[p1, p1] + 2*sp[p1, p2])*
        (sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]^2*(2*sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]*(-sp[p1, p2]^2 + 2*sp[p1, p2]*sp[p2, p2] + 
           sp[p2, p2]^2))) + 2*sp[l1, p2]*((2*sp[p1, p2] + sp[p2, p2])*
        (-(sp[p1, p1]^2*sp[p2, p2]) + sp[p1, p2]^2*(2*sp[p1, p2] + 
           sp[p2, p2]) + sp[p1, p1]*(sp[p1, p2]^2 - 2*sp[p1, p2]*sp[p2, p2] - 
           sp[p2, p2]^2)) + sp[l1, p1]*
        (sp[p1, p1]^2*(sp[p1, p2] + 2*sp[p2, p2]) + 
         sp[p1, p2]*(-4*sp[p1, p2]^2 + sp[p1, p2]*sp[p2, p2] + 
           sp[p2, p2]^2) + sp[p1, p1]*(sp[p1, p2]^2 + 8*sp[p1, p2]*
            sp[p2, p2] + 2*sp[p2, p2]^2)))) + 
   sp[l1, l1]*(sp[l1, p1]^2*(8*sp[p1, p2]^3 + sp[p1, p1]*sp[p1, p2]*
        (4*sp[p1, p2] - 5*sp[p2, p2]) - 3*sp[p1, p1]^2*sp[p2, p2] + 
       4*sp[p1, p2]^2*sp[p2, p2] + 3*sp[p1, p2]*sp[p2, p2]^2 + 
       sp[p2, p2]^3) + sp[l1, p2]^2*(sp[p1, p1]^3 + 
       3*sp[p1, p1]^2*sp[p1, p2] + 4*sp[p1, p2]^2*(2*sp[p1, p2] + 
         sp[p2, p2]) + sp[p1, p1]*(4*sp[p1, p2]^2 - 5*sp[p1, p2]*sp[p2, p2] - 
         3*sp[p2, p2]^2)) + sp[l1, p1]*(-3*sp[p1, p1]^3*sp[p2, p2] + 
       sp[p1, p1]^2*(3*sp[p1, p2]^2 - 7*sp[p1, p2]*sp[p2, p2] + 
         2*sp[p2, p2]^2) + sp[p1, p2]^2*(12*sp[p1, p2]^2 + 
         11*sp[p1, p2]*sp[p2, p2] + 5*sp[p2, p2]^2) + 
       sp[p1, p1]*(7*sp[p1, p2]^3 - 14*sp[p1, p2]^2*sp[p2, p2] - 
         11*sp[p1, p2]*sp[p2, p2]^2 - 5*sp[p2, p2]^3)) + 
     2*(sp[p1, p1]^4*sp[p2, p2] + sp[p1, p1]^3*(-sp[p1, p2]^2 + 
         5*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) - 
       sp[p1, p2]^2*(8*sp[p1, p2]^3 + 10*sp[p1, p2]^2*sp[p2, p2] + 
         5*sp[p1, p2]*sp[p2, p2]^2 + sp[p2, p2]^3) + 
       sp[p1, p1]^2*(-5*sp[p1, p2]^3 + 9*sp[p1, p2]^2*sp[p2, p2] + 
         6*sp[p1, p2]*sp[p2, p2]^2 + sp[p2, p2]^3) + 
       sp[p1, p1]*(-10*sp[p1, p2]^4 + 2*sp[p1, p2]^3*sp[p2, p2] + 
         9*sp[p1, p2]^2*sp[p2, p2]^2 + 5*sp[p1, p2]*sp[p2, p2]^3 + 
         sp[p2, p2]^4)) - sp[l1, p2]*(5*sp[p1, p1]^3*sp[p2, p2] + 
       sp[p1, p1]^2*(-5*sp[p1, p2]^2 + 11*sp[p1, p2]*sp[p2, p2] - 
         2*sp[p2, p2]^2) - sp[p1, p2]^2*(12*sp[p1, p2]^2 + 
         7*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2) + 
       sp[p1, p1]*(-11*sp[p1, p2]^3 + 14*sp[p1, p2]^2*sp[p2, p2] + 
         7*sp[p1, p2]*sp[p2, p2]^2 + 3*sp[p2, p2]^3) + 
       2*sp[l1, p1]*(sp[p1, p1]^2*(sp[p1, p2] + 2*sp[p2, p2]) + 
         sp[p1, p2]*(-4*sp[p1, p2]^2 + sp[p1, p2]*sp[p2, p2] + 
           sp[p2, p2]^2) + sp[p1, p1]*(sp[p1, p2]^2 + 8*sp[p1, p2]*
            sp[p2, p2] + 2*sp[p2, p2]^2))))))/
 (4*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {-l1 + p1 + p2, l1 - p1 - p2}]*sp[l1, l1]*(sp[l1, l1] - 2*sp[l1, p1] - 
   2*sp[l1, p2] + sp[p1, p1] + 2*sp[p1, p2] + sp[p2, p2])*
  (-sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2])*(sp[p1, p1]^2 + 4*sp[p1, p2]^2 + 
   sp[p1, p1]*(2*sp[p1, p2] - sp[p2, p2]) + 2*sp[p1, p2]*sp[p2, p2] + 
   sp[p2, p2]^2))
