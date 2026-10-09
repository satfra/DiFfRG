(* Created with the Wolfram Language : www.wolfram.com *)
(4*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1, l1 + p1, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1, -l1 - p1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p2, l1, -l1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, l1 - p2, -l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
  (sp[l1, l1]^5*sp[p1, p1]*sp[p2, p2]*(3*sp[l1, p2]^2*sp[p1, p1] - 
     6*sp[l1, p1]*sp[l1, p2]*sp[p1, p2] + 
     sp[p2, p2]*(3*sp[l1, p1]^2 + 4*sp[p1, p1]*(sp[p1, p1] - 6*sp[p1, p2] + 
         sp[p2, p2]))) - sp[l1, p1]^2*sp[l1, p2]^2*
    (sp[l1, p2]^4*sp[p1, p1]^2 - 2*sp[l1, p2]^3*sp[p1, p1]*
      (sp[l1, p1]*sp[p1, p2] + sp[p1, p1]*sp[p2, p2]) + 
     sp[p2, p2]^2*(sp[l1, p1]^4 + 2*sp[l1, p1]^3*sp[p1, p1] + 
       2*sp[l1, p1]*sp[p1, p1]*(sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       sp[p1, p1]^2*(sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]^2*(sp[p1, p1]^2 + sp[p1, p2]^2 - 2*sp[p1, p1]*
          sp[p2, p2])) + sp[l1, p2]^2*
      (sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       2*sp[l1, p1]*sp[p1, p1]*(sp[p1, p2]^2 - 2*sp[p1, p1]*sp[p2, p2] + 
         2*sp[p1, p2]*sp[p2, p2]) + sp[p1, p1]^2*(sp[p1, p2]^2 - 
         2*sp[p1, p1]*sp[p2, p2] + sp[p2, p2]^2)) - 
     2*sp[l1, p2]*sp[p2, p2]*(sp[l1, p1]^3*sp[p1, p2] + 
       sp[l1, p1]^2*(sp[p1, p2]^2 + 2*sp[p1, p1]*(sp[p1, p2] - sp[p2, p2])) + 
       sp[p1, p1]^2*(sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]*sp[p1, p1]*(sp[p1, p1]*(sp[p1, p2] - 14*sp[p2, p2]) + 
         sp[p1, p2]*(2*sp[p1, p2] + sp[p2, p2])))) + 
   sp[l1, l1]^4*(-3*sp[l1, p2]^4*sp[p1, p1]^2 + 6*sp[l1, p2]^3*sp[p1, p1]*
      (sp[l1, p1]*sp[p1, p2] - sp[p1, p1]*sp[p2, p2]) + 
     sp[l1, p2]^2*(6*sp[l1, p1]*sp[p1, p1]*(sp[p1, p1] + 2*sp[p1, p2])*
        sp[p2, p2] - 3*sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       2*sp[p1, p1]^2*(sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2] + 
         12*sp[p1, p2]*sp[p2, p2])) - 2*sp[l1, p2]*sp[p2, p2]*
      (-3*sp[l1, p1]^3*sp[p1, p2] + 3*sp[l1, p1]^2*sp[p1, p1]*
        (2*sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]^2*(3*sp[p1, p2]^2 - 
         18*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]*(sp[p1, p1] + 2*sp[p2, p2])) + 
       2*sp[l1, p1]*sp[p1, p1]*(sp[p1, p1]*(2*sp[p1, p2] - 3*sp[p2, p2]) + 
         sp[p1, p2]*(3*sp[p1, p2] + 2*sp[p2, p2]))) + 
     sp[p2, p2]^2*(-3*sp[l1, p1]^4 + 6*sp[l1, p1]^3*sp[p1, p1] + 
       2*sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*(12*sp[p1, p2] - 
           sp[p2, p2])) + 2*sp[l1, p1]*sp[p1, p1]*(2*sp[p1, p1]^2 + 
         3*sp[p1, p2]^2 + sp[p1, p1]*(-18*sp[p1, p2] + sp[p2, p2])) + 
       2*sp[p1, p1]^2*(sp[p1, p1]^2 + sp[p1, p2]^2 - 12*sp[p1, p2]*
          sp[p2, p2] + sp[p2, p2]^2 + sp[p1, p1]*(-12*sp[p1, p2] + 
           7*sp[p2, p2])))) - sp[l1, l1]^2*(sp[l1, p1]^6*sp[p2, p2]^2 + 
     2*sp[l1, p1]^5*sp[p2, p2]*(sp[p2, p2]*(sp[p1, p1] + 2*sp[p2, p2]) - 
       sp[l1, p2]*(sp[p1, p2] + 6*sp[p2, p2])) + 
     sp[l1, p1]^4*(sp[p2, p2]^2*(sp[p1, p1]^2 - 2*sp[p1, p2]^2 - 
         7*sp[p1, p1]*sp[p2, p2]) - 4*sp[l1, p2]*sp[p2, p2]*
        (sp[p1, p1]*(sp[p1, p2] - 3*sp[p2, p2]) + 2*sp[p1, p2]*sp[p2, p2]) + 
       sp[l1, p2]^2*(sp[p1, p2]^2 + 24*sp[p1, p2]*sp[p2, p2] + 
         sp[p2, p2]*(sp[p1, p1] + sp[p2, p2]))) + 
     sp[p1, p1]^2*(sp[l1, p2]^6 - 2*sp[l1, p2]^5*(2*sp[p1, p1] + 
         sp[p2, p2]) - 2*sp[l1, p2]*sp[p1, p2]^2*sp[p2, p2]*
        (sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + sp[p1, p2]^2*sp[p2, p2]^2*
        (sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p2]^4*(-2*sp[p1, p2]^2 + sp[p2, p2]*(-7*sp[p1, p1] + 
           sp[p2, p2])) + sp[l1, p2]^2*(sp[p1, p2]^4 - 
         24*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2 + 5*sp[p1, p1]*sp[p2, p2]^2*
          (sp[p1, p1] + sp[p2, p2]) - sp[p1, p2]^2*sp[p2, p2]*
          (5*sp[p1, p1] + 2*sp[p2, p2])) + sp[l1, p2]^3*
        (-4*sp[p1, p1]^2*sp[p2, p2] + 4*sp[p1, p2]^2*sp[p2, p2] + 
         2*sp[p1, p1]*(sp[p1, p2]^2 + 18*sp[p1, p2]*sp[p2, p2] - 
           2*sp[p2, p2]^2))) - 2*sp[l1, p1]^3*
      (-(sp[l1, p2]*sp[p2, p2]*(-(sp[p1, p1]^2*(sp[p1, p2] - 4*sp[p2, p2])) + 
          2*sp[p1, p2]^2*sp[p2, p2] + sp[p1, p1]*sp[p2, p2]*
           (37*sp[p1, p2] + 2*sp[p2, p2]))) + sp[l1, p2]^3*
        (sp[p1, p2]*(6*sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]*
          (sp[p1, p2] + 6*sp[p2, p2])) + sp[l1, p2]^2*
        (2*sp[p1, p1]^2*sp[p2, p2] - 4*sp[p1, p2]^2*sp[p2, p2] - 
         sp[p1, p1]*(sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] - 
           12*sp[p2, p2]^2)) + sp[p2, p2]^2*(-2*sp[p1, p1]^2*sp[p2, p2] + 
         sp[p1, p2]^2*sp[p2, p2] + 2*sp[p1, p1]*(sp[p1, p2]^2 + 
           9*sp[p1, p2]*sp[p2, p2] - sp[p2, p2]^2))) + 
     2*sp[l1, p1]*sp[p1, p1]*(-(sp[l1, p2]^5*(6*sp[p1, p1] + sp[p1, p2])) + 
       sp[p1, p2]^2*sp[p2, p2]^2*(sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       2*sp[l1, p2]^4*(sp[p1, p1]*(2*sp[p1, p2] - 3*sp[p2, p2]) + 
         sp[p1, p2]*sp[p2, p2]) + sp[l1, p2]*sp[p2, p2]*
        (-2*sp[p1, p2]^4 + 20*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] - 
         6*sp[p1, p1]^2*sp[p2, p2]^2 + 5*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]*
          (sp[p1, p1] + sp[p2, p2])) + sp[l1, p2]^2*(sp[p1, p2]^4 - 
         14*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + sp[p1, p1]*sp[p2, p2]^2*
          (11*sp[p1, p1] + sp[p2, p2]) - 2*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]*
          (sp[p1, p1] + 17*sp[p2, p2])) + sp[l1, p2]^3*
        (2*sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]*sp[p2, p2]^2 + 
         sp[p1, p1]*(2*sp[p1, p2]^2 + 37*sp[p1, p2]*sp[p2, p2] + 
           4*sp[p2, p2]^2))) + sp[l1, p1]^2*
      (sp[l1, p2]^4*(sp[p1, p1]^2 + sp[p1, p2]^2 + sp[p1, p1]*
          (24*sp[p1, p2] + sp[p2, p2])) - 2*sp[l1, p2]*sp[p2, p2]*
        (sp[p1, p2]^4 - 14*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] - 
         2*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]*(17*sp[p1, p1] + sp[p2, p2]) + 
         sp[p1, p1]^2*sp[p2, p2]*(sp[p1, p1] + 11*sp[p2, p2])) + 
       sp[l1, p2]^2*(-2*sp[p1, p1]^3*sp[p2, p2] + sp[p1, p1]^2*
          (sp[p1, p2]^2 - 8*sp[p1, p2]*sp[p2, p2] - 13*sp[p2, p2]^2) + 
         sp[p1, p2]^2*(sp[p1, p2]^2 + sp[p2, p2]^2) - 2*sp[p1, p1]*sp[p2, p2]*
          (11*sp[p1, p2]^2 + 4*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2)) + 
       2*sp[l1, p2]^3*(12*sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]^2*sp[p2, p2] + 
         sp[p1, p1]*(-4*sp[p1, p2]^2 - 6*sp[p1, p2]*sp[p2, p2] + 
           2*sp[p2, p2]^2)) + sp[p2, p2]^2*(sp[p1, p2]^4 + 
         5*sp[p1, p1]^3*sp[p2, p2] - 5*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
         sp[p1, p1]^2*(-2*sp[p1, p2]^2 - 24*sp[p1, p2]*sp[p2, p2] + 
           5*sp[p2, p2]^2)))) + sp[l1, l1]^3*(-6*sp[l1, p1]^5*sp[p2, p2]^2 + 
     sp[l1, p1]^4*sp[p2, p2]*(6*sp[l1, p2]*(2*sp[p1, p2] + sp[p2, p2]) - 
       sp[p2, p2]*(3*sp[p1, p1] + 2*sp[p2, p2])) - 
     2*sp[l1, p1]^3*(sp[l1, p2]*sp[p2, p2]*
        (-3*sp[p1, p1]*(sp[p1, p2] - 4*sp[p2, p2]) - 2*sp[p1, p2]*
          sp[p2, p2]) + 3*sp[l1, p2]^2*(sp[p1, p2]^2 + 
         sp[p1, p1]*sp[p2, p2] + 2*sp[p1, p2]*sp[p2, p2]) + 
       sp[p2, p2]^2*(sp[p1, p1]^2 - 2*sp[p1, p2]^2 - 
         sp[p1, p1]*(18*sp[p1, p2] + sp[p2, p2]))) - 
     2*sp[l1, p1]*sp[p1, p1]*(3*sp[l1, p2]^4*(sp[p1, p1] + 2*sp[p1, p2]) + 
       sp[l1, p2]^3*(-2*sp[p1, p1]*(sp[p1, p2] - 6*sp[p2, p2]) - 
         3*sp[p1, p2]*sp[p2, p2]) - 2*sp[p1, p1]*sp[p2, p2]^2*
        (3*sp[p1, p2]^2 - 9*sp[p1, p2]*sp[p2, p2] + 
         sp[p2, p2]*(sp[p1, p1] + sp[p2, p2])) + sp[l1, p2]^2*
        (2*sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]*sp[p2, p2]*
          (9*sp[p1, p2] + 2*sp[p2, p2]) - sp[p1, p1]*(sp[p1, p2]^2 + 
           32*sp[p1, p2]*sp[p2, p2] - 6*sp[p2, p2]^2)) + 
       sp[l1, p2]*sp[p2, p2]*(sp[p1, p1]^2*(sp[p1, p2] - 2*sp[p2, p2]) + 
         sp[p1, p2]*sp[p2, p2]*(6*sp[p1, p2] + sp[p2, p2]) + 
         2*sp[p1, p1]*(3*sp[p1, p2]^2 - 9*sp[p1, p2]*sp[p2, p2] - 
           sp[p2, p2]^2))) + sp[l1, p1]^2*
      (6*sp[l1, p2]^3*(sp[p1, p2]^2 + sp[p1, p1]*(2*sp[p1, p2] + 
           sp[p2, p2])) + sp[p2, p2]^2*(-sp[p1, p1]^3 + 
         3*sp[p1, p1]^2*(8*sp[p1, p2] - 7*sp[p2, p2]) + 
         sp[p1, p2]^2*sp[p2, p2] + sp[p1, p1]*(15*sp[p1, p2]^2 + 
           24*sp[p1, p2]*sp[p2, p2] - 2*sp[p2, p2]^2)) - 
       2*sp[l1, p2]*sp[p2, p2]*(2*sp[p1, p1]^2*(sp[p1, p2] - 3*sp[p2, p2]) + 
         sp[p1, p2]^2*sp[p2, p2] + sp[p1, p1]*(9*sp[p1, p2]^2 + 
           32*sp[p1, p2]*sp[p2, p2] - 2*sp[p2, p2]^2)) + 
       sp[l1, p2]^2*(3*sp[p1, p1]^2*sp[p2, p2] - 4*sp[p1, p2]^2*sp[p2, p2] + 
         sp[p1, p1]*(-4*sp[p1, p2]^2 + 24*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2))) + sp[p1, p1]^2*(6*sp[l1, p2]^5 - 
       sp[l1, p2]^4*(2*sp[p1, p1] + 3*sp[p2, p2]) - 
       2*sp[l1, p2]^3*(2*sp[p1, p2]^2 + 18*sp[p1, p2]*sp[p2, p2] + 
         (sp[p1, p1] - sp[p2, p2])*sp[p2, p2]) - 4*sp[l1, p2]*sp[p2, p2]^2*
        (sp[p1, p1]^2 + 3*sp[p1, p2]^2 + sp[p1, p1]*(-9*sp[p1, p2] + 
           sp[p2, p2])) + sp[l1, p2]^2*(-2*sp[p1, p1]^2*sp[p2, p2] + 
         sp[p1, p1]*(sp[p1, p2]^2 + 24*sp[p1, p2]*sp[p2, p2] - 
           21*sp[p2, p2]^2) + sp[p2, p2]*(15*sp[p1, p2]^2 + 
           24*sp[p1, p2]*sp[p2, p2] - sp[p2, p2]^2)) + 
       sp[p2, p2]^2*(5*sp[p1, p1]^2*sp[p2, p2] + 3*sp[p1, p2]^2*sp[p2, p2] + 
         sp[p1, p1]*(3*sp[p1, p2]^2 - 24*sp[p1, p2]*sp[p2, p2] + 
           5*sp[p2, p2]^2)))) + sp[l1, l1]*sp[l1, p1]*sp[l1, p2]*
    (2*sp[l1, p1]^5*sp[p2, p2]^2 - 2*sp[l1, p1]^4*sp[p2, p2]*
      (-2*sp[p1, p1]*sp[p2, p2] + sp[l1, p2]*(2*sp[p1, p2] + sp[p2, p2])) - 
     2*sp[p1, p1]^2*(sp[l1, p2]^5 - 2*sp[l1, p2]^4*sp[p2, p2] + 
       sp[l1, p2]^3*sp[p2, p2]*(-9*sp[p1, p1] + sp[p2, p2]) + 
       2*sp[l1, p2]*sp[p1, p2]*sp[p2, p2]*(sp[p1, p2]^2 - 
         7*sp[p1, p1]*sp[p2, p2]) - sp[p1, p2]*sp[p2, p2]^2*
        (sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p2]^2*(-sp[p1, p2]^3 + 3*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + 
         6*sp[p1, p1]*sp[p2, p2]^2)) + sp[l1, p1]^3*
      (2*sp[p1, p1]*(sp[p1, p1] - 9*sp[p2, p2])*sp[p2, p2]^2 + 
       2*sp[l1, p2]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2] + 
         2*sp[p1, p2]*sp[p2, p2]) + sp[l1, p2]*sp[p2, p2]*
        (-3*sp[p1, p2]^2 + sp[p1, p1]*(-8*sp[p1, p2] + 28*sp[p2, p2]))) - 
     2*sp[l1, p1]^2*(sp[p2, p2]^2*(-sp[p1, p2]^3 + 6*sp[p1, p1]^2*
          sp[p2, p2] + 3*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]) + 
       sp[l1, p2]^3*(sp[p1, p2]^2 + sp[p1, p1]*(2*sp[p1, p2] + sp[p2, p2])) + 
       sp[l1, p2]*sp[p2, p2]*(2*sp[p1, p1]^2*(sp[p1, p2] - 10*sp[p2, p2]) + 
         sp[p1, p2]^2*(2*sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]*(3*sp[p1, p2]^2 - 6*sp[p1, p2]*sp[p2, p2] - 
           2*sp[p2, p2]^2)) + sp[l1, p2]^2*(4*sp[p1, p1]^2*sp[p2, p2] - 
         sp[p1, p2]^2*(sp[p1, p2] + 2*sp[p2, p2]) - 2*sp[p1, p1]*
          (sp[p1, p2]^2 + sp[p1, p2]*sp[p2, p2] - 2*sp[p2, p2]^2))) + 
     sp[l1, p1]*sp[p1, p1]*(2*sp[l1, p2]^4*(sp[p1, p1] + 2*sp[p1, p2]) + 
       4*sp[p1, p2]*sp[p2, p2]^2*(sp[p1, p2]^2 - 7*sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p2]^3*(-3*sp[p1, p2]^2 + 28*sp[p1, p1]*sp[p2, p2] - 
         8*sp[p1, p2]*sp[p2, p2]) + sp[l1, p2]^2*
        (-4*sp[p1, p1]^2*sp[p2, p2] + 2*sp[p1, p1]*(sp[p1, p2]^2 - 
           6*sp[p1, p2]*sp[p2, p2] - 20*sp[p2, p2]^2) + 
         2*sp[p1, p2]*(2*sp[p1, p2]^2 + 3*sp[p1, p2]*sp[p2, p2] + 
           2*sp[p2, p2]^2)) + sp[l1, p2]*sp[p2, p2]*
        (11*sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]^2*(8*sp[p1, p2] + 
           3*sp[p2, p2]) + sp[p1, p1]*(-3*sp[p1, p2]^2 + 56*sp[p1, p2]*
            sp[p2, p2] + 11*sp[p2, p2]^2))))))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 - p1, l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p2, -l1 + p2}]*
  sp[l1, l1]^2*(sp[l1, l1] + 2*sp[l1, p1] + sp[p1, p1])*
  (sp[l1, l1] - 2*sp[l1, p2] + sp[p2, p2])*
  (sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
   11*sp[p1, p1]^2*sp[p2, p2]^2))
