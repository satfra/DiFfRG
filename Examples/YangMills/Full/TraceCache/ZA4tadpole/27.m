(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1, l1 + p1, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p2, l1, -l1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {p2, p1, l1 - p2, -l1 - p1}]*FunKit`dressing[FunKit`Rdot, {A, A}, 1, 
   {-l1, l1}]*(-4*sp[l1, l1]^4*sp[p1, p1]*sp[p2, p2]*
    (sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
   sp[l1, l1]^3*(sp[l1, p2]^2*sp[p1, p1]*(2*sp[p1, p2]^2 + 
       5*sp[p1, p1]*sp[p2, p2]) + sp[l1, p2]*
      (-2*sp[l1, p1]*(sp[p1, p2]^3 + 2*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]) + 
       sp[p1, p1]*(9*sp[p1, p2]^3 - 27*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + 
         22*sp[p1, p2]^2*sp[p2, p2] + 12*sp[p1, p1]*sp[p2, p2]^2)) - 
     sp[p2, p2]*(-(sp[l1, p1]^2*(2*sp[p1, p2]^2 + 5*sp[p1, p1]*sp[p2, p2])) + 
       sp[l1, p1]*(9*sp[p1, p2]^3 + sp[p1, p1]*sp[p1, p2]*
          (22*sp[p1, p2] - 27*sp[p2, p2]) + 12*sp[p1, p1]^2*sp[p2, p2]) + 
       2*sp[p1, p1]*(8*sp[p1, p1]^2*sp[p2, p2] + sp[p1, p2]^2*
          (-3*sp[p1, p2] + 4*sp[p2, p2]) + sp[p1, p1]*(4*sp[p1, p2]^2 - 
           37*sp[p1, p2]*sp[p2, p2] + 8*sp[p2, p2]^2)))) + 
   sp[l1, p1]*sp[l1, p2]*(sp[l1, p2]^4*sp[p1, p1]^2 + 
     sp[l1, p2]^3*sp[p1, p1]*(-2*sp[l1, p1]*sp[p1, p2] + 5*sp[p1, p2]^2 + 
       sp[p1, p1]*(-4*sp[p1, p2] + 6*sp[p2, p2])) + 
     sp[l1, p2]^2*(sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       sp[p1, p1]*(5*sp[p1, p2]^3 + 19*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] - 
         5*sp[p1, p2]^2*sp[p2, p2] - sp[p1, p1]*sp[p2, p2]*
          (17*sp[p1, p1] + 2*sp[p2, p2])) + sp[l1, p1]*
        (-5*sp[p1, p2]^3 - 32*sp[p1, p1]^2*sp[p2, p2] + sp[p1, p1]*sp[p1, p2]*
          (9*sp[p1, p2] + 11*sp[p2, p2]))) + 
     sp[l1, p2]*(-2*sp[l1, p1]^3*sp[p1, p2]*sp[p2, p2] + 
       sp[l1, p1]^2*(5*sp[p1, p2]^3 - 11*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] - 
         9*sp[p1, p2]^2*sp[p2, p2] + 32*sp[p1, p1]*sp[p2, p2]^2) + 
       sp[p1, p1]*(5*sp[p1, p2]^4 + 27*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] - 
         5*sp[p1, p2]^3*sp[p2, p2] + 98*sp[p1, p1]^2*sp[p2, p2]^2 - 
         9*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2) + 
       sp[l1, p1]*(5*sp[p1, p2]^3*(sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]^2*sp[p2, p2]*(-5*sp[p1, p2] + 192*sp[p2, p2]) + 
         sp[p1, p1]*sp[p1, p2]*(5*sp[p1, p2]^2 + 39*sp[p1, p2]*sp[p2, p2] - 
           5*sp[p2, p2]^2))) - sp[p2, p2]*(-(sp[l1, p1]^4*sp[p2, p2]) + 
       sp[l1, p1]^3*(5*sp[p1, p2]^2 + 6*sp[p1, p1]*sp[p2, p2] - 
         4*sp[p1, p2]*sp[p2, p2]) + sp[p1, p1]*(5*sp[p1, p2]^4 + 
         18*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 49*sp[p1, p1]^2*
          sp[p2, p2]^2) + sp[l1, p1]*(5*sp[p1, p2]^4 + 
         sp[p1, p1]*sp[p1, p2]^2*(-5*sp[p1, p2] + 27*sp[p2, p2]) + 
         sp[p1, p1]^2*sp[p2, p2]*(-9*sp[p1, p2] + 98*sp[p2, p2])) + 
       sp[l1, p1]^2*(-5*sp[p1, p2]^3 + 2*sp[p1, p1]^2*sp[p2, p2] + 
         sp[p1, p1]*(5*sp[p1, p2]^2 - 19*sp[p1, p2]*sp[p2, p2] + 
           17*sp[p2, p2]^2)))) + sp[l1, l1]^2*(-(sp[l1, p2]^4*sp[p1, p1]^2) + 
     sp[l1, p2]^3*sp[p1, p1]*(2*sp[l1, p1]*sp[p1, p2] - 9*sp[p1, p2]^2 + 
       sp[p1, p1]*(4*sp[p1, p2] - 13*sp[p2, p2])) - 
     sp[l1, p2]^2*(sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]*(-9*sp[p1, p2]^3 - 36*sp[p1, p1]^2*sp[p2, p2] + 
         sp[p1, p1]*sp[p1, p2]*(4*sp[p1, p2] + 3*sp[p2, p2])) - 
       sp[p1, p1]*(16*sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]^2*
          (14*sp[p1, p2] + 15*sp[p2, p2]) + sp[p1, p1]*(3*sp[p1, p2]^2 + 
           14*sp[p1, p2]*sp[p2, p2] + 10*sp[p2, p2]^2))) + 
     sp[p2, p2]*(-(sp[l1, p1]^4*sp[p2, p2]) + sp[l1, p1]^3*
        (9*sp[p1, p2]^2 + 13*sp[p1, p1]*sp[p2, p2] - 4*sp[p1, p2]*
          sp[p2, p2]) + sp[l1, p1]^2*(10*sp[p1, p1]^2*sp[p2, p2] + 
         sp[p1, p2]^2*(-14*sp[p1, p2] + 3*sp[p2, p2]) + 
         sp[p1, p1]*(-15*sp[p1, p2]^2 + 14*sp[p1, p2]*sp[p2, p2] + 
           16*sp[p2, p2]^2)) - sp[l1, p1]*(2*sp[p1, p1]^3*sp[p2, p2] + 
         sp[p1, p2]^3*(sp[p1, p2] + 5*sp[p2, p2]) - 5*sp[p1, p1]*sp[p1, p2]*
          (3*sp[p1, p2]^2 - 7*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2) + 
         sp[p1, p1]^2*(12*sp[p1, p2]^2 - 129*sp[p1, p2]*sp[p2, p2] + 
           28*sp[p2, p2]^2)) + sp[p1, p1]*
        (-(sp[p1, p2]^3*(sp[p1, p2] - 14*sp[p2, p2])) + 
         2*sp[p1, p1]^2*(29*sp[p1, p2] - 14*sp[p2, p2])*sp[p2, p2] + 
         sp[p1, p1]*sp[p1, p2]*(14*sp[p1, p2]^2 - 15*sp[p1, p2]*sp[p2, p2] + 
           58*sp[p2, p2]^2))) + sp[l1, p2]*
      (2*sp[l1, p1]^3*sp[p1, p2]*sp[p2, p2] + sp[l1, p1]^2*
        (-9*sp[p1, p2]^3 + 3*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + 
         4*sp[p1, p2]^2*sp[p2, p2] - 36*sp[p1, p1]*sp[p2, p2]^2) + 
       sp[l1, p1]*(sp[p1, p2]^3*(sp[p1, p2] + 7*sp[p2, p2]) - 
         sp[p1, p1]^2*sp[p2, p2]*(41*sp[p1, p2] + 29*sp[p2, p2]) + 
         sp[p1, p1]*sp[p1, p2]*(7*sp[p1, p2]^2 + 76*sp[p1, p2]*sp[p2, p2] - 
           41*sp[p2, p2]^2)) + sp[p1, p1]*(sp[p1, p1]^2*sp[p2, p2]*
          (-15*sp[p1, p2] + 28*sp[p2, p2]) + sp[p1, p2]^2*
          (sp[p1, p2]^2 - 15*sp[p1, p2]*sp[p2, p2] + 12*sp[p2, p2]^2) + 
         sp[p1, p1]*(5*sp[p1, p2]^3 + 35*sp[p1, p2]^2*sp[p2, p2] - 
           129*sp[p1, p2]*sp[p2, p2]^2 + 2*sp[p2, p2]^3)))) - 
   sp[l1, l1]*(sp[l1, p1]^5*sp[p2, p2]^2 - sp[l1, p1]^4*sp[p2, p2]*
      (5*sp[p1, p2]^2 + 6*sp[p1, p1]*sp[p2, p2] - 4*sp[p1, p2]*sp[p2, p2] + 
       sp[l1, p2]*(2*sp[p1, p2] + sp[p2, p2])) + 
     sp[p1, p1]*(-(sp[l1, p2]^5*sp[p1, p1]) + sp[l1, p2]^4*
        (-5*sp[p1, p2]^2 + sp[p1, p1]*(4*sp[p1, p2] - 6*sp[p2, p2])) - 
       sp[p1, p2]*sp[p2, p2]*(5*sp[p1, p2]^4 + 18*sp[p1, p1]*sp[p1, p2]^2*
          sp[p2, p2] + 49*sp[p1, p1]^2*sp[p2, p2]^2) + 
       sp[l1, p2]*sp[p1, p2]*(5*sp[p1, p2]^4 + 27*sp[p1, p1]*sp[p1, p2]^2*
          sp[p2, p2] - 5*sp[p1, p2]^3*sp[p2, p2] + 98*sp[p1, p1]^2*
          sp[p2, p2]^2 - 9*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2) + 
       sp[l1, p2]^3*(28*sp[p1, p1]^2*sp[p2, p2] + 5*sp[p1, p2]^2*
          (-sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]*(9*sp[p1, p2]^2 - 
           19*sp[p1, p2]*sp[p2, p2] + 2*sp[p2, p2]^2)) + 
       sp[l1, p2]^2*(5*sp[p1, p2]^3*(sp[p1, p2] + sp[p2, p2]) - 
         2*sp[p1, p1]^2*sp[p2, p2]*(3*sp[p1, p2] + 14*sp[p2, p2]) + 
         sp[p1, p1]*sp[p1, p2]*(10*sp[p1, p2]^2 + 15*sp[p1, p2]*sp[p2, p2] + 
           9*sp[p2, p2]^2))) + sp[l1, p1]^3*
      (sp[l1, p2]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2] + 
         2*sp[p1, p2]*sp[p2, p2]) + sp[l1, p2]*(5*sp[p1, p2]^3 + 
         9*sp[p1, p2]^2*sp[p2, p2] + 56*sp[p1, p1]*sp[p2, p2]^2 - 
         sp[p1, p2]*sp[p2, p2]*(11*sp[p1, p1] + 4*sp[p2, p2])) - 
       sp[p2, p2]*(2*sp[p1, p1]^2*sp[p2, p2] + sp[p1, p2]^2*
          (-5*sp[p1, p2] + 9*sp[p2, p2]) + sp[p1, p1]*(5*sp[p1, p2]^2 - 
           19*sp[p1, p2]*sp[p2, p2] + 28*sp[p2, p2]^2))) + 
     sp[l1, p1]*(sp[l1, p2]^4*sp[p1, p1]*(sp[p1, p1] + 2*sp[p1, p2]) + 
       sp[l1, p2]^3*(5*sp[p1, p2]^3 - 4*sp[p1, p1]^2*(sp[p1, p2] - 
           14*sp[p2, p2]) + sp[p1, p1]*sp[p1, p2]*(9*sp[p1, p2] - 
           11*sp[p2, p2])) + sp[p1, p2]*sp[p2, p2]*(-5*sp[p1, p2]^4 + 
         sp[p1, p1]*sp[p1, p2]^2*(5*sp[p1, p2] - 27*sp[p2, p2]) + 
         sp[p1, p1]^2*(9*sp[p1, p2] - 98*sp[p2, p2])*sp[p2, p2]) + 
       sp[l1, p2]^2*(5*sp[p1, p2]^3*(sp[p1, p2] - sp[p2, p2]) - 
         17*sp[p1, p1]^3*sp[p2, p2] - sp[p1, p1]^2*sp[p2, p2]*
          (13*sp[p1, p2] + 132*sp[p2, p2]) + sp[p1, p1]*sp[p1, p2]*
          (7*sp[p1, p2]^2 + 18*sp[p1, p2]*sp[p2, p2] + 5*sp[p2, p2]^2)) + 
       sp[l1, p2]*(5*sp[p1, p2]^4*(sp[p1, p2] - sp[p2, p2]) + 
         47*sp[p1, p1]^3*sp[p2, p2]^2 + sp[p1, p1]*sp[p1, p2]^2*
          (-5*sp[p1, p2]^2 + 42*sp[p1, p2]*sp[p2, p2] - 12*sp[p2, p2]^2) + 
         sp[p1, p1]^2*sp[p2, p2]*(-12*sp[p1, p2]^2 + 209*sp[p1, p2]*
            sp[p2, p2] + 47*sp[p2, p2]^2))) - 
     sp[l1, p1]^2*(sp[l1, p2]^3*(sp[p1, p2]^2 + sp[p1, p1]*
          (2*sp[p1, p2] + sp[p2, p2])) + sp[p2, p2]*
        (-5*sp[p1, p2]^3*(sp[p1, p2] + 2*sp[p2, p2]) + 
         sp[p1, p1]^2*sp[p2, p2]*(-9*sp[p1, p2] + 28*sp[p2, p2]) + 
         sp[p1, p1]*sp[p1, p2]*(-5*sp[p1, p2]^2 - 15*sp[p1, p2]*sp[p2, p2] + 
           6*sp[p2, p2]^2)) + sp[l1, p2]^2*
        (sp[p1, p2]^2*(26*sp[p1, p2] - 9*sp[p2, p2]) + 32*sp[p1, p1]^2*
          sp[p2, p2] + sp[p1, p1]*(-9*sp[p1, p2]^2 - 16*sp[p1, p2]*
            sp[p2, p2] + 32*sp[p2, p2]^2)) + sp[l1, p2]*
        (sp[p1, p1]^2*(5*sp[p1, p2] - 132*sp[p2, p2])*sp[p2, p2] + 
         sp[p1, p2]^3*(5*sp[p1, p2] + 7*sp[p2, p2]) - 
         sp[p1, p1]*(5*sp[p1, p2]^3 - 18*sp[p1, p2]^2*sp[p2, p2] + 
           13*sp[p1, p2]*sp[p2, p2]^2 + 17*sp[p2, p2]^3))))))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 - p1, l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p2, -l1 + p2}]*
  sp[l1, l1]*(sp[l1, l1] + 2*sp[l1, p1] + sp[p1, p1])*
  (sp[l1, l1] - 2*sp[l1, p2] + sp[p2, p2])*
  (sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
   11*sp[p1, p1]^2*sp[p2, p2]^2))
