(* Created with the Wolfram Language : www.wolfram.com *)
(6*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, -l1, l1 - p1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p1 - p2, -l1 + p1, l1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, -l1 - p2, l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {l1, -l1}]*
  (3*sp[l1, l1]^4*(sp[l1, p2]*sp[p1, p1]*(sp[p1, p1]*sp[p1, p2] + 
       2*sp[p1, p2]^2 + 2*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) - 
     sp[p2, p2]*(sp[p1, p1]*(-sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]*(sp[p1, p1]^2 + 2*sp[p1, p1]*sp[p1, p2] + 
         sp[p1, p2]*(2*sp[p1, p2] + sp[p2, p2])))) + 
   sp[l1, l1]^3*(3*sp[l1, p1]^3*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] + 
     sp[l1, p1]^2*(-3*sp[l1, p2]*(sp[p1, p1] + sp[p1, p2])*
        (sp[p1, p2] - sp[p2, p2]) + 3*sp[p2, p2]*(2*sp[p1, p1]^2 + 
         sp[p1, p1]*(5*sp[p1, p2] + sp[p2, p2]) + sp[p1, p2]*
          (5*sp[p1, p2] + 3*sp[p2, p2]))) + 
     sp[p1, p1]*(-3*sp[l1, p2]^3*(sp[p1, p2] + sp[p2, p2]) + 
       3*sp[l1, p2]^2*(5*sp[p1, p2]^2 + 5*sp[p1, p2]*sp[p2, p2] + 
         2*sp[p2, p2]^2 + sp[p1, p1]*(3*sp[p1, p2] + sp[p2, p2])) + 
       sp[p2, p2]*(-10*sp[p1, p1]^2*sp[p2, p2] + sp[p1, p2]^2*
          (9*sp[p1, p2] + 10*sp[p2, p2]) + sp[p1, p1]*(10*sp[p1, p2]^2 - 
           9*sp[p1, p2]*sp[p2, p2] - 10*sp[p2, p2]^2)) + 
       sp[l1, p2]*(8*sp[p1, p1]^2*sp[p1, p2] + 10*sp[p1, p2]^3 + 
         32*sp[p1, p2]^2*sp[p2, p2] + 21*sp[p1, p2]*sp[p2, p2]^2 + 
         8*sp[p2, p2]^3 + sp[p1, p1]*(21*sp[p1, p2]^2 + 24*sp[p1, p2]*
            sp[p2, p2] + 2*sp[p2, p2]^2))) - 
     sp[l1, p1]*(3*sp[l1, p2]^2*(sp[p1, p1] - sp[p1, p2])*
        (sp[p1, p2] + sp[p2, p2]) + 3*sp[l1, p2]*
        (sp[p1, p1]^2*(2*sp[p1, p2] + 3*sp[p2, p2]) + 
         sp[p1, p2]*(sp[p1, p2]^2 + 5*sp[p1, p2]*sp[p2, p2] + 
           2*sp[p2, p2]^2) + sp[p1, p1]*(5*sp[p1, p2]^2 + 
           11*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2)) + 
       sp[p2, p2]*(8*sp[p1, p1]^3 + sp[p1, p1]^2*(21*sp[p1, p2] + 
           2*sp[p2, p2]) + 8*sp[p1, p1]*sp[p1, p2]*(4*sp[p1, p2] + 
           3*sp[p2, p2]) + sp[p1, p2]*(10*sp[p1, p2]^2 + 21*sp[p1, p2]*
            sp[p2, p2] + 8*sp[p2, p2]^2)))) + sp[l1, p1]*sp[l1, p2]*
    (sp[l1, p1]^4*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] - 
     sp[l1, p1]^3*(sp[l1, p2]*sp[p1, p2]*(sp[p1, p1] + sp[p1, p2]) + 
       sp[p2, p2]*(6*sp[p1, p1]^2 + sp[p1, p2]*sp[p2, p2] + 
         5*sp[p1, p1]*(2*sp[p1, p2] + sp[p2, p2]))) - 
     sp[l1, p1]^2*(-2*sp[l1, p2]^2*(sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2]) + 
       sp[p2, p2]*(-3*sp[p1, p1]^3 + 3*sp[p1, p1]^2*sp[p2, p2] + 
         sp[p1, p2]*sp[p2, p2]*(5*sp[p1, p2] + 3*sp[p2, p2]) + 
         sp[p1, p1]*(9*sp[p1, p2]^2 + 21*sp[p1, p2]*sp[p2, p2] + 
           10*sp[p2, p2]^2)) + sp[l1, p2]*
        (sp[p1, p1]^2*(-6*sp[p1, p2] + 7*sp[p2, p2]) + 
         sp[p1, p2]*(-2*sp[p1, p2]^2 + 5*sp[p1, p2]*sp[p2, p2] + 
           5*sp[p2, p2]^2) + sp[p1, p1]*(-12*sp[p1, p2]^2 + 
           19*sp[p1, p2]*sp[p2, p2] + 16*sp[p2, p2]^2))) + 
     sp[p1, p1]*(sp[l1, p2]^4*(sp[p1, p2] + sp[p2, p2]) + 
       sp[l1, p2]^3*(2*sp[p2, p2]*(5*sp[p1, p2] + 3*sp[p2, p2]) + 
         sp[p1, p1]*(sp[p1, p2] + 5*sp[p2, p2])) - 
       sp[l1, p2]^2*(9*sp[p1, p2]^2*sp[p2, p2] - 3*sp[p2, p2]^3 + 
         sp[p1, p1]^2*(3*sp[p1, p2] + 10*sp[p2, p2]) + 
         sp[p1, p1]*(5*sp[p1, p2]^2 + 21*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2)) + sp[l1, p2]*(sp[p1, p2]^4 - 
         6*sp[p1, p1]^3*sp[p2, p2] + 10*sp[p1, p2]^3*sp[p2, p2] - 
         3*sp[p1, p2]*sp[p2, p2]^3 + 3*sp[p1, p1]^2*(sp[p1, p2]^2 - 
           5*sp[p1, p2]*sp[p2, p2] - 7*sp[p2, p2]^2) + 
         sp[p1, p1]*(6*sp[p1, p2]^3 + 5*sp[p1, p2]^2*sp[p2, p2] - 
           25*sp[p1, p2]*sp[p2, p2]^2 - 9*sp[p2, p2]^3)) + 
       sp[p2, p2]*(-3*sp[p1, p1]^3*sp[p2, p2] + sp[p1, p1]^2*
          (3*sp[p1, p2]^2 - 6*sp[p1, p2]*sp[p2, p2] - 8*sp[p2, p2]^2) + 
         sp[p1, p2]^2*(sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2) + sp[p1, p1]*(6*sp[p1, p2]^3 + 
           7*sp[p1, p2]^2*sp[p2, p2] - 6*sp[p1, p2]*sp[p2, p2]^2 - 
           3*sp[p2, p2]^3))) - sp[l1, p1]*
      (sp[l1, p2]^3*sp[p1, p2]*(sp[p1, p2] + sp[p2, p2]) - 
       sp[l1, p2]^2*(sp[p1, p1]^2*(5*sp[p1, p2] + 16*sp[p2, p2]) - 
         2*sp[p1, p2]*(sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2) + sp[p1, p1]*(5*sp[p1, p2]^2 + 
           19*sp[p1, p2]*sp[p2, p2] + 7*sp[p2, p2]^2)) + 
       sp[p2, p2]*(-3*sp[p1, p1]^3*(sp[p1, p2] + 3*sp[p2, p2]) - 
         sp[p1, p1]^2*sp[p2, p2]*(25*sp[p1, p2] + 21*sp[p2, p2]) + 
         sp[p1, p2]^2*(sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2) + sp[p1, p1]*(10*sp[p1, p2]^3 + 
           5*sp[p1, p2]^2*sp[p2, p2] - 15*sp[p1, p2]*sp[p2, p2]^2 - 
           6*sp[p2, p2]^3)) + sp[l1, p2]*
        (3*sp[p1, p1]^3*(sp[p1, p2] - 5*sp[p2, p2]) + 
         sp[p1, p1]^2*(12*sp[p1, p2]^2 - 39*sp[p1, p2]*sp[p2, p2] - 
           46*sp[p2, p2]^2) + 3*sp[p1, p1]*(4*sp[p1, p2]^3 + 
           sp[p1, p2]^2*sp[p2, p2] - 13*sp[p1, p2]*sp[p2, p2]^2 - 
           5*sp[p2, p2]^3) + sp[p1, p2]*(sp[p1, p2]^3 + 12*sp[p1, p2]^2*
            sp[p2, p2] + 12*sp[p1, p2]*sp[p2, p2]^2 + 3*sp[p2, p2]^3)))) + 
   sp[l1, l1]^2*(-6*sp[l1, p1]^4*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] + 
     sp[l1, p1]^3*(6*sp[l1, p2]*sp[p1, p2]*(sp[p1, p1] + sp[p1, p2]) + 
       sp[p2, p2]*(7*sp[p1, p1]^2 + sp[p1, p1]*(5*sp[p1, p2] + 
           2*sp[p2, p2]) - sp[p1, p2]*(7*sp[p1, p2] + 3*sp[p2, p2]))) + 
     sp[l1, p1]^2*(-12*sp[l1, p2]^2*(sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2]) + 
       sp[p2, p2]*(6*sp[p1, p1]^3 + sp[p1, p1]^2*(25*sp[p1, p2] + 
           19*sp[p2, p2]) + sp[p1, p1]*(41*sp[p1, p2]^2 + 
           60*sp[p1, p2]*sp[p2, p2] + 10*sp[p2, p2]^2) + 
         sp[p1, p2]*(16*sp[p1, p2]^2 + 46*sp[p1, p2]*sp[p2, p2] + 
           21*sp[p2, p2]^2)) + sp[l1, p2]*
        (sp[p1, p1]^2*(-7*sp[p1, p2] + 25*sp[p2, p2]) + 
         sp[p1, p2]*(5*sp[p1, p2]^2 + 39*sp[p1, p2]*sp[p2, p2] + 
           25*sp[p2, p2]^2) + sp[p1, p1]*(-5*sp[p1, p2]^2 + 
           61*sp[p1, p2]*sp[p2, p2] + 25*sp[p2, p2]^2))) + 
     sp[p1, p1]*(-6*sp[l1, p2]^4*(sp[p1, p2] + sp[p2, p2]) + 
       sp[l1, p2]^3*(7*sp[p1, p2]^2 + sp[p1, p1]*(3*sp[p1, p2] - 
           2*sp[p2, p2]) - 5*sp[p1, p2]*sp[p2, p2] - 7*sp[p2, p2]^2) + 
       sp[l1, p2]^2*(16*sp[p1, p2]^3 + 41*sp[p1, p2]^2*sp[p2, p2] + 
         25*sp[p1, p2]*sp[p2, p2]^2 + 6*sp[p2, p2]^3 + 
         sp[p1, p1]^2*(21*sp[p1, p2] + 10*sp[p2, p2]) + 
         sp[p1, p1]*(46*sp[p1, p2]^2 + 60*sp[p1, p2]*sp[p2, p2] + 
           19*sp[p2, p2]^2)) + sp[p2, p2]*(-3*sp[p1, p1]^3*sp[p2, p2] + 
         sp[p1, p1]^2*(3*sp[p1, p2]^2 + 2*sp[p1, p2]*sp[p2, p2] - 
           10*sp[p2, p2]^2) + sp[p1, p2]^2*(-9*sp[p1, p2]^2 - 
           2*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2) + 
         sp[p1, p1]*(-2*sp[p1, p2]^3 + 19*sp[p1, p2]^2*sp[p2, p2] + 
           2*sp[p1, p2]*sp[p2, p2]^2 - 3*sp[p2, p2]^3)) + 
       sp[l1, p2]*(3*sp[p1, p1]^3*sp[p1, p2] - 5*sp[p1, p2]^4 + 
         19*sp[p1, p2]^3*sp[p2, p2] + 25*sp[p1, p2]^2*sp[p2, p2]^2 + 
         9*sp[p1, p2]*sp[p2, p2]^3 + 3*sp[p2, p2]^4 + 3*sp[p1, p1]^2*
          (3*sp[p1, p2]^2 + 8*sp[p1, p2]*sp[p2, p2] - 3*sp[p2, p2]^2) + 
         sp[p1, p1]*(3*sp[p1, p2]^3 + 60*sp[p1, p2]^2*sp[p2, p2] + 
           27*sp[p1, p2]*sp[p2, p2]^2 + 2*sp[p2, p2]^3))) + 
     sp[l1, p1]*(6*sp[l1, p2]^3*sp[p1, p2]*(sp[p1, p2] + sp[p2, p2]) - 
       sp[l1, p2]^2*(25*sp[p1, p1]^2*(sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p2]*(5*sp[p1, p2]^2 - 5*sp[p1, p2]*sp[p2, p2] - 
           7*sp[p2, p2]^2) + sp[p1, p1]*(39*sp[p1, p2]^2 + 
           61*sp[p1, p2]*sp[p2, p2] + 25*sp[p2, p2]^2)) - 
       sp[p2, p2]*(3*sp[p1, p1]^4 + sp[p1, p1]^3*(9*sp[p1, p2] + 
           2*sp[p2, p2]) + sp[p1, p1]^2*(25*sp[p1, p2]^2 + 
           27*sp[p1, p2]*sp[p2, p2] - 9*sp[p2, p2]^2) + sp[p1, p1]*sp[p1, p2]*
          (19*sp[p1, p2]^2 + 60*sp[p1, p2]*sp[p2, p2] + 24*sp[p2, p2]^2) + 
         sp[p1, p2]*(-5*sp[p1, p2]^3 + 3*sp[p1, p2]^2*sp[p2, p2] + 
           9*sp[p1, p2]*sp[p2, p2]^2 + 3*sp[p2, p2]^3)) - 
       sp[l1, p2]*(3*sp[p1, p1]^3*(2*sp[p1, p2] + 7*sp[p2, p2]) + 
         sp[p1, p1]^2*(25*sp[p1, p2]^2 + 95*sp[p1, p2]*sp[p2, p2] + 
           27*sp[p2, p2]^2) + sp[p1, p2]*(-sp[p1, p2]^3 + 
           21*sp[p1, p2]^2*sp[p2, p2] + 25*sp[p1, p2]*sp[p2, p2]^2 + 
           6*sp[p2, p2]^3) + sp[p1, p1]*(21*sp[p1, p2]^3 + 
           126*sp[p1, p2]^2*sp[p2, p2] + 95*sp[p1, p2]*sp[p2, p2]^2 + 
           21*sp[p2, p2]^3)))) + sp[l1, l1]*
    (sp[l1, p1]^5*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] - 
     sp[l1, p1]^4*(sp[l1, p2]*(sp[p1, p1] + sp[p1, p2])*
        (sp[p1, p2] + 10*sp[p2, p2]) + sp[p2, p2]*(6*sp[p1, p1]^2 + 
         5*sp[p1, p2]*sp[p2, p2] + sp[p1, p1]*(10*sp[p1, p2] + 
           9*sp[p2, p2]))) + sp[l1, p1]^3*
      (sp[l1, p2]^2*(sp[p1, p1]*(10*sp[p1, p2] - 11*sp[p2, p2]) + 
         3*sp[p1, p2]*(4*sp[p1, p2] - 3*sp[p2, p2])) + 
       sp[l1, p2]*(sp[p1, p1]^2*(6*sp[p1, p2] + 5*sp[p2, p2]) - 
         sp[p1, p2]*sp[p2, p2]*(19*sp[p1, p2] + 21*sp[p2, p2]) + 
         sp[p1, p1]*(10*sp[p1, p2]^2 - 8*sp[p1, p2]*sp[p2, p2] - 
           19*sp[p2, p2]^2)) - sp[p2, p2]*(-3*sp[p1, p1]^3 - 
         2*sp[p1, p1]^2*sp[p2, p2] + 2*sp[p1, p2]*(sp[p1, p2]^2 + 
           8*sp[p1, p2]*sp[p2, p2] + 5*sp[p2, p2]^2) + 
         sp[p1, p1]*(11*sp[p1, p2]^2 + 19*sp[p1, p2]*sp[p2, p2] + 
           9*sp[p2, p2]^2))) + sp[p1, p1]*
      (-(sp[l1, p2]^5*(sp[p1, p2] + sp[p2, p2])) - 
       sp[l1, p2]^4*(2*sp[p2, p2]*(5*sp[p1, p2] + 3*sp[p2, p2]) + 
         sp[p1, p1]*(5*sp[p1, p2] + 9*sp[p2, p2])) + 
       sp[l1, p2]^3*(2*sp[p1, p2]^3 + 11*sp[p1, p2]^2*sp[p2, p2] - 
         3*sp[p2, p2]^3 + sp[p1, p1]^2*(10*sp[p1, p2] + 9*sp[p2, p2]) + 
         sp[p1, p1]*(16*sp[p1, p2]^2 + 19*sp[p1, p2]*sp[p2, p2] - 
           2*sp[p2, p2]^2)) - sp[p1, p2]*sp[p2, p2]*
        (-3*sp[p1, p1]^3*sp[p2, p2] + sp[p1, p1]^2*(3*sp[p1, p2]^2 - 
           6*sp[p1, p2]*sp[p2, p2] - 8*sp[p2, p2]^2) + 
         sp[p1, p2]^2*(sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2) + sp[p1, p1]*(6*sp[p1, p2]^3 + 
           7*sp[p1, p2]^2*sp[p2, p2] - 6*sp[p1, p2]*sp[p2, p2]^2 - 
           3*sp[p2, p2]^3)) + sp[l1, p2]^2*
        (3*sp[p1, p1]^3*(2*sp[p1, p2] + sp[p2, p2]) + sp[p1, p2]*sp[p2, p2]*
          (11*sp[p1, p2]^2 + 12*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2) + 
         sp[p1, p1]^2*(15*sp[p1, p2]^2 + 32*sp[p1, p2]*sp[p2, p2] + 
           10*sp[p2, p2]^2) + sp[p1, p1]*(7*sp[p1, p2]^3 + 
           41*sp[p1, p2]^2*sp[p2, p2] + 25*sp[p1, p2]*sp[p2, p2]^2 + 
           3*sp[p2, p2]^3)) + sp[l1, p2]*sp[p1, p2]*
        (6*sp[p1, p1]^3*sp[p2, p2] - 3*sp[p1, p1]^2*(sp[p1, p2]^2 - 
           5*sp[p1, p2]*sp[p2, p2] - 7*sp[p2, p2]^2) - 
         sp[p1, p2]*(sp[p1, p2]^3 + 10*sp[p1, p2]^2*sp[p2, p2] - 
           3*sp[p2, p2]^3) + sp[p1, p1]*(-6*sp[p1, p2]^3 - 
           5*sp[p1, p2]^2*sp[p2, p2] + 25*sp[p1, p2]*sp[p2, p2]^2 + 
           9*sp[p2, p2]^3))) + sp[l1, p1]^2*
      (sp[l1, p2]^3*(-2*sp[p1, p2]*(6*sp[p1, p2] + 5*sp[p2, p2]) + 
         sp[p1, p1]*(9*sp[p1, p2] + 11*sp[p2, p2])) + 
       sp[l1, p2]^2*(sp[p1, p1]^2*(-5*sp[p1, p2] + 41*sp[p2, p2]) + 
         sp[p1, p2]*(5*sp[p1, p2]^2 - 3*sp[p1, p2]*sp[p2, p2] - 
           5*sp[p2, p2]^2) + sp[p1, p1]*(-3*sp[p1, p2]^2 + 
           81*sp[p1, p2]*sp[p2, p2] + 41*sp[p2, p2]^2)) + 
       sp[p2, p2]*(3*sp[p1, p1]^3*(sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p2]*sp[p2, p2]*(7*sp[p1, p2]^2 + 15*sp[p1, p2]*sp[p2, p2] + 
           6*sp[p2, p2]^2) + sp[p1, p1]^2*(12*sp[p1, p2]^2 + 
           25*sp[p1, p2]*sp[p2, p2] + 10*sp[p2, p2]^2) + 
         sp[p1, p1]*(11*sp[p1, p2]^3 + 41*sp[p1, p2]^2*sp[p2, p2] + 
           32*sp[p1, p2]*sp[p2, p2]^2 + 3*sp[p2, p2]^3)) + 
       sp[l1, p2]*(-3*sp[p1, p1]^3*(sp[p1, p2] - 5*sp[p2, p2]) + 
         sp[p1, p1]^2*sp[p2, p2]*(61*sp[p1, p2] + 60*sp[p2, p2]) + 
         sp[p1, p2]*sp[p2, p2]*(19*sp[p1, p2]^2 + 39*sp[p1, p2]*sp[p2, p2] + 
           15*sp[p2, p2]^2) + sp[p1, p1]*(9*sp[p1, p2]^3 + 
           81*sp[p1, p2]^2*sp[p2, p2] + 126*sp[p1, p2]*sp[p2, p2]^2 + 
           32*sp[p2, p2]^3))) + sp[l1, p1]*
      (sp[l1, p2]^4*(10*sp[p1, p1] + sp[p1, p2])*(sp[p1, p2] + sp[p2, p2]) + 
       sp[l1, p2]^3*(2*sp[p1, p2]*sp[p2, p2]*(5*sp[p1, p2] + 3*sp[p2, p2]) - 
         sp[p1, p1]^2*(21*sp[p1, p2] + 19*sp[p2, p2]) + 
         sp[p1, p1]*(-19*sp[p1, p2]^2 - 8*sp[p1, p2]*sp[p2, p2] + 
           5*sp[p2, p2]^2)) + sp[p1, p2]*sp[p2, p2]*
        (-3*sp[p1, p1]^3*(sp[p1, p2] + 3*sp[p2, p2]) - 
         sp[p1, p1]^2*sp[p2, p2]*(25*sp[p1, p2] + 21*sp[p2, p2]) + 
         sp[p1, p2]^2*(sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2) + sp[p1, p1]*(10*sp[p1, p2]^3 + 
           5*sp[p1, p2]^2*sp[p2, p2] - 15*sp[p1, p2]*sp[p2, p2]^2 - 
           6*sp[p2, p2]^3)) - sp[l1, p2]^2*(9*sp[p1, p2]^3*sp[p2, p2] - 
         3*sp[p1, p2]*sp[p2, p2]^3 + sp[p1, p1]^3*(15*sp[p1, p2] + 
           32*sp[p2, p2]) + 3*sp[p1, p1]^2*(13*sp[p1, p2]^2 + 
           42*sp[p1, p2]*sp[p2, p2] + 20*sp[p2, p2]^2) + 
         sp[p1, p1]*(19*sp[p1, p2]^3 + 81*sp[p1, p2]^2*sp[p2, p2] + 
           61*sp[p1, p2]*sp[p2, p2]^2 + 15*sp[p2, p2]^3)) - 
       sp[l1, p2]*(6*sp[p1, p1]^4*sp[p2, p2] + 3*sp[p1, p1]^3*
          (sp[p1, p2]^2 + 11*sp[p1, p2]*sp[p2, p2] + 8*sp[p2, p2]^2) + 
         sp[p1, p1]^2*sp[p2, p2]*(61*sp[p1, p2]^2 + 95*sp[p1, p2]*
            sp[p2, p2] + 24*sp[p2, p2]^2) - sp[p1, p2]^2*
          (sp[p1, p2]^3 + 10*sp[p1, p2]^2*sp[p2, p2] - 3*sp[p2, p2]^3) + 
         sp[p1, p1]*(-10*sp[p1, p2]^4 + 8*sp[p1, p2]^3*sp[p2, p2] + 
           61*sp[p1, p2]^2*sp[p2, p2]^2 + 33*sp[p1, p2]*sp[p2, p2]^3 + 
           6*sp[p2, p2]^4))))))/(FunKit`dressing[FunKit`InverseProp, {A, A}, 
   1, {-l1, l1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 - p2, l1 + p2}]*
  sp[l1, l1]*(sp[l1, l1] - 2*sp[l1, p1] + sp[p1, p1])*
  (sp[l1, l1] + 2*sp[l1, p2] + sp[p2, p2])*
  (-sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2])*(3*sp[p1, p1]^2 + sp[p1, p2]^2 + 
   6*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2 + 
   sp[p1, p1]*(6*sp[p1, p2] + 8*sp[p2, p2])))
