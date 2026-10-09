(* Created with the Wolfram Language : www.wolfram.com *)
(-6*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
   {p1, l1 - p1 - p2, -l1 + p2}]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, 
   {-p1 - p2, l1, -l1 + p1 + p2}]*FunKit`dressing[FunKit`GammaN, {A, A, A}, 
   1, {p2, l1 - p2, -l1}]*FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
  (-3*sp[l1, l1]^4*(sp[l1, p2]*sp[p1, p1]*(sp[p1, p1]*sp[p1, p2] + 
       2*sp[p1, p2]^2 + 2*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) - 
     sp[p2, p2]*(sp[p1, p1]^2*sp[p2, p2] - sp[p1, p2]^2*
        (2*sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]*(-sp[p1, p2]^2 + 
         2*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) + 
       sp[l1, p1]*(sp[p1, p1]^2 + 2*sp[p1, p1]*sp[p1, p2] + 
         sp[p1, p2]*(2*sp[p1, p2] + sp[p2, p2])))) + 
   sp[l1, l1]^3*(-3*sp[l1, p1]^3*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] + 
     3*sp[l1, p2]^3*sp[p1, p1]*(sp[p1, p2] + sp[p2, p2]) + 
     3*sp[l1, p1]^2*(sp[l1, p2]*(sp[p1, p1] + sp[p1, p2])*
        (sp[p1, p2] - sp[p2, p2]) - sp[p2, p2]*(2*sp[p1, p1]^2 + 
         3*sp[p1, p1]*sp[p1, p2] + 3*sp[p1, p2]^2 + 2*sp[p1, p2]*
          sp[p2, p2])) + 3*sp[l1, p2]^2*
      (sp[p1, p1]^2*(5*sp[p1, p2] - sp[p2, p2]) + 
       sp[p1, p2]^2*(sp[p1, p2] + sp[p2, p2]) + 
       sp[p1, p1]*(10*sp[p1, p2]^2 + 7*sp[p1, p2]*sp[p2, p2] + 
         3*sp[p2, p2]^2)) + sp[p2, p2]*(10*sp[p1, p1]^3*sp[p2, p2] - 
       11*sp[p1, p2]^2*(2*sp[p1, p2]^2 + 3*sp[p1, p2]*sp[p2, p2] + 
         sp[p2, p2]^2) + sp[p1, p1]^2*(-10*sp[p1, p2]^2 + 
         31*sp[p1, p2]*sp[p2, p2] + 21*sp[p2, p2]^2) + 
       sp[p1, p1]*(-31*sp[p1, p2]^3 + sp[p1, p2]^2*sp[p2, p2] + 
         33*sp[p1, p2]*sp[p2, p2]^2 + 11*sp[p2, p2]^3)) - 
     sp[l1, p2]*(8*sp[p1, p1]^3*sp[p1, p2] - 12*sp[p1, p2]^2*sp[p2, p2]*
        (2*sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]^2*(27*sp[p1, p2]^2 + 
         27*sp[p1, p2]*sp[p2, p2] + 20*sp[p2, p2]^2) + 
       sp[p1, p1]*(22*sp[p1, p2]^3 + 32*sp[p1, p2]^2*sp[p2, p2] + 
         57*sp[p1, p2]*sp[p2, p2]^2 + 23*sp[p2, p2]^3)) + 
     sp[l1, p1]*(3*sp[l1, p2]^2*(sp[p1, p1] - sp[p1, p2])*
        (sp[p1, p2] + sp[p2, p2]) + 3*sp[l1, p2]*
        (sp[p1, p1]^2*(2*sp[p1, p2] - 5*sp[p2, p2]) + 
         sp[p1, p1]*(3*sp[p1, p2]^2 - 5*sp[p1, p2]*sp[p2, p2] + 
           sp[p2, p2]^2) - sp[p1, p2]*(sp[p1, p2]^2 + 7*sp[p1, p2]*
            sp[p2, p2] + 4*sp[p2, p2]^2)) + sp[p2, p2]*
        (8*sp[p1, p1]^3 + sp[p1, p1]^2*(27*sp[p1, p2] + 5*sp[p2, p2]) + 
         2*sp[p1, p1]*(22*sp[p1, p2]^2 + 9*sp[p1, p2]*sp[p2, p2] - 
           3*sp[p2, p2]^2) + sp[p1, p2]*(34*sp[p1, p2]^2 + 
           39*sp[p1, p2]*sp[p2, p2] + 11*sp[p2, p2]^2)))) + 
   sp[l1, p2]*(sp[l1, p1] + sp[l1, p2])*
    (sp[l1, p1]^4*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] - 
     2*sp[l1, p2]^4*sp[p1, p1]*(sp[p1, p2] + sp[p2, p2]) + 
     sp[l1, p2]^3*(-2*sp[p1, p2]^2*(sp[p1, p2] + sp[p2, p2]) + 
       2*sp[p1, p1]^2*(sp[p1, p2] + 7*sp[p2, p2]) + 
       sp[p1, p1]*(sp[p1, p2]^2 + 28*sp[p1, p2]*sp[p2, p2] + 
         15*sp[p2, p2]^2)) + sp[l1, p2]^2*(-22*sp[p1, p1]^3*sp[p2, p2] + 
       sp[p1, p1]^2*(7*sp[p1, p2]^2 - 58*sp[p1, p2]*sp[p2, p2] - 
         51*sp[p2, p2]^2) + sp[p1, p2]^2*(5*sp[p1, p2]^2 + 
         26*sp[p1, p2]*sp[p2, p2] + 13*sp[p2, p2]^2) + 
       sp[p1, p1]*(16*sp[p1, p2]^3 - 5*sp[p1, p2]^2*sp[p2, p2] - 
         62*sp[p1, p2]*sp[p2, p2]^2 - 25*sp[p2, p2]^3)) + 
     sp[p2, p2]*(-3*sp[p1, p1]^4*sp[p2, p2] + sp[p1, p1]^3*
        (3*sp[p1, p2]^2 - 12*sp[p1, p2]*sp[p2, p2] - 11*sp[p2, p2]^2) + 
       sp[p1, p1]^2*(12*sp[p1, p2]^3 - 2*sp[p1, p2]^2*sp[p2, p2] - 
         28*sp[p1, p2]*sp[p2, p2]^2 - 11*sp[p2, p2]^3) + 
       sp[p1, p2]^2*(2*sp[p1, p2]^3 + 13*sp[p1, p2]^2*sp[p2, p2] + 
         12*sp[p1, p2]*sp[p2, p2]^2 + 3*sp[p2, p2]^3) + 
       sp[p1, p1]*(13*sp[p1, p2]^4 + 26*sp[p1, p2]^3*sp[p2, p2] - 
         2*sp[p1, p2]^2*sp[p2, p2]^2 - 12*sp[p1, p2]*sp[p2, p2]^3 - 
         3*sp[p2, p2]^4)) + sp[l1, p2]*(6*sp[p1, p1]^4*sp[p2, p2] - 
       3*sp[p1, p1]^3*(sp[p1, p2]^2 - 8*sp[p1, p2]*sp[p2, p2] - 
         11*sp[p2, p2]^2) - 2*sp[p1, p2]^2*(sp[p1, p2]^3 + 
         13*sp[p1, p2]^2*sp[p2, p2] + 18*sp[p1, p2]*sp[p2, p2]^2 + 
         6*sp[p2, p2]^3) + sp[p1, p1]^2*(-12*sp[p1, p2]^3 + 
         4*sp[p1, p2]^2*sp[p2, p2] + 84*sp[p1, p2]*sp[p2, p2]^2 + 
         44*sp[p2, p2]^3) + sp[p1, p1]*(-13*sp[p1, p2]^4 - 
         52*sp[p1, p2]^3*sp[p2, p2] + 6*sp[p1, p2]^2*sp[p2, p2]^2 + 
         48*sp[p1, p2]*sp[p2, p2]^3 + 15*sp[p2, p2]^4)) - 
     sp[l1, p1]^3*(sp[l1, p2]*(sp[p1, p1] + sp[p1, p2])*
        (sp[p1, p2] - 3*sp[p2, p2]) + sp[p2, p2]*(6*sp[p1, p1]^2 + 
         7*sp[p1, p1]*(2*sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p2]*(4*sp[p1, p2] + 3*sp[p2, p2]))) + 
     sp[l1, p1]^2*(sp[l1, p2]^2*(-(sp[p1, p2]*(sp[p1, p2] - 3*sp[p2, p2])) + 
         sp[p1, p1]*(-3*sp[p1, p2] + sp[p2, p2])) + 
       sp[l1, p2]*(sp[p1, p1]^2*(6*sp[p1, p2] - 5*sp[p2, p2]) + 
         sp[p1, p2]*(2*sp[p1, p2]^2 - 23*sp[p1, p2]*sp[p2, p2] - 
           13*sp[p2, p2]^2) + sp[p1, p1]*(12*sp[p1, p2]^2 - 
           21*sp[p1, p2]*sp[p2, p2] - 10*sp[p2, p2]^2)) + 
       sp[p2, p2]*(3*sp[p1, p1]^3 + 6*sp[p1, p1]^2*(3*sp[p1, p2] + 
           sp[p2, p2]) + 2*sp[p1, p2]*(3*sp[p1, p2]^2 + 8*sp[p1, p2]*
            sp[p2, p2] + 3*sp[p2, p2]^2) + sp[p1, p1]*(27*sp[p1, p2]^2 + 
           27*sp[p1, p2]*sp[p2, p2] + 5*sp[p2, p2]^2))) - 
     sp[l1, p1]*(sp[l1, p2]^3*(3*sp[p1, p1] - 2*sp[p1, p2])*
        (sp[p1, p2] + sp[p2, p2]) - sp[l1, p2]^2*
        (sp[p1, p1]^2*(7*sp[p1, p2] + 19*sp[p2, p2]) - 
         sp[p1, p2]*(4*sp[p1, p2]^2 + 29*sp[p1, p2]*sp[p2, p2] + 
           13*sp[p2, p2]^2) + sp[p1, p1]*(9*sp[p1, p2]^2 + 
           25*sp[p1, p2]*sp[p2, p2] + 16*sp[p2, p2]^2)) + 
       sp[p2, p2]*(3*sp[p1, p1]^3*(sp[p1, p2] - 2*sp[p2, p2]) + 
         2*sp[p1, p1]^2*(9*sp[p1, p2]^2 - sp[p1, p2]*sp[p2, p2] - 
           7*sp[p2, p2]^2) + sp[p1, p1]*(26*sp[p1, p2]^3 + 
           39*sp[p1, p2]^2*sp[p2, p2] - 2*sp[p1, p2]*sp[p2, p2]^2 - 
           6*sp[p2, p2]^3) + sp[p1, p2]*(5*sp[p1, p2]^3 + 
           26*sp[p1, p2]^2*sp[p2, p2] + 18*sp[p1, p2]*sp[p2, p2]^2 + 
           3*sp[p2, p2]^3)) + sp[l1, p2]*
        (3*sp[p1, p1]^3*(sp[p1, p2] + 4*sp[p2, p2]) + sp[p1, p1]*sp[p2, p2]*
          (-45*sp[p1, p2]^2 + 16*sp[p1, p2]*sp[p2, p2] + 19*sp[p2, p2]^2) + 
         sp[p1, p1]^2*(6*sp[p1, p2]^2 + 18*sp[p1, p2]*sp[p2, p2] + 
           37*sp[p2, p2]^2) - sp[p1, p2]*(sp[p1, p2]^3 + 46*sp[p1, p2]^2*
            sp[p2, p2] + 49*sp[p1, p2]*sp[p2, p2]^2 + 12*sp[p2, p2]^3)))) + 
   sp[l1, l1]^2*(6*sp[l1, p1]^4*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2] - 
     12*sp[l1, p2]^4*sp[p1, p1]*(sp[p1, p2] + sp[p2, p2]) + 
     sp[l1, p2]^3*(-12*sp[p1, p2]^2*(sp[p1, p2] + sp[p2, p2]) + 
       sp[p1, p1]^2*(-21*sp[p1, p2] + 20*sp[p2, p2]) + 
       sp[p1, p1]*(-43*sp[p1, p2]^2 + 8*sp[p1, p2]*sp[p2, p2] + 
         10*sp[p2, p2]^2)) + sp[l1, p2]^2*
      (sp[p1, p1]^3*(27*sp[p1, p2] - 10*sp[p2, p2]) + 
       sp[p1, p2]^2*(7*sp[p1, p2]^2 - 4*sp[p1, p2]*sp[p2, p2] - 
         2*sp[p2, p2]^2) + sp[p1, p1]^2*(91*sp[p1, p2]^2 + 
         59*sp[p1, p2]*sp[p2, p2] + 14*sp[p2, p2]^2) + 
       sp[p1, p1]*(77*sp[p1, p2]^3 + 117*sp[p1, p2]^2*sp[p2, p2] + 
         88*sp[p1, p2]*sp[p2, p2]^2 + 30*sp[p2, p2]^3)) + 
     sp[p2, p2]*(3*sp[p1, p1]^4*sp[p2, p2] + sp[p1, p1]^3*
        (-3*sp[p1, p2]^2 + 20*sp[p1, p2]*sp[p2, p2] + 21*sp[p2, p2]^2) - 
       sp[p1, p2]^2*(14*sp[p1, p2]^3 + 51*sp[p1, p2]^2*sp[p2, p2] + 
         44*sp[p1, p2]*sp[p2, p2]^2 + 11*sp[p2, p2]^3) + 
       sp[p1, p1]^2*(-20*sp[p1, p2]^3 + 14*sp[p1, p2]^2*sp[p2, p2] + 
         72*sp[p1, p2]*sp[p2, p2]^2 + 29*sp[p2, p2]^3) + 
       sp[p1, p1]*(-35*sp[p1, p2]^4 - 58*sp[p1, p2]^3*sp[p2, p2] + 
         22*sp[p1, p2]^2*sp[p2, p2]^2 + 44*sp[p1, p2]*sp[p2, p2]^3 + 
         11*sp[p2, p2]^4)) - sp[l1, p2]*(3*sp[p1, p1]^4*sp[p1, p2] + 
       sp[p1, p1]^3*(15*sp[p1, p2]^2 + 27*sp[p1, p2]*sp[p2, p2] + 
         31*sp[p2, p2]^2) - 2*sp[p1, p2]^2*(sp[p1, p2]^3 + 
         29*sp[p1, p2]^2*sp[p2, p2] + 42*sp[p1, p2]*sp[p2, p2]^2 + 
         14*sp[p2, p2]^3) + sp[p1, p1]^2*(21*sp[p1, p2]^3 + 
         59*sp[p1, p2]^2*sp[p2, p2] + 142*sp[p1, p2]*sp[p2, p2]^2 + 
         72*sp[p2, p2]^3) + sp[p1, p1]*(5*sp[p1, p2]^4 - 
         24*sp[p1, p2]^3*sp[p2, p2] + 76*sp[p1, p2]^2*sp[p2, p2]^2 + 
         128*sp[p1, p2]*sp[p2, p2]^3 + 39*sp[p2, p2]^4)) - 
     sp[l1, p1]^3*(6*sp[l1, p2]*(sp[p1, p1] + sp[p1, p2])*
        (sp[p1, p2] - 3*sp[p2, p2]) + sp[p2, p2]*(7*sp[p1, p1]^2 + 
         sp[p1, p2]*(11*sp[p1, p2] + 6*sp[p2, p2]) + 
         sp[p1, p1]*(23*sp[p1, p2] + 11*sp[p2, p2]))) - 
     sp[l1, p1]^2*(6*sp[l1, p2]^2*(sp[p1, p2]*(sp[p1, p2] - 3*sp[p2, p2]) + 
         sp[p1, p1]*(3*sp[p1, p2] - sp[p2, p2])) - 
       sp[l1, p2]*(13*sp[p1, p2]^3 + 2*sp[p1, p2]*sp[p2, p2]^2 + 
         sp[p1, p1]^2*(7*sp[p1, p2] + 11*sp[p2, p2]) + 
         sp[p1, p1]*(23*sp[p1, p2]^2 + 2*sp[p1, p2]*sp[p2, p2] - 
           10*sp[p2, p2]^2)) + sp[p2, p2]*(6*sp[p1, p1]^3 + 
         sp[p1, p1]^2*(11*sp[p1, p2] + 12*sp[p2, p2]) + 
         sp[p1, p1]*(13*sp[p1, p2]^2 + 20*sp[p1, p2]*sp[p2, p2] - 
           3*sp[p2, p2]^2) + sp[p1, p2]*(14*sp[p1, p2]^2 + 
           31*sp[p1, p2]*sp[p2, p2] + 14*sp[p2, p2]^2))) + 
     sp[l1, p1]*(-6*sp[l1, p2]^3*(3*sp[p1, p1] - 2*sp[p1, p2])*
        (sp[p1, p2] + sp[p2, p2]) - sp[l1, p2]^2*
        (sp[p1, p1]^2*(11*sp[p1, p2] - 43*sp[p2, p2]) + 
         sp[p1, p1]*(15*sp[p1, p2]^2 - 40*sp[p1, p2]*sp[p2, p2] - 
           16*sp[p2, p2]^2) + sp[p1, p2]*(sp[p1, p2]^2 + 14*sp[p1, p2]*
            sp[p2, p2] - 2*sp[p2, p2]^2)) + sp[p2, p2]*
        (3*sp[p1, p1]^4 + 5*sp[p1, p1]^3*(3*sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]^2*(43*sp[p1, p2]^2 + 26*sp[p1, p2]*sp[p2, p2] - 
           14*sp[p2, p2]^2) + sp[p1, p1]*(69*sp[p1, p2]^3 + 
           97*sp[p1, p2]^2*sp[p2, p2] + 2*sp[p1, p2]*sp[p2, p2]^2 - 
           14*sp[p2, p2]^3) + sp[p1, p2]*(33*sp[p1, p2]^3 + 
           88*sp[p1, p2]^2*sp[p2, p2] + 58*sp[p1, p2]*sp[p2, p2]^2 + 
           11*sp[p2, p2]^3)) + sp[l1, p2]*
        (3*sp[p1, p1]^3*(2*sp[p1, p2] - 9*sp[p2, p2]) + 
         sp[p1, p1]^2*(11*sp[p1, p2]^2 - 49*sp[p1, p2]*sp[p2, p2] - 
           26*sp[p2, p2]^2) - sp[p1, p1]*(7*sp[p1, p2]^3 + 
           67*sp[p1, p2]^2*sp[p2, p2] + 34*sp[p1, p2]*sp[p2, p2]^2 - 
           16*sp[p2, p2]^3) - sp[p1, p2]*(9*sp[p1, p2]^3 + 
           76*sp[p1, p2]^2*sp[p2, p2] + 86*sp[p1, p2]*sp[p2, p2]^2 + 
           28*sp[p2, p2]^3)))) + sp[l1, l1]*
    (-(sp[l1, p1]^5*(sp[p1, p1] + sp[p1, p2])*sp[p2, p2]) + 
     13*sp[l1, p2]^5*sp[p1, p1]*(sp[p1, p2] + sp[p2, p2]) + 
     sp[l1, p2]^4*(5*sp[p1, p1]^2*(sp[p1, p2] - 7*sp[p2, p2]) + 
       13*sp[p1, p2]^2*(sp[p1, p2] + sp[p2, p2]) + 
       sp[p1, p1]*(14*sp[p1, p2]^2 - 65*sp[p1, p2]*sp[p2, p2] - 
         39*sp[p2, p2]^2)) - sp[l1, p2]^3*
      (sp[p1, p1]^3*(22*sp[p1, p2] - 31*sp[p2, p2]) + 
       sp[p1, p1]^2*(77*sp[p1, p2]^2 - 24*sp[p1, p2]*sp[p2, p2] - 
         58*sp[p2, p2]^2) + 2*sp[p1, p2]^2*(8*sp[p1, p2]^2 + 
         26*sp[p1, p2]*sp[p2, p2] + 13*sp[p2, p2]^2) + 
       sp[p1, p1]*(74*sp[p1, p2]^3 + 72*sp[p1, p2]^2*sp[p2, p2] - 
         46*sp[p1, p2]*sp[p2, p2]^2 - 24*sp[p2, p2]^3)) + 
     sp[l1, p2]*(sp[p1, p2] + sp[p2, p2])*(-6*sp[p1, p1]^4*sp[p2, p2] + 
       3*sp[p1, p1]^3*(sp[p1, p2]^2 - 8*sp[p1, p2]*sp[p2, p2] - 
         11*sp[p2, p2]^2) + 4*sp[p1, p1]^2*(3*sp[p1, p2]^3 - 
         sp[p1, p2]^2*sp[p2, p2] - 21*sp[p1, p2]*sp[p2, p2]^2 - 
         11*sp[p2, p2]^3) + 2*sp[p1, p2]^2*(sp[p1, p2]^3 + 
         13*sp[p1, p2]^2*sp[p2, p2] + 18*sp[p1, p2]*sp[p2, p2]^2 + 
         6*sp[p2, p2]^3) + sp[p1, p1]*(13*sp[p1, p2]^4 + 
         52*sp[p1, p2]^3*sp[p2, p2] - 6*sp[p1, p2]^2*sp[p2, p2]^2 - 
         48*sp[p1, p2]*sp[p2, p2]^3 - 15*sp[p2, p2]^4)) + 
     sp[p2, p2]*(sp[p1, p2] + sp[p2, p2])*(3*sp[p1, p1]^4*sp[p2, p2] + 
       sp[p1, p1]^3*(-3*sp[p1, p2]^2 + 12*sp[p1, p2]*sp[p2, p2] + 
         11*sp[p2, p2]^2) - sp[p1, p2]^2*(2*sp[p1, p2]^3 + 
         13*sp[p1, p2]^2*sp[p2, p2] + 12*sp[p1, p2]*sp[p2, p2]^2 + 
         3*sp[p2, p2]^3) + sp[p1, p1]^2*(-12*sp[p1, p2]^3 + 
         2*sp[p1, p2]^2*sp[p2, p2] + 28*sp[p1, p2]*sp[p2, p2]^2 + 
         11*sp[p2, p2]^3) + sp[p1, p1]*(-13*sp[p1, p2]^4 - 
         26*sp[p1, p2]^3*sp[p2, p2] + 2*sp[p1, p2]^2*sp[p2, p2]^2 + 
         12*sp[p1, p2]*sp[p2, p2]^3 + 3*sp[p2, p2]^4)) + 
     sp[l1, p2]^2*(sp[p1, p1]^4*(6*sp[p1, p2] - 3*sp[p2, p2]) + 
       sp[p1, p1]^3*(30*sp[p1, p2]^2 + 32*sp[p1, p2]*sp[p2, p2] + 
         sp[p2, p2]^2) - sp[p1, p2]^2*(sp[p1, p2]^3 + 5*sp[p1, p2]^2*
          sp[p2, p2] + 6*sp[p1, p2]*sp[p2, p2]^2 + 2*sp[p2, p2]^3) + 
       sp[p1, p1]^2*(43*sp[p1, p2]^3 + 117*sp[p1, p2]^2*sp[p2, p2] + 
         76*sp[p1, p2]*sp[p2, p2]^2 + 22*sp[p2, p2]^3) + 
       2*sp[p1, p1]*(7*sp[p1, p2]^4 + 36*sp[p1, p2]^3*sp[p2, p2] + 
         44*sp[p1, p2]^2*sp[p2, p2]^2 + 27*sp[p1, p2]*sp[p2, p2]^3 + 
         7*sp[p2, p2]^4)) + sp[l1, p1]^4*
      (sp[l1, p2]*(sp[p1, p1] + sp[p1, p2])*(sp[p1, p2] - 14*sp[p2, p2]) + 
       sp[p2, p2]*(6*sp[p1, p1]^2 + sp[p1, p2]*(4*sp[p1, p2] + 
           7*sp[p2, p2]) + sp[p1, p1]*(14*sp[p1, p2] + 11*sp[p2, p2]))) + 
     sp[l1, p1]^3*(sp[l1, p2]^2*(sp[p1, p1]*(14*sp[p1, p2] - 25*sp[p2, p2]) + 
         3*sp[p1, p2]*(4*sp[p1, p2] - 9*sp[p2, p2])) - 
       sp[p2, p2]*(3*sp[p1, p1]^3 + sp[p1, p1]^2*(18*sp[p1, p2] + 
           11*sp[p2, p2]) + sp[p1, p2]*(4*sp[p1, p2]^2 + 17*sp[p1, p2]*
            sp[p2, p2] + 5*sp[p2, p2]^2) + sp[p1, p1]*(25*sp[p1, p2]^2 + 
           41*sp[p1, p2]*sp[p2, p2] + 12*sp[p2, p2]^2)) + 
       sp[l1, p2]*(sp[p1, p1]^2*(-6*sp[p1, p2] + 23*sp[p2, p2]) + 
         sp[p1, p2]*(-4*sp[p1, p2]^2 + 35*sp[p1, p2]*sp[p2, p2] + 
           27*sp[p2, p2]^2) + sp[p1, p1]*(-14*sp[p1, p2]^2 + 
           68*sp[p1, p2]*sp[p2, p2] + 41*sp[p2, p2]^2))) + 
     sp[l1, p1]^2*(sp[l1, p2]^3*(-26*sp[p1, p2]*sp[p2, p2] + 
         sp[p1, p1]*(27*sp[p1, p2] + sp[p2, p2])) + 
       sp[l1, p2]^2*(-(sp[p1, p1]^2*(23*sp[p1, p2] + 13*sp[p2, p2])) + 
         sp[p1, p1]*(-59*sp[p1, p2]^2 + 13*sp[p1, p2]*sp[p2, p2] + 
           8*sp[p2, p2]^2) + 3*sp[p1, p2]*(-7*sp[p1, p2]^2 + 
           24*sp[p1, p2]*sp[p2, p2] + 13*sp[p2, p2]^2)) + 
       sp[l1, p2]*(3*sp[p1, p1]^3*(sp[p1, p2] + 3*sp[p2, p2]) + 
         2*sp[p1, p1]^2*(9*sp[p1, p2]^2 + sp[p1, p2]*sp[p2, p2] + 
           10*sp[p2, p2]^2) + sp[p1, p1]*(27*sp[p1, p2]^3 - 
           13*sp[p1, p2]^2*sp[p2, p2] - 15*sp[p1, p2]*sp[p2, p2]^2 - 
           6*sp[p2, p2]^3) + sp[p1, p2]*(6*sp[p1, p2]^3 + 
           4*sp[p1, p2]^2*sp[p2, p2] - 11*sp[p1, p2]*sp[p2, p2]^2 + 
           2*sp[p2, p2]^3)) + sp[p2, p2]*(3*sp[p1, p1]^3*sp[p1, p2] + 
         sp[p1, p1]^2*(6*sp[p1, p2]^2 + 10*sp[p1, p2]*sp[p2, p2] + 
           3*sp[p2, p2]^2) - 2*sp[p1, p2]*(sp[p1, p2]^3 + 
           3*sp[p1, p2]^2*sp[p2, p2] + 7*sp[p1, p2]*sp[p2, p2]^2 + 
           3*sp[p2, p2]^3) + sp[p1, p1]*(-sp[p1, p2]^3 + 8*sp[p1, p2]^2*
            sp[p2, p2] + 6*sp[p1, p2]*sp[p2, p2]^2 + 5*sp[p2, p2]^3))) + 
     sp[l1, p1]*(13*sp[l1, p2]^4*(2*sp[p1, p1] - sp[p1, p2])*
        (sp[p1, p2] + sp[p2, p2]) - sp[l1, p2]^3*
        (sp[p1, p1]^2*(7*sp[p1, p2] + 69*sp[p2, p2]) - 
         2*sp[p1, p2]*(8*sp[p1, p2]^2 + 39*sp[p1, p2]*sp[p2, p2] + 
           13*sp[p2, p2]^2) + sp[p1, p1]*(11*sp[p1, p2]^2 + 
           102*sp[p1, p2]*sp[p2, p2] + 65*sp[p2, p2]^2)) + 
       sp[p2, p2]*(sp[p1, p2] + sp[p2, p2])*
        (3*sp[p1, p1]^3*(sp[p1, p2] - 2*sp[p2, p2]) + 2*sp[p1, p1]^2*
          (9*sp[p1, p2]^2 - sp[p1, p2]*sp[p2, p2] - 7*sp[p2, p2]^2) + 
         sp[p1, p1]*(26*sp[p1, p2]^3 + 39*sp[p1, p2]^2*sp[p2, p2] - 
           2*sp[p1, p2]*sp[p2, p2]^2 - 6*sp[p2, p2]^3) + 
         sp[p1, p2]*(5*sp[p1, p2]^3 + 26*sp[p1, p2]^2*sp[p2, p2] + 
           18*sp[p1, p2]*sp[p2, p2]^2 + 3*sp[p2, p2]^3)) + 
       sp[l1, p2]^2*(sp[p1, p1]^3*(-9*sp[p1, p2] + 44*sp[p2, p2]) + 
         sp[p1, p1]^2*(-15*sp[p1, p2]^2 + 67*sp[p1, p2]*sp[p2, p2] + 
           97*sp[p2, p2]^2) + sp[p1, p2]*(10*sp[p1, p2]^3 - 
           40*sp[p1, p2]^2*sp[p2, p2] - 33*sp[p1, p2]*sp[p2, p2]^2 + 
           2*sp[p2, p2]^3) + sp[p1, p1]*(11*sp[p1, p2]^3 - 
           17*sp[p1, p2]^2*sp[p2, p2] + 92*sp[p1, p2]*sp[p2, p2]^2 + 
           37*sp[p2, p2]^3)) - sp[l1, p2]*(6*sp[p1, p1]^4*sp[p2, p2] + 
         3*sp[p1, p1]^3*(sp[p1, p2]^2 + 5*sp[p1, p2]*sp[p2, p2] + 
           6*sp[p2, p2]^2) + 2*sp[p1, p1]^2*(9*sp[p1, p2]^3 + 
           20*sp[p1, p2]^2*sp[p2, p2] + 17*sp[p1, p2]*sp[p2, p2]^2 + 
           sp[p2, p2]^3) + 2*sp[p1, p1]*(13*sp[p1, p2]^4 + 
           51*sp[p1, p2]^3*sp[p2, p2] + 46*sp[p1, p2]^2*sp[p2, p2]^2 + 
           9*sp[p1, p2]*sp[p2, p2]^3 - 4*sp[p2, p2]^4) + 
         sp[p1, p2]*(5*sp[p1, p2]^4 + 53*sp[p1, p2]^3*sp[p2, p2] + 
           78*sp[p1, p2]^2*sp[p2, p2]^2 + 50*sp[p1, p2]*sp[p2, p2]^3 + 
           12*sp[p2, p2]^4))))))/(FunKit`dressing[FunKit`InverseProp, {A, A}, 
   1, {-l1, l1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {l1 - p1 - p2, -l1 + p1 + p2}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 
   1, {-l1 + p2, l1 - p2}]*sp[l1, l1]*(sp[l1, l1] - 2*sp[l1, p2] + 
   sp[p2, p2])*(sp[l1, l1] - 2*sp[l1, p1] - 2*sp[l1, p2] + sp[p1, p1] + 
   2*sp[p1, p2] + sp[p2, p2])*(-sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2])*
  (3*sp[p1, p1]^2 + sp[p1, p2]^2 + 6*sp[p1, p2]*sp[p2, p2] + 3*sp[p2, p2]^2 + 
   sp[p1, p1]*(6*sp[p1, p2] + 8*sp[p2, p2])))
