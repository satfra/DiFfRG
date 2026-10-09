(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, -l1, l1 - p1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {-p2, -l1 + p1, l1 - p1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1, p2, -l1 + p1 - p2, l1}]*FunKit`dressing[FunKit`Rdot, {A, A}, 1, 
   {l1, -l1}]*(-4*sp[l1, l1]^4*sp[p1, p1]*sp[p2, p2]*
    (sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
   sp[l1, l1]^3*(sp[l1, p2]^2*sp[p1, p1]*(2*sp[p1, p2]^2 + 
       5*sp[p1, p1]*sp[p2, p2]) + sp[l1, p1]^2*sp[p2, p2]*
      (2*sp[p1, p2]^2 + 5*sp[p1, p1]*sp[p2, p2]) + 
     sp[l1, p1]*(2*sp[p1, p2]^4 + 10*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
       9*sp[p1, p2]^3*sp[p2, p2] + 10*sp[p1, p1]^2*sp[p2, p2]^2 - 
       27*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2) - 
     sp[p1, p1]*(9*sp[p1, p2]^4 - 7*sp[p1, p2]^3*sp[p2, p2] + 
       35*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2 + 4*sp[p1, p2]^2*sp[p2, p2]*
        (-7*sp[p1, p1] + 2*sp[p2, p2]) + sp[p1, p1]*sp[p2, p2]^2*
        (15*sp[p1, p1] + 16*sp[p2, p2])) - 
     sp[l1, p2]*(2*sp[l1, p1]*(sp[p1, p2]^3 + 2*sp[p1, p1]*sp[p1, p2]*
          sp[p2, p2]) + sp[p1, p1]*(-7*sp[p1, p2]^3 + 33*sp[p1, p1]*
          sp[p1, p2]*sp[p2, p2] + 22*sp[p1, p2]^2*sp[p2, p2] + 
         12*sp[p1, p1]*sp[p2, p2]^2))) - 
   sp[l1, l1]^2*(sp[l1, p2]^4*sp[p1, p1]^2 + sp[l1, p1]^4*sp[p2, p2]^2 + 
     sp[l1, p1]^3*sp[p2, p2]*(5*sp[p1, p2]^2 + 13*sp[p1, p1]*sp[p2, p2] - 
       4*sp[p1, p2]*sp[p2, p2]) + sp[l1, p1]^2*(4*sp[p1, p2]^4 + 
       36*sp[p1, p2]^3*sp[p2, p2] - 100*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2 - 
       16*sp[p1, p1]*sp[p2, p2]^2*(sp[p1, p1] + sp[p2, p2]) - 
       sp[p1, p2]^2*sp[p2, p2]*(2*sp[p1, p1] + 3*sp[p2, p2])) - 
     sp[l1, p2]^3*sp[p1, p1]*(2*sp[l1, p1]*sp[p1, p2] + 9*sp[p1, p2]^2 + 
       sp[p1, p1]*(6*sp[p1, p2] + 13*sp[p2, p2])) + 
     sp[l1, p2]^2*(sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
       sp[l1, p1]*(9*sp[p1, p2]^3 + sp[p1, p1]*sp[p1, p2]*
          (20*sp[p1, p2] - 3*sp[p2, p2]) - 8*sp[p1, p1]^2*sp[p2, p2]) + 
       sp[p1, p1]*(4*sp[p1, p2]^3 + 56*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + 
         15*sp[p1, p2]^2*sp[p2, p2] + 2*sp[p1, p1]*(3*sp[p1, p1] - 
           5*sp[p2, p2])*sp[p2, p2])) - sp[l1, p1]*
      (45*sp[p1, p1]^3*sp[p2, p2]^2 + sp[p1, p2]^3*(-8*sp[p1, p2]^2 - 
         8*sp[p1, p2]*sp[p2, p2] + 5*sp[p2, p2]^2) + sp[p1, p1]^2*sp[p2, p2]*
        (-110*sp[p1, p2]^2 + 83*sp[p1, p2]*sp[p2, p2] + 36*sp[p2, p2]^2) + 
       sp[p1, p1]*sp[p1, p2]*(31*sp[p1, p2]^3 + 13*sp[p1, p2]^2*sp[p2, p2] + 
         48*sp[p1, p2]*sp[p2, p2]^2 - 15*sp[p2, p2]^3)) + 
     sp[p1, p1]*(23*sp[p1, p1]^3*sp[p2, p2]^2 + 7*sp[p1, p2]^3*
        (-2*sp[p1, p2]^2 - sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) + 
       sp[p1, p1]^2*sp[p2, p2]*(-48*sp[p1, p2]^2 + 31*sp[p1, p2]*sp[p2, p2] + 
         32*sp[p2, p2]^2) + sp[p1, p1]*sp[p1, p2]*(13*sp[p1, p2]^3 + 
         31*sp[p1, p2]^2*sp[p2, p2] - 97*sp[p1, p2]*sp[p2, p2]^2 + 
         41*sp[p2, p2]^3)) - sp[l1, p2]*(2*sp[l1, p1]^3*sp[p1, p2]*
        sp[p2, p2] + sp[l1, p1]^2*(5*sp[p1, p2]^3 + 23*sp[p1, p1]*sp[p1, p2]*
          sp[p2, p2] - 4*sp[p1, p2]^2*sp[p2, p2] + 36*sp[p1, p1]*
          sp[p2, p2]^2) + sp[l1, p1]*(sp[p1, p2]^3*(17*sp[p1, p2] + 
           7*sp[p2, p2]) + sp[p1, p1]^2*sp[p2, p2]*(69*sp[p1, p2] + 
           29*sp[p2, p2]) + sp[p1, p1]*sp[p1, p2]*(-13*sp[p1, p2]^2 + 
           58*sp[p1, p2]*sp[p2, p2] - 41*sp[p2, p2]^2)) - 
       sp[p1, p1]*(3*sp[p1, p1]^2*sp[p2, p2]*(10*sp[p1, p2] + 
           19*sp[p2, p2]) + sp[p1, p2]^2*(19*sp[p1, p2]^2 - 
           8*sp[p1, p2]*sp[p2, p2] + 12*sp[p2, p2]^2) + 
         2*sp[p1, p1]*(-4*sp[p1, p2]^3 - 22*sp[p1, p2]^2*sp[p2, p2] + 
           54*sp[p1, p2]*sp[p2, p2]^2 + sp[p2, p2]^3)))) + 
   sp[l1, l1]*(3*sp[l1, p1]^5*sp[p2, p2]^2 - sp[l1, p1]^4*sp[p2, p2]*
      (sp[p1, p2]^2 - sp[p1, p1]*sp[p2, p2] + 11*sp[p1, p2]*sp[p2, p2] + 
       sp[l1, p2]*(6*sp[p1, p2] + sp[p2, p2])) + 
     sp[l1, p1]^3*(-4*sp[p1, p2]^4 + 34*sp[p1, p2]^3*sp[p2, p2] - 
       125*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2 - 18*sp[p1, p1]*sp[p2, p2]^2*
        (3*sp[p1, p1] + 2*sp[p2, p2]) - sp[p1, p2]^2*sp[p2, p2]*
        (5*sp[p1, p1] + 7*sp[p2, p2]) + sp[l1, p2]^2*(3*sp[p1, p2]^2 + 
         3*sp[p1, p1]*sp[p2, p2] + 2*sp[p1, p2]*sp[p2, p2]) + 
       sp[l1, p2]*(sp[p1, p2]^3 + 21*sp[p1, p2]^2*sp[p2, p2] - 
         84*sp[p1, p1]*sp[p2, p2]^2 + sp[p1, p2]*sp[p2, p2]*
          (-35*sp[p1, p1] + 4*sp[p2, p2]))) - 
     sp[p1, p1]*(sp[l1, p2]^5*sp[p1, p1] + 5*sp[p1, p2]^5*
        (sp[p1, p2] - sp[p2, p2]) + 12*sp[p1, p1]^4*sp[p2, p2]^2 + 
       sp[p1, p1]*sp[p1, p2]^3*(-9*sp[p1, p2]^2 + 16*sp[p1, p2]*sp[p2, p2] - 
         4*sp[p2, p2]^2) + sp[p1, p1]^3*sp[p2, p2]*(-16*sp[p1, p2]^2 - 
         9*sp[p1, p2]*sp[p2, p2] + 16*sp[p2, p2]^2) + sp[p1, p1]^2*sp[p1, p2]*
        (4*sp[p1, p2]^3 + 18*sp[p1, p2]^2*sp[p2, p2] - 
         37*sp[p1, p2]*sp[p2, p2]^2 + 9*sp[p2, p2]^3) + 
       sp[l1, p2]^4*(sp[p1, p1]^2 - 5*sp[p1, p2]^2 - 
         sp[p1, p1]*(7*sp[p1, p2] + 6*sp[p2, p2])) + 
       sp[l1, p2]^3*(5*sp[p1, p2]^2*(2*sp[p1, p2] - sp[p2, p2]) + 
         sp[p1, p1]^2*(-6*sp[p1, p2] + 3*sp[p2, p2]) + 
         sp[p1, p1]*(sp[p1, p2]^2 + 16*sp[p1, p2]*sp[p2, p2] - 
           2*sp[p2, p2]^2)) + sp[l1, p2]^2*(11*sp[p1, p1]^3*sp[p2, p2] + 
         5*sp[p1, p2]^3*sp[p2, p2] + 2*sp[p1, p1]*sp[p1, p2]*
          (3*sp[p1, p2]^2 + 12*sp[p1, p2]*sp[p2, p2] + sp[p2, p2]^2) + 
         2*sp[p1, p1]^2*(sp[p1, p2]^2 + 2*sp[p1, p2]*sp[p2, p2] + 
           26*sp[p2, p2]^2)) - sp[l1, p2]*
        (5*sp[p1, p2]^4*(2*sp[p1, p2] - sp[p2, p2]) + 
         sp[p1, p1]^3*(3*sp[p1, p2] - 50*sp[p2, p2])*sp[p2, p2] + 
         sp[p1, p1]*sp[p1, p2]^2*(-8*sp[p1, p2]^2 + 50*sp[p1, p2]*
            sp[p2, p2] - 23*sp[p2, p2]^2) + sp[p1, p1]^2*
          (sp[p1, p2]^3 + 10*sp[p1, p2]^2*sp[p2, p2] + 22*sp[p1, p2]*
            sp[p2, p2]^2 - 34*sp[p2, p2]^3))) - 
     sp[l1, p1]^2*(15*sp[p1, p1]^3*sp[p2, p2]^2 - 6*sp[p1, p1]^2*sp[p2, p2]*
        (16*sp[p1, p2]^2 - 3*sp[p1, p2]*sp[p2, p2] + 4*sp[p2, p2]^2) + 
       sp[p1, p2]^3*(-12*sp[p1, p2]^2 - 11*sp[p1, p2]*sp[p2, p2] + 
         10*sp[p2, p2]^2) + sp[p1, p1]*sp[p1, p2]*(18*sp[p1, p2]^3 + 
         54*sp[p1, p2]^2*sp[p2, p2] + 75*sp[p1, p2]*sp[p2, p2]^2 - 
         37*sp[p2, p2]^3) + sp[l1, p2]^3*(sp[p1, p2]^2 + 
         sp[p1, p1]*(6*sp[p1, p2] + sp[p2, p2])) + 
       sp[l1, p2]^2*(59*sp[p1, p1]^2*sp[p2, p2] + sp[p1, p2]^2*
          (-13*sp[p1, p2] + 9*sp[p2, p2]) - sp[p1, p1]*(42*sp[p1, p2]^2 + 
           sp[p1, p2]*sp[p2, p2] + 32*sp[p2, p2]^2)) + 
       sp[l1, p2]*(3*sp[p1, p2]^3*(8*sp[p1, p2] + sp[p2, p2]) - 
         2*sp[p1, p1]^2*sp[p2, p2]*(16*sp[p1, p2] + 29*sp[p2, p2]) + 
         sp[p1, p1]*(18*sp[p1, p2]^3 + 44*sp[p1, p2]^2*sp[p2, p2] - 
           75*sp[p1, p2]*sp[p2, p2]^2 - 17*sp[p2, p2]^3))) + 
     sp[l1, p1]*(sp[l1, p2]^4*sp[p1, p1]*(3*sp[p1, p1] + 2*sp[p1, p2]) + 
       5*sp[p1, p2]^5*(sp[p1, p2] - sp[p2, p2]) + 45*sp[p1, p1]^4*
        sp[p2, p2]^2 + sp[p1, p1]*sp[p1, p2]^3*(-27*sp[p1, p2]^2 + 
         6*sp[p1, p2]*sp[p2, p2] + 8*sp[p2, p2]^2) + sp[p1, p1]^3*sp[p2, p2]*
        (-80*sp[p1, p2]^2 + 19*sp[p1, p2]*sp[p2, p2] + 36*sp[p2, p2]^2) + 
       sp[p1, p1]^2*sp[p1, p2]*(19*sp[p1, p2]^3 + 64*sp[p1, p2]^2*
          sp[p2, p2] - 101*sp[p1, p2]*sp[p2, p2]^2 + 35*sp[p2, p2]^3) + 
       sp[l1, p2]^3*(-5*sp[p1, p2]^3 + 11*sp[p1, p1]*sp[p1, p2]*
          (-3*sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]^2*(-16*sp[p1, p2] + 
           6*sp[p2, p2])) + sp[l1, p2]^2*
        (5*sp[p1, p2]^3*(2*sp[p1, p2] - sp[p2, p2]) + 50*sp[p1, p1]^3*
          sp[p2, p2] + sp[p1, p1]*sp[p1, p2]*(36*sp[p1, p2]^2 + 
           27*sp[p1, p2]*sp[p2, p2] + 5*sp[p2, p2]^2) + 
         sp[p1, p1]^2*(-7*sp[p1, p2]^2 + 17*sp[p1, p2]*sp[p2, p2] + 
           28*sp[p2, p2]^2)) + sp[l1, p2]*
        (5*sp[p1, p2]^4*(-2*sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]^3*sp[p2, p2]*(-11*sp[p1, p2] + 97*sp[p2, p2]) + 
         sp[p1, p1]*sp[p1, p2]^2*(22*sp[p1, p2]^2 - 41*sp[p1, p2]*
            sp[p2, p2] + 26*sp[p2, p2]^2) + sp[p1, p1]^2*
          (sp[p1, p2]^3 - 21*sp[p1, p2]^2*sp[p2, p2] + 43*sp[p1, p2]*
            sp[p2, p2]^2 + 21*sp[p2, p2]^3)))) + 
   sp[l1, p1]*(-2*sp[l1, p1]^5*sp[p2, p2]^2 + sp[l1, p1]^4*sp[p2, p2]*
      (6*sp[p1, p2]^2 + 9*sp[p1, p1]*sp[p2, p2] + 7*sp[p1, p2]*sp[p2, p2] + 
       sp[l1, p2]*(4*sp[p1, p2] + sp[p2, p2])) + 
     sp[l1, p1]^3*(8*sp[p1, p2]^4 + sp[p1, p2]^3*sp[p2, p2] + 
       60*sp[p1, p1]*sp[p1, p2]*sp[p2, p2]^2 - sp[p1, p2]^2*sp[p2, p2]*
        (11*sp[p1, p1] + 2*sp[p2, p2]) + 2*sp[p1, p1]*sp[p2, p2]^2*
        (19*sp[p1, p1] + 4*sp[p2, p2]) - 2*sp[l1, p2]^2*
        (sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2] + sp[p1, p2]*sp[p2, p2]) + 
       sp[l1, p2]*(-6*sp[p1, p2]^3 - 25*sp[p1, p2]^2*sp[p2, p2] + 
         4*sp[p1, p2]*(3*sp[p1, p1] - sp[p2, p2])*sp[p2, p2] + 
         34*sp[p1, p1]*sp[p2, p2]^2)) + sp[l1, p2]*sp[p1, p1]*
      (sp[l1, p2]^4*sp[p1, p1] + 5*sp[p1, p2]^4*(sp[p1, p2] - sp[p2, p2]) - 
       sp[p1, p1]^3*sp[p2, p2]*(12*sp[p1, p2] + 11*sp[p2, p2]) + 
       sp[p1, p1]^2*sp[p2, p2]*(24*sp[p1, p2]^2 + 11*sp[p1, p2]*sp[p2, p2] - 
         15*sp[p2, p2]^2) + sp[p1, p1]*sp[p1, p2]^2*(-5*sp[p1, p2]^2 + 
         12*sp[p1, p2]*sp[p2, p2] - 4*sp[p2, p2]^2) + 
       sp[l1, p2]^3*(sp[p1, p1]^2 - 5*sp[p1, p2]^2 - 
         sp[p1, p1]*(7*sp[p1, p2] + 6*sp[p2, p2])) + 
       sp[l1, p2]^2*(5*sp[p1, p2]^2*(2*sp[p1, p2] - sp[p2, p2]) + 
         sp[p1, p1]^2*(-6*sp[p1, p2] + 14*sp[p2, p2]) + 
         2*sp[p1, p1]*(5*sp[p1, p2]^2 + 8*sp[p1, p2]*sp[p2, p2] - 
           sp[p2, p2]^2)) + sp[l1, p2]*(10*sp[p1, p1]^3*sp[p2, p2] + 
         5*sp[p1, p2]^3*(-2*sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]^2*(5*sp[p1, p2]^2 - 37*sp[p1, p2]*sp[p2, p2] - 
           18*sp[p2, p2]^2) + sp[p1, p1]*sp[p1, p2]*(sp[p1, p2]^2 - 
           18*sp[p1, p2]*sp[p2, p2] + 2*sp[p2, p2]^2))) + 
     sp[l1, p1]^2*(4*sp[p1, p2]^4*(sp[p1, p2] - sp[p2, p2]) - 
       45*sp[p1, p1]^3*sp[p2, p2]^2 + sp[p1, p1]^2*sp[p2, p2]*
        (12*sp[p1, p2]^2 - 37*sp[p1, p2]*sp[p2, p2] - 36*sp[p2, p2]^2) + 
       sp[p1, p1]*sp[p1, p2]*(-12*sp[p1, p2]^3 + 17*sp[p1, p2]^2*sp[p2, p2] + 
         56*sp[p1, p2]*sp[p2, p2]^2 - 31*sp[p2, p2]^3) + 
       sp[l1, p2]^3*(sp[p1, p2]^2 + sp[p1, p1]*(4*sp[p1, p2] + sp[p2, p2])) + 
       sp[l1, p2]^2*(45*sp[p1, p1]^2*sp[p2, p2] + sp[p1, p2]^2*
          (8*sp[p1, p2] + 9*sp[p2, p2]) - sp[p1, p1]*(21*sp[p1, p2]^2 + 
           6*sp[p1, p2]*sp[p2, p2] + 32*sp[p2, p2]^2)) + 
       sp[l1, p2]*(sp[p1, p2]^3*(-13*sp[p1, p2] + sp[p2, p2]) - 
         3*sp[p1, p1]^2*sp[p2, p2]*(23*sp[p1, p2] + 36*sp[p2, p2]) + 
         sp[p1, p1]*(29*sp[p1, p2]^3 + 13*sp[p1, p2]^2*sp[p2, p2] - 
           43*sp[p1, p2]*sp[p2, p2]^2 - 17*sp[p2, p2]^3))) + 
     sp[l1, p1]*(-2*sp[l1, p2]^4*sp[p1, p1]*(sp[p1, p1] + sp[p1, p2]) + 
       sp[l1, p2]^3*(5*sp[p1, p2]^3 + sp[p1, p1]*sp[p1, p2]*
          (15*sp[p1, p2] - 11*sp[p2, p2]) + 10*sp[p1, p1]^2*
          (sp[p1, p2] - 3*sp[p2, p2])) + sp[l1, p2]^2*
        (-42*sp[p1, p1]^3*sp[p2, p2] + 5*sp[p1, p2]^3*sp[p2, p2] + 
         sp[p1, p1]*sp[p1, p2]*(-39*sp[p1, p2]^2 + 30*sp[p1, p2]*sp[p2, p2] - 
           5*sp[p2, p2]^2) + sp[p1, p1]^2*(sp[p1, p2]^2 + 
           82*sp[p1, p2]*sp[p2, p2] + 32*sp[p2, p2]^2)) + 
       2*sp[p1, p1]*(6*sp[p1, p1]^3*sp[p2, p2]^2 + 2*sp[p1, p2]^4*
          (-sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]^2*sp[p2, p2]*
          (-2*sp[p1, p2]^2 + sp[p1, p2]*sp[p2, p2] + 8*sp[p2, p2]^2) + 
         sp[p1, p1]*sp[p1, p2]*(2*sp[p1, p2]^3 - 3*sp[p1, p2]^2*sp[p2, p2] - 
           24*sp[p1, p2]*sp[p2, p2]^2 + 12*sp[p2, p2]^3)) + 
       sp[l1, p2]*(5*sp[p1, p2]^4*(-sp[p1, p2] + sp[p2, p2]) + 
         sp[p1, p1]^3*sp[p2, p2]*(55*sp[p1, p2] + 68*sp[p2, p2]) + 
         sp[p1, p1]*sp[p1, p2]^2*(30*sp[p1, p2]^2 - 34*sp[p1, p2]*
            sp[p2, p2] + 13*sp[p2, p2]^2) + sp[p1, p1]^2*(-13*sp[p1, p2]^3 - 
           54*sp[p1, p2]^2*sp[p2, p2] + 41*sp[p1, p2]*sp[p2, p2]^2 + 
           30*sp[p2, p2]^3))))))/(FunKit`dressing[FunKit`InverseProp, {A, A}, 
   1, {-l1, l1}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {-l1 + p1 - p2, l1 - p1 + p2}]*sp[l1, l1]*(sp[l1, l1] - 2*sp[l1, p1] + 
   sp[p1, p1])*(sp[l1, l1] - 2*sp[l1, p1] + 2*sp[l1, p2] + sp[p1, p1] - 
   2*sp[p1, p2] + sp[p2, p2])*(sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*
    sp[p2, p2] + 11*sp[p1, p1]^2*sp[p2, p2]^2))
