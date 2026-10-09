(* Created with the Wolfram Language : www.wolfram.com *)
(3*FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1, p2, l1 + p1 - p2, -l1}]*FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 
   1, {-p2, p1, l1, -l1 - p1 + p2}]*FunKit`dressing[FunKit`Rdot, {A, A}, 1, 
   {-l1, l1}]*(2*sp[l1, p2]^4*sp[p1, p1]^2 + 2*sp[l1, p2]^3*sp[p1, p1]*
    (-2*sp[l1, p1]*sp[p1, p2] - 12*sp[p1, p2]^2 + 
     sp[p1, p1]*(sp[p1, p2] - 36*sp[p2, p2])) + 
   sp[l1, l1]^2*(30*sp[p1, p2]^4 + 20*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
     94*sp[p1, p1]^2*sp[p2, p2]^2) + sp[l1, p2]^2*
    (-2*sp[p1, p1]^2*sp[p1, p2]^2 + 4*sp[p1, p1]*sp[p1, p2]^3 - 
     13*sp[p1, p2]^4 + 31*sp[p1, p1]^3*sp[p2, p2] - 
     122*sp[p1, p1]^2*sp[p1, p2]*sp[p2, p2] + 32*sp[p1, p1]*sp[p1, p2]^2*
      sp[p2, p2] + 26*sp[p1, p1]^2*sp[p2, p2]^2 + 
     2*sp[l1, p1]^2*(sp[p1, p2]^2 + sp[p1, p1]*sp[p2, p2]) + 
     2*sp[l1, l1]*sp[p1, p1]*(13*sp[p1, p2]^2 + 34*sp[p1, p1]*sp[p2, p2]) + 
     sp[l1, p1]*(50*sp[p1, p2]^3 + 2*sp[p1, p1]*sp[p1, p2]*
        (11*sp[p1, p2] - 26*sp[p2, p2]) + 70*sp[p1, p1]^2*sp[p2, p2])) + 
   sp[l1, l1]*(138*sp[p1, p1]^3*sp[p2, p2]^2 + 
     sp[p1, p2]^4*(-32*sp[p1, p2] + 15*sp[p2, p2]) + 
     2*sp[l1, p1]^2*sp[p2, p2]*(13*sp[p1, p2]^2 + 34*sp[p1, p1]*sp[p2, p2]) + 
     sp[p1, p1]*sp[p1, p2]^2*(15*sp[p1, p2]^2 - 102*sp[p1, p2]*sp[p2, p2] + 
       71*sp[p2, p2]^2) + sp[p1, p1]^2*sp[p2, p2]*(71*sp[p1, p2]^2 - 
       314*sp[p1, p2]*sp[p2, p2] + 138*sp[p2, p2]^2) + 
     sp[l1, p1]*(34*sp[p1, p2]^4 + 94*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
       256*sp[p1, p1]^2*sp[p2, p2]^2 - 96*sp[p1, p1]*sp[p1, p2]*
        sp[p2, p2]^2)) + sp[l1, p1]^2*(-13*sp[p1, p2]^4 + 
     4*sp[p1, p2]^3*sp[p2, p2] + 2*sp[p1, p2]^2*(12*sp[l1, p1] + 
       16*sp[p1, p1] - sp[p2, p2])*sp[p2, p2] - 
     2*(sp[l1, p1] + 61*sp[p1, p1])*sp[p1, p2]*sp[p2, p2]^2 + 
     sp[p2, p2]^2*(2*sp[l1, p1]^2 + 72*sp[l1, p1]*sp[p1, p1] + 
       sp[p1, p1]*(26*sp[p1, p1] + 31*sp[p2, p2]))) + 
   sp[l1, p2]*(-2*sp[l1, l1]*(17*sp[p1, p2]^4 - 48*sp[p1, p1]^2*sp[p1, p2]*
        sp[p2, p2] + 47*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
       128*sp[p1, p1]^2*sp[p2, p2]^2 + sp[l1, p1]*(26*sp[p1, p2]^3 - 
         28*sp[p1, p1]*sp[p1, p2]*sp[p2, p2])) + 
     2*sp[l1, p1]*(-2*sp[l1, p1]^2*sp[p1, p2]*sp[p2, p2] + 
       2*sp[p1, p2]^3*(5*sp[p1, p2] + sp[p2, p2]) + sp[p1, p1]^2*sp[p2, p2]*
        (17*sp[p1, p2] + 6*sp[p2, p2]) + sp[p1, p1]*sp[p1, p2]*
        (2*sp[p1, p2]^2 - 10*sp[p1, p2]*sp[p2, p2] + 17*sp[p2, p2]^2) - 
       sp[l1, p1]*(25*sp[p1, p2]^3 - 26*sp[p1, p1]*sp[p1, p2]*sp[p2, p2] + 
         11*sp[p1, p2]^2*sp[p2, p2] + 35*sp[p1, p1]*sp[p2, p2]^2)))))/
 (4*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]^2*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {-l1 - p1 + p2, l1 + p1 - p2}]*sp[l1, l1]*(sp[l1, l1] + 2*sp[l1, p1] - 
   2*sp[l1, p2] + sp[p1, p1] - 2*sp[p1, p2] + sp[p2, p2])*
  (sp[p1, p2]^4 + 6*sp[p1, p1]*sp[p1, p2]^2*sp[p2, p2] + 
   11*sp[p1, p1]^2*sp[p2, p2]^2))
