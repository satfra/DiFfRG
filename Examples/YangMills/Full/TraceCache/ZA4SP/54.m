(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, l1 - p1 - p2, -l1 + p1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1 - p2 - p3, p3, l1, -l1 + p1 + p2}]*FunKit`dressing[FunKit`Rdot, 
   {A, A}, 1, {-l1, l1}]*(-27*(40 + 9*cos[p1, l1]^4 + 9*cos[p2, l1]^4 + 
     9*cos[p2, l1]^3*cos[p3, l1] + 66*cos[p3, l1]^2 - 18*cos[p3, l1]^4 + 
     9*cos[p1, l1]^3*(3*cos[p2, l1] + cos[p3, l1]) + 
     3*cos[p1, l1]^2*(-7 + 6*cos[p2, l1]^2 + 5*cos[p2, l1]*cos[p3, l1] - 
       3*cos[p3, l1]^2) - 3*cos[p2, l1]^2*(7 + 3*cos[p3, l1]^2) + 
     cos[p2, l1]*(66*cos[p3, l1] - 36*cos[p3, l1]^3) + 
     cos[p1, l1]*(27*cos[p2, l1]^3 + 66*cos[p3, l1] + 
       15*cos[p2, l1]^2*cos[p3, l1] - 36*cos[p3, l1]^3 + 
       cos[p2, l1]*(28 - 30*cos[p3, l1]^2)))*sp[l1, l1]^3 + 
   9*(81*cos[p1, l1]^5 + 27*cos[p1, l1]^4*(10*cos[p2, l1] + 3*cos[p3, l1]) + 
     9*cos[p1, l1]^3*(-32 + 27*cos[p2, l1]^2 + 18*cos[p2, l1]*cos[p3, l1] - 
       9*cos[p3, l1]^2) + 3*cos[p2, l1]*(124 + 9*cos[p2, l1]^4 + 
       9*cos[p2, l1]^3*cos[p3, l1] + 36*cos[p3, l1]^2 - 18*cos[p3, l1]^4 + 
       cos[p2, l1]^2*(6 - 9*cos[p3, l1]^2) - 36*cos[p2, l1]*cos[p3, l1]*
        (-1 + cos[p3, l1]^2)) + 3*cos[p1, l1]^2*(99*cos[p2, l1]^3 + 
       400*cos[p3, l1] + 60*cos[p2, l1]^2*cos[p3, l1] - 108*cos[p3, l1]^3 - 
       cos[p2, l1]*(58 + 99*cos[p3, l1]^2)) + 
     cos[p1, l1]*(428 + 162*cos[p2, l1]^4 + 126*cos[p2, l1]^3*cos[p3, l1] + 
       1200*cos[p3, l1]^2 - 162*cos[p3, l1]^4 - 9*cos[p2, l1]^2*
        (-56 + 19*cos[p3, l1]^2) + cos[p2, l1]*(1308*cos[p3, l1] - 
         432*cos[p3, l1]^3)))*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] - 
   9*(1016 + 54*cos[p1, l1]^6 + 153*cos[p2, l1]^4 - 
     180*cos[p2, l1]^3*cos[p3, l1] + 606*cos[p3, l1]^2 - 36*cos[p3, l1]^4 + 
     27*cos[p1, l1]^5*(7*cos[p2, l1] + 2*cos[p3, l1]) + 
     9*cos[p1, l1]^4*(-39 + 21*cos[p2, l1]^2 + 13*cos[p2, l1]*cos[p3, l1] - 
       6*cos[p3, l1]^2) - 3*cos[p2, l1]^2*(-85 + 72*cos[p3, l1]^2) + 
     cos[p2, l1]*(606*cos[p3, l1] - 72*cos[p3, l1]^3) + 
     9*cos[p1, l1]^3*(24*cos[p2, l1]^3 + 15*cos[p2, l1]^2*cos[p3, l1] + 
       4*cos[p3, l1]*(59 - 6*cos[p3, l1]^2) - cos[p2, l1]*
        (109 + 23*cos[p3, l1]^2)) + cos[p1, l1]^2*(-473 + 135*cos[p2, l1]^4 + 
       99*cos[p2, l1]^3*cos[p3, l1] + 2088*cos[p3, l1]^2 - 
       108*cos[p3, l1]^4 - 6*cos[p2, l1]^2*(-197 + 24*cos[p3, l1]^2) + 
       cos[p2, l1]*(2718*cos[p3, l1] - 324*cos[p3, l1]^3)) + 
     3*cos[p1, l1]*(9*cos[p2, l1]^5 + 202*cos[p3, l1] + 
       9*cos[p2, l1]^4*cos[p3, l1] - 24*cos[p3, l1]^3 - 
       9*cos[p2, l1]^3*(-33 + cos[p3, l1]^2) - 6*cos[p2, l1]^2*cos[p3, l1]*
        (-23 + 6*cos[p3, l1]^2) + cos[p2, l1]*(368 + 174*cos[p3, l1]^2 - 
         18*cos[p3, l1]^4)))*sp[l1, l1]^2*sp[p, p] - 
   9*(72*cos[p1, l1]^5 + 3*cos[p1, l1]^4*(207*cos[p2, l1] - 
       338*cos[p3, l1]) - 3*cos[p2, l1]*(720 + 137*cos[p2, l1]^2 + 
       126*cos[p2, l1]*cos[p3, l1] + 126*cos[p3, l1]^2) - 
     3*cos[p1, l1]^3*(-887 + 99*cos[p2, l1]^2 + 480*cos[p2, l1]*cos[p3, l1] + 
       326*cos[p3, l1]^2) - cos[p1, l1]^2*(963*cos[p2, l1]^3 + 
       1882*cos[p3, l1] + 246*cos[p2, l1]^2*cos[p3, l1] - 72*cos[p3, l1]^3 + 
       3*cos[p2, l1]*(-527 + 118*cos[p3, l1]^2)) + 
     cos[p1, l1]*(-2800 - 153*cos[p2, l1]^4 + 180*cos[p2, l1]^3*cos[p3, l1] - 
       1882*cos[p3, l1]^2 + 36*cos[p3, l1]^4 + 4*cos[p2, l1]*cos[p3, l1]*
        (-565 + 18*cos[p3, l1]^2) + cos[p2, l1]^2*
        (-2027 + 216*cos[p3, l1]^2)))*sp[l1, l1]^(3/2)*sp[p, p]^(3/2) + 
   3*(5724*cos[p1, l1]^4 + 3*cos[p1, l1]^3*(3779*cos[p2, l1] - 
       1128*cos[p3, l1]) - 12*(381 + 384*cos[p2, l1]^2 + 
       85*cos[p2, l1]*cos[p3, l1] + 85*cos[p3, l1]^2) - 
     3*cos[p1, l1]*(501*cos[p2, l1]^3 + 340*cos[p3, l1] + 
       318*cos[p2, l1]^2*cos[p3, l1] + cos[p2, l1]*
        (2836 + 318*cos[p3, l1]^2)) + cos[p1, l1]^2*(2532*cos[p2, l1]^2 - 
       4338*cos[p2, l1]*cos[p3, l1] - 8*(61 + 423*cos[p3, l1]^2)))*sp[l1, l1]*
    sp[p, p]^2 + 2*(-10869*cos[p1, l1]^3 + 7470*cos[p2, l1] - 
     6*cos[p1, l1]^2*(2432*cos[p2, l1] - 225*cos[p3, l1]) + 
     cos[p1, l1]*(9026 - 3429*cos[p2, l1]^2 + 1350*cos[p2, l1]*cos[p3, l1] + 
       1350*cos[p3, l1]^2))*Sqrt[sp[l1, l1]]*sp[p, p]^(5/2) + 
   8*(-584 + 843*cos[p1, l1]^2 + 777*cos[p1, l1]*cos[p2, l1])*sp[p, p]^3))/
 (882*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p1, -l1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {l1 - p1 - p2, -l1 + p1 + p2}]*(sp[l1, l1] - 
   2*cos[p1, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
  (3*sp[l1, l1] - 6*(cos[p1, l1] + cos[p2, l1])*Sqrt[sp[l1, l1]]*
    Sqrt[sp[p, p]] + 4*sp[p, p]))
