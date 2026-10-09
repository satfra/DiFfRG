(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p1, l1 - p1 - p2, -l1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, l1 - p2, -l1}]*
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
   9*(27*cos[p1, l1]^5 + 27*cos[p1, l1]^4*(6*cos[p2, l1] + cos[p3, l1]) + 
     9*cos[p1, l1]^3*(2 + 33*cos[p2, l1]^2 + 14*cos[p2, l1]*cos[p3, l1] - 
       3*cos[p3, l1]^2) + 9*cos[p1, l1]^2*(27*cos[p2, l1]^3 + 
       20*cos[p2, l1]^2*cos[p3, l1] + cos[p2, l1]*(56 - 19*cos[p3, l1]^2) - 
       12*cos[p3, l1]*(-1 + cos[p3, l1]^2)) + 
     cos[p2, l1]*(428 + 81*cos[p2, l1]^4 + 81*cos[p2, l1]^3*cos[p3, l1] + 
       1200*cos[p3, l1]^2 - 162*cos[p3, l1]^4 - 9*cos[p2, l1]^2*
        (32 + 9*cos[p3, l1]^2) + cos[p2, l1]*(1200*cos[p3, l1] - 
         324*cos[p3, l1]^3)) + 3*cos[p1, l1]*(124 + 90*cos[p2, l1]^4 + 
       54*cos[p2, l1]^3*cos[p3, l1] + 36*cos[p3, l1]^2 - 18*cos[p3, l1]^4 - 
       cos[p2, l1]^2*(58 + 99*cos[p3, l1]^2) + cos[p2, l1]*
        (436*cos[p3, l1] - 144*cos[p3, l1]^3)))*sp[l1, l1]^(5/2)*
    Sqrt[sp[p, p]] - 9*(1016 + 27*cos[p1, l1]^5*cos[p2, l1] + 
     54*cos[p2, l1]^6 + 54*cos[p2, l1]^5*cos[p3, l1] + 606*cos[p3, l1]^2 - 
     36*cos[p3, l1]^4 + 9*cos[p1, l1]^4*(17 + 15*cos[p2, l1]^2 + 
       3*cos[p2, l1]*cos[p3, l1]) - 27*cos[p2, l1]^4*(13 + 2*cos[p3, l1]^2) - 
     36*cos[p2, l1]^3*cos[p3, l1]*(-59 + 6*cos[p3, l1]^2) + 
     cos[p2, l1]*(606*cos[p3, l1] - 72*cos[p3, l1]^3) + 
     cos[p2, l1]^2*(-473 + 2088*cos[p3, l1]^2 - 108*cos[p3, l1]^4) + 
     9*cos[p1, l1]^3*(24*cos[p2, l1]^3 - 20*cos[p3, l1] + 
       11*cos[p2, l1]^2*cos[p3, l1] - 3*cos[p2, l1]*(-33 + cos[p3, l1]^2)) + 
     3*cos[p1, l1]^2*(85 + 63*cos[p2, l1]^4 + 45*cos[p2, l1]^3*cos[p3, l1] - 
       72*cos[p3, l1]^2 + cos[p2, l1]^2*(394 - 48*cos[p3, l1]^2) - 
       6*cos[p2, l1]*cos[p3, l1]*(-23 + 6*cos[p3, l1]^2)) + 
     3*cos[p1, l1]*(63*cos[p2, l1]^5 + 202*cos[p3, l1] + 
       39*cos[p2, l1]^4*cos[p3, l1] - 24*cos[p3, l1]^3 - 
       3*cos[p2, l1]^3*(109 + 23*cos[p3, l1]^2) + cos[p2, l1]^2*
        (906*cos[p3, l1] - 108*cos[p3, l1]^3) + cos[p2, l1]*
        (368 + 174*cos[p3, l1]^2 - 18*cos[p3, l1]^4)))*sp[l1, l1]^2*
    sp[p, p] + 9*(153*cos[p1, l1]^4*cos[p2, l1] + 
     3*cos[p1, l1]^3*(137 + 321*cos[p2, l1]^2 - 60*cos[p2, l1]*cos[p3, l1]) + 
     cos[p1, l1]^2*(297*cos[p2, l1]^3 + 378*cos[p3, l1] + 
       246*cos[p2, l1]^2*cos[p3, l1] + cos[p2, l1]*
        (2027 - 216*cos[p3, l1]^2)) + cos[p2, l1]*(2800 - 72*cos[p2, l1]^4 + 
       1014*cos[p2, l1]^3*cos[p3, l1] + 1882*cos[p3, l1]^2 - 
       36*cos[p3, l1]^4 + 3*cos[p2, l1]^2*(-887 + 326*cos[p3, l1]^2) + 
       cos[p2, l1]*(1882*cos[p3, l1] - 72*cos[p3, l1]^3)) + 
     cos[p1, l1]*(-621*cos[p2, l1]^4 + 1440*cos[p2, l1]^3*cos[p3, l1] + 
       54*(40 + 7*cos[p3, l1]^2) + 3*cos[p2, l1]^2*
        (-527 + 118*cos[p3, l1]^2) + cos[p2, l1]*(2260*cos[p3, l1] - 
         72*cos[p3, l1]^3)))*sp[l1, l1]^(3/2)*sp[p, p]^(3/2) - 
   3*(1503*cos[p1, l1]^3*cos[p2, l1] + cos[p1, l1]^2*
      (4608 - 2532*cos[p2, l1]^2 + 954*cos[p2, l1]*cos[p3, l1]) + 
     4*(-1431*cos[p2, l1]^4 + 255*cos[p2, l1]*cos[p3, l1] + 
       846*cos[p2, l1]^3*cos[p3, l1] + 3*(381 + 85*cos[p3, l1]^2) + 
       2*cos[p2, l1]^2*(61 + 423*cos[p3, l1]^2)) + 
     cos[p1, l1]*(-11337*cos[p2, l1]^3 + 1020*cos[p3, l1] + 
       4338*cos[p2, l1]^2*cos[p3, l1] + cos[p2, l1]*
        (8508 + 954*cos[p3, l1]^2)))*sp[l1, l1]*sp[p, p]^2 + 
   2*(-3429*cos[p1, l1]^2*cos[p2, l1] + cos[p1, l1]*
      (7470 - 14592*cos[p2, l1]^2 + 1350*cos[p2, l1]*cos[p3, l1]) + 
     cos[p2, l1]*(9026 - 10869*cos[p2, l1]^2 + 1350*cos[p2, l1]*cos[p3, l1] + 
       1350*cos[p3, l1]^2))*Sqrt[sp[l1, l1]]*sp[p, p]^(5/2) + 
   8*(-584 + 777*cos[p1, l1]*cos[p2, l1] + 843*cos[p2, l1]^2)*sp[p, p]^3))/
 (882*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {l1 - p1 - p2, -l1 + p1 + p2}]*FunKit`dressing[FunKit`InverseProp, {A, A}, 
   1, {-l1 + p2, l1 - p2}]*(sp[l1, l1] - 2*cos[p2, l1]*Sqrt[sp[l1, l1]]*
    Sqrt[sp[p, p]] + sp[p, p])*(3*sp[l1, l1] - 6*(cos[p1, l1] + cos[p2, l1])*
    Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 4*sp[p, p]))
