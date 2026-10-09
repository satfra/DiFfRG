(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, l1 - p2, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p3, l1 - p2 - p3, -l1 + p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1 - p2 - p3, p1, l1, -l1 + p2 + p3}]*FunKit`dressing[FunKit`Rdot, 
   {A, A}, 1, {-l1, l1}]*(27*(-40 + 18*cos[p1, l1]^4 - 9*cos[p2, l1]^4 - 
     27*cos[p2, l1]^3*cos[p3, l1] + 21*cos[p3, l1]^2 - 9*cos[p3, l1]^4 + 
     36*cos[p1, l1]^3*(cos[p2, l1] + cos[p3, l1]) - 
     3*cos[p2, l1]^2*(-7 + 6*cos[p3, l1]^2) + cos[p1, l1]^2*
      (-66 + 9*cos[p2, l1]^2 + 30*cos[p2, l1]*cos[p3, l1] + 
       9*cos[p3, l1]^2) - cos[p2, l1]*cos[p3, l1]*(28 + 27*cos[p3, l1]^2) - 
     3*cos[p1, l1]*(3*cos[p2, l1]^3 + 5*cos[p2, l1]^2*cos[p3, l1] + 
       cos[p3, l1]*(22 + 3*cos[p3, l1]^2) + cos[p2, l1]*
        (22 + 5*cos[p3, l1]^2)))*sp[l1, l1]^3 + 
   9*(81*cos[p2, l1]^5 + 27*cos[p2, l1]^4*(3*cos[p1, l1] + 10*cos[p3, l1]) - 
     9*cos[p2, l1]^3*(32 + 9*cos[p1, l1]^2 - 18*cos[p1, l1]*cos[p3, l1] - 
       27*cos[p3, l1]^2) + 3*cos[p3, l1]*(124 - 18*cos[p1, l1]^4 - 
       36*cos[p1, l1]^3*cos[p3, l1] + 6*cos[p3, l1]^2 + 9*cos[p3, l1]^4 - 
       9*cos[p1, l1]^2*(-4 + cos[p3, l1]^2) + 9*cos[p1, l1]*cos[p3, l1]*
        (4 + cos[p3, l1]^2)) - 3*cos[p2, l1]^2*(108*cos[p1, l1]^3 + 
       58*cos[p3, l1] + 99*cos[p1, l1]^2*cos[p3, l1] - 99*cos[p3, l1]^3 - 
       20*cos[p1, l1]*(20 + 3*cos[p3, l1]^2)) + 
     cos[p2, l1]*(428 - 162*cos[p1, l1]^4 - 432*cos[p1, l1]^3*cos[p3, l1] + 
       504*cos[p3, l1]^2 + 162*cos[p3, l1]^4 + 6*cos[p1, l1]*cos[p3, l1]*
        (218 + 21*cos[p3, l1]^2) - 3*cos[p1, l1]^2*
        (-400 + 57*cos[p3, l1]^2)))*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
   9*(-1016 - 54*cos[p2, l1]^6 - 189*cos[p2, l1]^5*cos[p3, l1] - 
     255*cos[p3, l1]^2 - 153*cos[p3, l1]^4 + 18*cos[p1, l1]^4*
      (2 + 6*cos[p2, l1]^2 + 3*cos[p2, l1]*cos[p3, l1]) - 
     27*cos[p2, l1]^4*(-13 + 7*cos[p3, l1]^2) + 
     cos[p2, l1]^3*(981*cos[p3, l1] - 216*cos[p3, l1]^3) + 
     cos[p2, l1]^2*(473 - 1182*cos[p3, l1]^2 - 135*cos[p3, l1]^4) - 
     3*cos[p2, l1]*cos[p3, l1]*(368 + 297*cos[p3, l1]^2 + 9*cos[p3, l1]^4) + 
     3*cos[p1, l1]^2*(-202 + 18*cos[p2, l1]^4 + 69*cos[p2, l1]^3*
        cos[p3, l1] + 72*cos[p3, l1]^2 + 24*cos[p2, l1]^2*
        (-29 + 2*cos[p3, l1]^2) + 3*cos[p2, l1]*cos[p3, l1]*
        (-58 + 3*cos[p3, l1]^2)) + 36*cos[p1, l1]^3*(6*cos[p2, l1]^3 + 
       2*cos[p3, l1] + 9*cos[p2, l1]^2*cos[p3, l1] + 
       cos[p2, l1]*(2 + 3*cos[p3, l1]^2)) - 3*cos[p1, l1]*
      (18*cos[p2, l1]^5 + 202*cos[p3, l1] + 39*cos[p2, l1]^4*cos[p3, l1] - 
       60*cos[p3, l1]^3 + cos[p2, l1]^3*(708 + 45*cos[p3, l1]^2) + 
       cos[p2, l1]^2*(906*cos[p3, l1] + 33*cos[p3, l1]^3) + 
       cos[p2, l1]*(202 + 138*cos[p3, l1]^2 + 9*cos[p3, l1]^4)))*sp[l1, l1]^2*
    sp[p, p] - 9*(72*cos[p2, l1]^5 + cos[p2, l1]^4*(-1014*cos[p1, l1] + 
       621*cos[p3, l1]) - 3*cos[p2, l1]^3*(-887 + 326*cos[p1, l1]^2 + 
       480*cos[p1, l1]*cos[p3, l1] + 99*cos[p3, l1]^2) - 
     3*cos[p3, l1]*(720 + 126*cos[p1, l1]^2 + 126*cos[p1, l1]*cos[p3, l1] + 
       137*cos[p3, l1]^2) + cos[p2, l1]*(-2800 + 36*cos[p1, l1]^4 + 
       72*cos[p1, l1]^3*cos[p3, l1] - 2027*cos[p3, l1]^2 - 
       153*cos[p3, l1]^4 + 20*cos[p1, l1]*cos[p3, l1]*
        (-113 + 9*cos[p3, l1]^2) + 2*cos[p1, l1]^2*
        (-941 + 108*cos[p3, l1]^2)) + cos[p2, l1]^2*(72*cos[p1, l1]^3 + 
       1581*cos[p3, l1] - 354*cos[p1, l1]^2*cos[p3, l1] - 963*cos[p3, l1]^3 - 
       2*cos[p1, l1]*(941 + 123*cos[p3, l1]^2)))*sp[l1, l1]^(3/2)*
    sp[p, p]^(3/2) - 3*(4572 - 5724*cos[p2, l1]^4 - 
     11337*cos[p2, l1]^3*cos[p3, l1] + 4608*cos[p3, l1]^2 + 
     6*cos[p1, l1]^2*(170 + 564*cos[p2, l1]^2 + 159*cos[p2, l1]*
        cos[p3, l1]) + cos[p2, l1]^2*(488 - 2532*cos[p3, l1]^2) + 
     3*cos[p2, l1]*cos[p3, l1]*(2836 + 501*cos[p3, l1]^2) + 
     6*cos[p1, l1]*(564*cos[p2, l1]^3 + 170*cos[p3, l1] + 
       723*cos[p2, l1]^2*cos[p3, l1] + cos[p2, l1]*
        (170 + 159*cos[p3, l1]^2)))*sp[l1, l1]*sp[p, p]^2 + 
   2*(-10869*cos[p2, l1]^3 + 6*cos[p2, l1]^2*(225*cos[p1, l1] - 
       2432*cos[p3, l1]) + 7470*cos[p3, l1] + 
     cos[p2, l1]*(9026 + 1350*cos[p1, l1]^2 + 1350*cos[p1, l1]*cos[p3, l1] - 
       3429*cos[p3, l1]^2))*Sqrt[sp[l1, l1]]*sp[p, p]^(5/2) + 
   8*(-584 + 843*cos[p2, l1]^2 + 777*cos[p2, l1]*cos[p3, l1])*sp[p, p]^3))/
 (882*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p2, -l1 + p2}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {l1 - p2 - p3, -l1 + p2 + p3}]*(sp[l1, l1] - 
   2*cos[p2, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
  (3*sp[l1, l1] - 6*(cos[p2, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*
    Sqrt[sp[p, p]] + 4*sp[p, p]))
