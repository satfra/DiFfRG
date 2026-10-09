(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p2, -l1, l1 - p2}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A}, 1, {p3, -l1 - p3, l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1 - p2 - p3, p1, -l1 + p2, l1 + p3}]*FunKit`dressing[FunKit`Rdot, 
   {A, A}, 1, {l1, -l1}]*(3*(-40 + 18*cos[p1, l1]^4 - 9*cos[p2, l1]^4 - 
     27*cos[p2, l1]^3*cos[p3, l1] + 21*cos[p3, l1]^2 - 9*cos[p3, l1]^4 + 
     36*cos[p1, l1]^3*(cos[p2, l1] + cos[p3, l1]) - 
     3*cos[p2, l1]^2*(-7 + 6*cos[p3, l1]^2) + cos[p1, l1]^2*
      (-66 + 9*cos[p2, l1]^2 + 30*cos[p2, l1]*cos[p3, l1] + 
       9*cos[p3, l1]^2) - cos[p2, l1]*cos[p3, l1]*(28 + 27*cos[p3, l1]^2) - 
     3*cos[p1, l1]*(3*cos[p2, l1]^3 + 5*cos[p2, l1]^2*cos[p3, l1] + 
       cos[p3, l1]*(22 + 3*cos[p3, l1]^2) + cos[p2, l1]*
        (22 + 5*cos[p3, l1]^2)))*sp[l1, l1]^3 + 3*(cos[p2, l1] - cos[p3, l1])*
    (104 - 18*cos[p1, l1]^4 + 9*cos[p2, l1]^4 + 27*cos[p2, l1]^3*
      cos[p3, l1] - 6*cos[p3, l1]^2 + 9*cos[p3, l1]^4 - 
     36*cos[p1, l1]^3*(cos[p2, l1] + cos[p3, l1]) + 
     6*cos[p2, l1]^2*(-1 + 3*cos[p3, l1]^2) - 3*cos[p1, l1]^2*
      (-8 + 3*cos[p2, l1]^2 + 10*cos[p2, l1]*cos[p3, l1] + 3*cos[p3, l1]^2) + 
     cos[p2, l1]*cos[p3, l1]*(284 + 27*cos[p3, l1]^2) + 
     3*cos[p1, l1]*(3*cos[p2, l1]^3 + 5*cos[p2, l1]^2*cos[p3, l1] + 
       cos[p3, l1]*(8 + 3*cos[p3, l1]^2) + cos[p2, l1]*
        (8 + 5*cos[p3, l1]^2)))*sp[l1, l1]^(5/2)*Sqrt[sp[p, p]] + 
   (-1072 - 54*cos[p1, l1]^4*cos[p2, l1]*cos[p3, l1] + 
     27*cos[p2, l1]^5*cos[p3, l1] + 402*cos[p3, l1]^2 - 99*cos[p3, l1]^4 - 
     108*cos[p1, l1]^3*cos[p2, l1]*cos[p3, l1]*(cos[p2, l1] + cos[p3, l1]) + 
     9*cos[p2, l1]^4*(-11 + 9*cos[p3, l1]^2) + 
     cos[p2, l1]^3*(-789*cos[p3, l1] + 54*cos[p3, l1]^3) + 
     cos[p2, l1]*cos[p3, l1]*(-772 - 789*cos[p3, l1]^2 + 27*cos[p3, l1]^4) + 
     3*cos[p2, l1]^2*(134 + 602*cos[p3, l1]^2 + 27*cos[p3, l1]^4) - 
     3*cos[p1, l1]^2*(24 + 9*cos[p2, l1]^3*cos[p3, l1] - 78*cos[p3, l1]^2 + 
       6*cos[p2, l1]^2*(-13 + 5*cos[p3, l1]^2) + cos[p2, l1]*cos[p3, l1]*
        (34 + 9*cos[p3, l1]^2)) + 3*cos[p1, l1]*
      (9*cos[p2, l1]^4*cos[p3, l1] + 3*cos[p2, l1]^3*(26 + 5*cos[p3, l1]^2) + 
       6*cos[p3, l1]*(-4 + 13*cos[p3, l1]^2) + cos[p2, l1]^2*cos[p3, l1]*
        (44 + 15*cos[p3, l1]^2) + cos[p2, l1]*(-24 + 44*cos[p3, l1]^2 + 
         9*cos[p3, l1]^4)))*sp[l1, l1]^2*sp[p, p] - 
   (cos[p2, l1] - cos[p3, l1])*(99*cos[p2, l1]^3*cos[p3, l1] + 
     cos[p1, l1]^2*(84 - 234*cos[p2, l1]*cos[p3, l1]) + 
     6*(-234 + 49*cos[p3, l1]^2) + cos[p2, l1]*cos[p3, l1]*
      (-3136 + 99*cos[p3, l1]^2) + 6*cos[p2, l1]^2*(49 + 130*cos[p3, l1]^2) - 
     6*cos[p1, l1]*(-14*cos[p3, l1] + 39*cos[p2, l1]^2*cos[p3, l1] + 
       cos[p2, l1]*(-14 + 39*cos[p3, l1]^2)))*sp[l1, l1]^(3/2)*
    sp[p, p]^(3/2) - 2*(682 + 102*cos[p2, l1]^3*cos[p3, l1] - 
     203*cos[p3, l1]^2 + 24*cos[p1, l1]^2*(-1 + 3*cos[p2, l1]*cos[p3, l1]) + 
     2*cos[p2, l1]*cos[p3, l1]*(251 + 51*cos[p3, l1]^2) - 
     7*cos[p2, l1]^2*(29 + 330*cos[p3, l1]^2) + 
     24*cos[p1, l1]*(-cos[p3, l1] + 3*cos[p2, l1]^2*cos[p3, l1] + 
       cos[p2, l1]*(-1 + 3*cos[p3, l1]^2)))*sp[l1, l1]*sp[p, p]^2 + 
   772*(-cos[p3, l1] + 3*cos[p2, l1]^2*cos[p3, l1] + 
     cos[p2, l1]*(1 - 3*cos[p3, l1]^2))*Sqrt[sp[l1, l1]]*sp[p, p]^(5/2) - 
   392*(1 + 3*cos[p2, l1]*cos[p3, l1])*sp[p, p]^3))/
 (294*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1 - p2, -l1 + p2}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1 - p3, l1 + p3}]*
  (sp[l1, l1] - 2*cos[p2, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p])*
  (sp[l1, l1] + 2*cos[p3, l1]*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + sp[p, p]))
