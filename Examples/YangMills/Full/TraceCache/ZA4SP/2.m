(* Created with the Wolfram Language : www.wolfram.com *)
-1/49*(FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {p1, -lf1, lf1 - p1}]*
   FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, 
    {p2, -lf1 + p1, lf1 - p1 - p2}]*FunKit`dressing[FunKit`GammaN, 
    {A, cb, c}, 1, {-p1 - p2 - p3, -lf1 + p1 + p2 + p3, lf1}]*
   FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {p3, -lf1 + p1 + p2, 
     lf1 - p1 - p2 - p3}]*FunKit`dressing[FunKit`Rdot, {cb, c}, 1, 
    {-lf1, lf1}]*sp[lf1, lf1]*(-9*cos[p3, lf1]^2*sp[lf1, lf1] + 
    9*(3*cos[p1, lf1]^4 + 3*cos[p1, lf1]^3*(cos[p2, lf1] + 3*cos[p3, lf1]) + 
      cos[p1, lf1]^2*(-3*cos[p2, lf1]^2 + 5*cos[p2, lf1]*cos[p3, lf1] + 
        6*cos[p3, lf1]^2) + cos[p1, lf1]*(-12*cos[p2, lf1]^3 - 
        10*cos[p2, lf1]^2*cos[p3, lf1] + 5*cos[p2, lf1]*cos[p3, lf1]^2 + 
        9*cos[p3, lf1]^3) + 3*(-2*cos[p2, lf1]^4 - 4*cos[p2, lf1]^3*
         cos[p3, lf1] - cos[p2, lf1]^2*cos[p3, lf1]^2 + 
        cos[p2, lf1]*cos[p3, lf1]^3 + cos[p3, lf1]^4))*sp[lf1, lf1] + 
    12*(-cos[p2, lf1] + cos[p3, lf1])*Sqrt[sp[lf1, lf1]]*Sqrt[sp[p, p]] + 
    3*(-12*cos[p1, lf1]^3 - cos[p1, lf1]^2*(17*cos[p2, lf1] + 
        19*cos[p3, lf1]) + cos[p1, lf1]*(7*cos[p2, lf1]^2 + 
        5*cos[p3, lf1]^2) + 3*(6*cos[p2, lf1]^3 + 7*cos[p2, lf1]^2*
         cos[p3, lf1] + cos[p2, lf1]*cos[p3, lf1]^2 - 2*cos[p3, lf1]^3))*
     Sqrt[sp[lf1, lf1]]*Sqrt[sp[p, p]] - 16*sp[p, p] + 
    9*cos[p2, lf1]^2*(2*sp[lf1, lf1] + sp[p, p]) + 
    6*cos[p2, lf1]*cos[p3, lf1]*(3*sp[lf1, lf1] + sp[p, p]) + 
    cos[p1, lf1]^2*(-9*sp[lf1, lf1] + 43*sp[p, p]) + 
    2*cos[p1, lf1]*(cos[p3, lf1]*(-18*sp[lf1, lf1] + 17*sp[p, p]) + 
      cos[p2, lf1]*(9*sp[lf1, lf1] + 26*sp[p, p]))))/
  (FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {-lf1, lf1}]^2*
   FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {-lf1 + p1, lf1 - p1}]*
   FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, 
    {-lf1 + p1 + p2, lf1 - p1 - p2}]*FunKit`dressing[FunKit`InverseProp, 
    {cb, c}, 1, {-lf1 + p1 + p2 + p3, lf1 - p1 - p2 - p3}])
