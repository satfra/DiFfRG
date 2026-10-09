(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1 - p2 - p3, p2, l1, -l1 + p1 + p3}]*FunKit`dressing[FunKit`GammaN, 
   {A, A, A, A}, 1, {p3, p1, l1 - p1 - p3, -l1}]*
  FunKit`dressing[FunKit`Rdot, {A, A}, 1, {-l1, l1}]*
  (3*(1304 + 27*cos[p1, l1]^4 - 54*cos[p2, l1]^4 - 
     108*cos[p2, l1]^3*cos[p3, l1] + 399*cos[p3, l1]^2 + 27*cos[p3, l1]^4 + 
     27*cos[p1, l1]^3*(cos[p2, l1] + 3*cos[p3, l1]) + 
     cos[p2, l1]^2*(786 - 27*cos[p3, l1]^2) + 3*cos[p2, l1]*cos[p3, l1]*
      (262 + 9*cos[p3, l1]^2) + cos[p1, l1]^2*(399 - 27*cos[p2, l1]^2 + 
       45*cos[p2, l1]*cos[p3, l1] + 54*cos[p3, l1]^2) + 
     cos[p1, l1]*(-108*cos[p2, l1]^3 - 90*cos[p2, l1]^2*cos[p3, l1] + 
       3*cos[p3, l1]*(4 + 27*cos[p3, l1]^2) + cos[p2, l1]*
        (786 + 45*cos[p3, l1]^2)))*sp[l1, l1] - 9*(cos[p1, l1] + cos[p3, l1])*
    (872 + 151*cos[p1, l1]^2 + 282*cos[p2, l1]^2 + 
     282*cos[p2, l1]*cos[p3, l1] + 151*cos[p3, l1]^2 + 
     cos[p1, l1]*(282*cos[p2, l1] + 20*cos[p3, l1]))*Sqrt[sp[l1, l1]]*
    Sqrt[sp[p, p]] - 16*(-382 + 9*cos[p1, l1]^2 - 57*cos[p2, l1]^2 - 
     57*cos[p2, l1]*cos[p3, l1] + 9*cos[p3, l1]^2 + 
     cos[p1, l1]*(-57*cos[p2, l1] + 75*cos[p3, l1]))*sp[p, p]))/
 (588*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {l1 - p1 - p3, -l1 + p1 + p3}]*(3*sp[l1, l1] - 
   6*(cos[p1, l1] + cos[p3, l1])*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
   4*sp[p, p]))
