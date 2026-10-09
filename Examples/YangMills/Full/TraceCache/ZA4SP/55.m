(* Created with the Wolfram Language : www.wolfram.com *)
(FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, {p2, p1, l1 - p1 - p2, -l1}]*
  FunKit`dressing[FunKit`GammaN, {A, A, A, A}, 1, 
   {-p1 - p2 - p3, p3, l1, -l1 + p1 + p2}]*FunKit`dressing[FunKit`Rdot, 
   {A, A}, 1, {-l1, l1}]*(3*(1304 + 27*cos[p1, l1]^4 + 27*cos[p2, l1]^4 + 
     27*cos[p2, l1]^3*cos[p3, l1] + 786*cos[p3, l1]^2 - 54*cos[p3, l1]^4 + 
     27*cos[p1, l1]^3*(3*cos[p2, l1] + cos[p3, l1]) + 
     cos[p2, l1]^2*(399 - 27*cos[p3, l1]^2) + 3*cos[p1, l1]^2*
      (133 + 18*cos[p2, l1]^2 + 15*cos[p2, l1]*cos[p3, l1] - 
       9*cos[p3, l1]^2) + cos[p2, l1]*(786*cos[p3, l1] - 108*cos[p3, l1]^3) + 
     3*cos[p1, l1]*(27*cos[p2, l1]^3 + 262*cos[p3, l1] + 
       15*cos[p2, l1]^2*cos[p3, l1] - 36*cos[p3, l1]^3 + 
       cos[p2, l1]*(4 - 30*cos[p3, l1]^2)))*sp[l1, l1] - 
   9*(cos[p1, l1] + cos[p2, l1])*(872 + 151*cos[p1, l1]^2 + 
     151*cos[p2, l1]^2 + 282*cos[p2, l1]*cos[p3, l1] + 282*cos[p3, l1]^2 + 
     cos[p1, l1]*(20*cos[p2, l1] + 282*cos[p3, l1]))*Sqrt[sp[l1, l1]]*
    Sqrt[sp[p, p]] - 16*(-382 + 9*cos[p1, l1]^2 + 9*cos[p2, l1]^2 + 
     cos[p1, l1]*(75*cos[p2, l1] - 57*cos[p3, l1]) - 
     57*cos[p2, l1]*cos[p3, l1] - 57*cos[p3, l1]^2)*sp[p, p]))/
 (588*FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {l1, -l1}]*
  FunKit`dressing[FunKit`InverseProp, {A, A}, 1, 
   {l1 - p1 - p2, -l1 + p1 + p2}]*(3*sp[l1, l1] - 
   6*(cos[p1, l1] + cos[p2, l1])*Sqrt[sp[l1, l1]]*Sqrt[sp[p, p]] + 
   4*sp[p, p]))
