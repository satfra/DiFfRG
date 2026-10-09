(* Created with the Wolfram Language : www.wolfram.com *)
(3*FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {lf1 - p1, p1, -lf1}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {-lf1 + p1, lf1, -p1}]*
  FunKit`dressing[FunKit`Rdot, {cb, c}, 1, {lf1, -lf1}]*
  (sp[lf1, p1]^2 - sp[lf1, lf1]*sp[p1, p1]))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {lf1 - p1, -lf1 + p1}]*
  FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {lf1, -lf1}]^2*
  (sp[lf1, lf1] - 2*sp[lf1, p1] + sp[p1, p1]))
