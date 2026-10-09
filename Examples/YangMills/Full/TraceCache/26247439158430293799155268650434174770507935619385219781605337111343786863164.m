(* Created with the Wolfram Language : www.wolfram.com *)
(3*FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {-l1, l1 + p1, -p1}]*
  FunKit`dressing[FunKit`GammaN, {A, cb, c}, 1, {l1, p1, -l1 - p1}]*
  FunKit`dressing[FunKit`Rdot, {cb, c}, 1, {l1 + p1, -l1 - p1}]*
  (sp[l1, p1]^2/sp[l1, l1] - sp[p1, p1]))/
 (FunKit`dressing[FunKit`InverseProp, {A, A}, 1, {-l1, l1}]*
  FunKit`dressing[FunKit`InverseProp, {cb, c}, 1, {l1 + p1, -l1 - p1}]^2)
