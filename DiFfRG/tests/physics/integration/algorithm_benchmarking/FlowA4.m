(* Created with the Wolfram Language : www.wolfram.com *)
{(1.653061224489796*l1^2*(1.*cosl1p1^4*l1^2 - 2.*cosl1p2^4*l1^2 - 
    0.3333333333333333*cosl1p3^2*l1^2 + 1.*cosl1p3^4*l1^2 + 
    cosl1p1^3*l1*(1.*cosl1p2*l1 + 3.*cosl1p3*l1 - 1.3333333333333333*p) + 
    0.4444444444444444*cosl1p3*l1*p - 0.6666666666666666*cosl1p3^3*l1*p - 
    0.5925925925925926*p^2 + cosl1p2^3*l1*(-4.*cosl1p3*l1 + 2.*p) + 
    cosl1p2^2*((0.6666666666666666 - 1.*cosl1p3^2)*l1^2 + 
      2.3333333333333335*cosl1p3*l1*p + 0.3333333333333333*p^2) + 
    cosl1p2*(0.6666666666666666*cosl1p3*l1^2 + 1.*cosl1p3^3*l1^2 - 
      0.4444444444444444*l1*p + 0.3333333333333333*cosl1p3^2*l1*p + 
      0.2222222222222222*cosl1p3*p^2) + 
    cosl1p1^2*((-0.3333333333333333 - 1.*cosl1p2^2 + 1.6666666666666667*
         cosl1p2*cosl1p3 + 2.*cosl1p3^2)*l1^2 + 1.5925925925925926*p^2 + 
      l1*(-1.8888888888888888*cosl1p2*p - 2.111111111111111*cosl1p3*p)) + 
    cosl1p1*(-4.*cosl1p2^3*l1^2 + cosl1p2^2*l1*
       (-3.3333333333333335*cosl1p3*l1 + 0.7777777777777778*p) + 
      cosl1p3*((-1.3333333333333333 + 3.*cosl1p3^2)*l1^2 + 
        0.5555555555555556*cosl1p3*l1*p + 1.2592592592592593*p^2) + 
      cosl1p2*((0.6666666666666666 + 1.6666666666666667*cosl1p3^2)*l1^2 + 
        1.9259259259259258*p^2)))*
   ZAcbc[Sqrt[2/3]*Sqrt[l1^2 - cosl1p1*l1*p + p^2]]*
   ZAcbc[Sqrt[2/3]*Sqrt[l1^2 - (cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]]*
   ZAcbc[Sqrt[6*l1^2 - 6*(2*cosl1p1 + cosl1p2)*l1*p + 10*p^2]/3]*
   ZAcbc[Sqrt[6*l1^2 - 6*(2*cosl1p1 + 2*cosl1p2 + cosl1p3)*l1*p + 10*p^2]/3]*
   (dtZc[k]*RB[k^2, l1^2] + RBdot[k^2, l1^2]*Zc[k] + 
    RB[k^2, l1^2]*(-50.*Zc[k] + 50.*Zc[1.02*k])))/
  ((RB[k^2, l1^2]*Zc[k] + l1^2*Zc[l1])^2*
   (RB[k^2, l1^2 - 2*cosl1p1*l1*p + p^2]*Zc[k] + 
    (l1^2 - 2.*cosl1p1*l1*p + p^2)*Zc[Sqrt[l1^2 - 2*cosl1p1*l1*p + p^2]])*
   (RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]*Zc[k] + 
    (l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*p + p^2)*
     Zc[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]])*
   (3.*RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2)*l1*p + (4*p^2)/3]*Zc[k] + 
    (3.*l1^2 - 6.*cosl1p1*l1*p - 6.*cosl1p2*l1*p + 4.*p^2)*
     Zc[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2)*l1*p + (4*p^2)/3]])), 
 (-0.16326530612244897*(-243.*cosl1p3^2*l1^10 + 729.*cosl1p3^4*l1^10 + 
    324.*cosl1p3*l1^9*p - 1458.*cosl1p3^5*l1^9*p - 432.*l1^8*p^2 - 
    756.*cosl1p3^2*l1^8*p^2 + 3645.*cosl1p3^4*l1^8*p^2 + 
    243.*cosl1p3^6*l1^8*p^2 + 972.*cosl1p2^7*l1^7*p^3 + 
    1008.*cosl1p3*l1^7*p^3 - 54.*cosl1p3^3*l1^7*p^3 - 
    3726.*cosl1p3^5*l1^7*p^3 - 3756.*l1^6*p^4 + 972.*cosl1p1^8*l1^6*p^4 + 
    1764.*cosl1p3^2*l1^6*p^4 + 4050.*cosl1p3^4*l1^6*p^4 + 
    162.*cosl1p3^6*l1^6*p^4 + 3624.*cosl1p3*l1^5*p^5 - 
    1206.*cosl1p3^3*l1^5*p^5 - 2052.*cosl1p3^5*l1^5*p^5 - 7160.*l1^4*p^6 + 
    4128.*cosl1p3^2*l1^4*p^6 + 1548.*cosl1p3^4*l1^4*p^6 + 
    3712.*cosl1p3*l1^3*p^7 - 1248.*cosl1p3^3*l1^3*p^7 - 4834.*l1^2*p^8 + 
    2176.*cosl1p3^2*l1^2*p^8 + 836.*cosl1p3*l1*p^9 - 1024.*p^10 + 
    cosl1p2^6*l1^6*p^2*(-6318.*l1^2 + 3402.*cosl1p3*l1*p - 9072.*p^2) + 
    cosl1p1^7*l1^5*p^3*(-6804.*l1^2 + 3402.*cosl1p2*l1*p + 
      4374.*cosl1p3*l1*p - 5184.*p^2) + cosl1p2^5*l1^5*p*
     (5832.*l1^4 - 18954.*cosl1p3*l1^3*p + (20898. + 3888.*cosl1p3^2)*l1^2*
       p^2 - 27216.*cosl1p3*l1*p^3 + 19926.*p^4) + 
    cosl1p2^4*l1^4*(-1458.*l1^6 + 14580.*cosl1p3*l1^5*p + 
      (-9234. - 16281.*cosl1p3^2)*l1^4*p^2 + 
      cosl1p3*(50625. + 1215.*cosl1p3^2)*l1^3*p^3 + 
      (-25353. - 22113.*cosl1p3^2)*l1^2*p^4 + 49869.*cosl1p3*l1*p^5 - 
      17802.*p^6) + cosl1p2^2*l1^2*((486. - 729.*cosl1p3^2)*l1^8 + 
      cosl1p3*(-1215. - 1458.*cosl1p3^2)*l1^7*p + 
      (1755. - 4131.*cosl1p3^2 + 6075.*cosl1p3^4)*l1^6*p^2 + 
      cosl1p3*(8586. - 10449.*cosl1p3^2 - 972.*cosl1p3^4)*l1^5*p^3 + 
      (-3528. - 22275.*cosl1p3^2 + 9153.*cosl1p3^4)*l1^4*p^4 + 
      cosl1p3*(17712. - 1404.*cosl1p3^2)*l1^3*p^5 + 
      (-17742. - 15012.*cosl1p3^2)*l1^2*p^6 + 10824.*cosl1p3*l1*p^7 - 
      11368.*p^8) + cosl1p1^2*
     ((-243. - 729.*cosl1p2^2 + 1215.*cosl1p2*cosl1p3 + 1458.*cosl1p3^2)*
       l1^10 + (20412.*cosl1p2^3 + 4779.*cosl1p3 + 11178.*cosl1p2^2*cosl1p3 - 
        16038.*cosl1p3^3 + cosl1p2*(-3321. - 15552.*cosl1p3^2))*l1^9*p + 
      (1053. - 68769.*cosl1p2^4 - 97848.*cosl1p2^3*cosl1p3 - 
        8586.*cosl1p3^2 + 27702.*cosl1p3^4 + cosl1p2^2*
         (14013. + 1944.*cosl1p3^2) + cosl1p2*(7209.*cosl1p3 + 
          61560.*cosl1p3^3))*l1^8*p^2 + (55404.*cosl1p2^5 - 14148.*cosl1p3 + 
        120123.*cosl1p2^4*cosl1p3 - 28323.*cosl1p3^3 - 9234.*cosl1p3^5 + 
        cosl1p2^3*(54648. + 55080.*cosl1p3^2) + 
        cosl1p2^2*(33291.*cosl1p3 - 39042.*cosl1p3^3) + 
        cosl1p2*(-26838. - 21870.*cosl1p3^2 - 43983.*cosl1p3^4))*l1^7*p^3 + 
      (4644. - 5832.*cosl1p2^6 - 16686.*cosl1p2^5*cosl1p3 + 
        10881.*cosl1p3^2 + 35721.*cosl1p3^4 + 486.*cosl1p3^6 + 
        cosl1p2^4*(-121689. - 14256.*cosl1p3^2) + 
        cosl1p2^3*(-188892.*cosl1p3 - 324.*cosl1p3^3) + 
        cosl1p2^2*(48708. - 21816.*cosl1p3^2 + 6237.*cosl1p3^4) + 
        cosl1p2*(90351.*cosl1p3 + 77085.*cosl1p3^3 + 3402.*cosl1p3^5))*l1^6*
       p^4 + (21762.*cosl1p2^5 - 90102.*cosl1p3 + 51111.*cosl1p2^4*cosl1p3 - 
        3474.*cosl1p3^3 - 3240.*cosl1p3^5 + cosl1p2^3*
         (42561. + 25974.*cosl1p3^2) + cosl1p2^2*(-6372.*cosl1p3 - 
          11988.*cosl1p3^3) + cosl1p2*(-70008. - 28809.*cosl1p3^2 - 
          12825.*cosl1p3^4))*l1^5*p^5 + (-5424. - 16830.*cosl1p2^4 - 
        24525.*cosl1p2^3*cosl1p3 + 69840.*cosl1p3^2 - 702.*cosl1p3^4 + 
        cosl1p2^2*(110952. - 11736.*cosl1p3^2) + 
        cosl1p2*(225468.*cosl1p3 - 5958.*cosl1p3^3))*l1^4*p^6 + 
      (-51858.*cosl1p2^3 - 94120.*cosl1p3 - 106866.*cosl1p2^2*cosl1p3 + 
        2700.*cosl1p3^3 + cosl1p2*(-65252. - 48816.*cosl1p3^2))*l1^3*p^7 + 
      (-1594. + 82452.*cosl1p2^2 + 115242.*cosl1p2*cosl1p3 + 
        33312.*cosl1p3^2)*l1^2*p^8 + (-34188.*cosl1p2 - 27480.*cosl1p3)*l1*
       p^9 + 3072.*p^10) + cosl1p1^6*l1^4*p^2*(8991.*l1^4 + 
      (23328. + 3402.*cosl1p2^2 + 12555.*cosl1p2*cosl1p3 + 6804.*cosl1p3^2)*
       l1^2*p^2 + (-22086.*cosl1p2 - 14202.*cosl1p3)*l1*p^3 + 21960.*p^4 + 
      l1^3*(-20169.*cosl1p2*p - 27459.*cosl1p3*p)) + 
    cosl1p1^4*l1^2*(729.*l1^8 + (5589. + 6075.*cosl1p2^2 + 
        59940.*cosl1p2*cosl1p3 + 36450.*cosl1p3^2)*l1^6*p^2 + 
      (33048.*cosl1p2^3 - 46602.*cosl1p3 - 21708.*cosl1p2^2*cosl1p3 - 
        36693.*cosl1p3^3 + cosl1p2*(-61128. - 72657.*cosl1p3^2))*l1^5*p^3 + 
      (49581. - 13122.*cosl1p2^4 - 14580.*cosl1p2^3*cosl1p3 + 
        36855.*cosl1p3^2 + 6318.*cosl1p3^4 + cosl1p2^2*
         (71955. + 7695.*cosl1p3^2) + cosl1p2*(141075.*cosl1p3 + 
          16605.*cosl1p3^3))*l1^4*p^4 + (-6318.*cosl1p2^3 - 135891.*cosl1p3 - 
        31563.*cosl1p2^2*cosl1p3 - 2322.*cosl1p3^3 + 
        cosl1p2*(-183429. - 20547.*cosl1p3^2))*l1^3*p^5 + 
      (100098. + 103734.*cosl1p2^2 + 132255.*cosl1p2*cosl1p3 + 
        19944.*cosl1p3^2)*l1^2*p^6 + (-151758.*cosl1p2 - 98622.*cosl1p3)*l1*
       p^7 + 48972.*p^8 + l1^7*(-7290.*cosl1p2*p - 14580.*cosl1p3*p)) + 
    cosl1p1^5*l1^3*p*(-4374.*l1^6 + (-21546. - 13608.*cosl1p2^2 - 
        65772.*cosl1p2*cosl1p3 - 35478.*cosl1p3^2)*l1^4*p^2 + 
      (-3888.*cosl1p2^3 + 62964.*cosl1p3 + 8100.*cosl1p2^2*cosl1p3 + 
        7290.*cosl1p3^3 + cosl1p2*(77004. + 16200.*cosl1p3^2))*l1^3*p^3 + 
      (-63864. - 31698.*cosl1p2^2 - 44982.*cosl1p2*cosl1p3 - 8046.*cosl1p3^2)*
       l1^2*p^4 + (82638.*cosl1p2 + 49122.*cosl1p3)*l1*p^5 - 50076.*p^6 + 
      l1^5*(20898.*cosl1p2*p + 33048.*cosl1p3*p)) + 
    cosl1p2^3*l1^3*(-486.*l1^6*p + 4644.*l1^4*p^3 - 972.*cosl1p3^4*l1^4*p^3 + 
      14526.*l1^2*p^5 + 12480.*p^7 + cosl1p3^3*(-972.*l1^5*p^2 + 
        1620.*l1^3*p^4) + cosl1p3^2*(8748.*l1^6*p + 27378.*l1^4*p^3 + 
        33507.*l1^2*p^5) + cosl1p3*(-2916.*l1^7 - 16524.*l1^5*p^2 - 
        50031.*l1^3*p^4 - 34020.*l1*p^6)) + 
    cosl1p2*l1*(-324.*l1^8*p + 1584.*l1^6*p^3 - 243.*cosl1p3^6*l1^6*p^3 + 
      13344.*l1^4*p^5 + 17428.*l1^2*p^7 + 6016.*p^9 + 
      cosl1p3^5*(3402.*l1^7*p^2 + 3726.*l1^5*p^4) + 
      cosl1p3^4*(-4374.*l1^8*p - 14742.*l1^6*p^3 - 7992.*l1^4*p^5) + 
      cosl1p3^3*(729.*l1^9 + 5103.*l1^7*p^2 + 4104.*l1^5*p^4 + 
        1728.*l1^3*p^6) + cosl1p3^2*(243.*l1^8*p + 4482.*l1^6*p^3 + 
        1530.*l1^4*p^5 - 2364.*l1^2*p^7) + 
      cosl1p3*(486.*l1^9 - 270.*l1^7*p^2 - 2088.*l1^5*p^4 - 6054.*l1^3*p^6 - 
        3198.*l1*p^8)) + cosl1p1^3*l1*(486.*l1^8*p - 13662.*l1^6*p^3 - 
      13122.*cosl1p2^5*l1^5*p^4 + 2916.*cosl1p3^5*l1^5*p^4 - 
      53370.*l1^4*p^5 - 53124.*l1^2*p^7 - 20556.*p^9 + 
      cosl1p2^4*l1^4*p^3*(73629.*l1^2 - 27783.*cosl1p3*l1*p + 25056.*p^2) + 
      cosl1p2^3*l1^3*p^2*(-44712.*l1^4 + 93312.*cosl1p3*l1^3*p + 
        (-48600. - 12312.*cosl1p3^2)*l1^2*p^2 + 34938.*cosl1p3*l1*p^3 + 
        36432.*p^4) + cosl1p3^4*(-29403.*l1^6*p^3 - 6372.*l1^4*p^5) + 
      cosl1p2^2*l1^2*p*(1458.*l1^6 - 3564.*cosl1p3*l1^5*p + 
        (-41661. - 19926.*cosl1p3^2)*l1^4*p^2 + 
        cosl1p3*(16551. + 10692.*cosl1p3^2)*l1^3*p^3 + 
        (-140949. - 378.*cosl1p3^2)*l1^2*p^4 + 85905.*cosl1p3*l1*p^5 - 
        152982.*p^6) + cosl1p3^3*(39852.*l1^7*p^2 + 32940.*l1^5*p^4 - 
        8676.*l1^3*p^6) + cosl1p3^2*(-13122.*l1^8*p - 12609.*l1^6*p^3 - 
        45873.*l1^4*p^5 - 46710.*l1^2*p^7) + 
      cosl1p2*l1*(729.*l1^8 - 17496.*cosl1p3*l1^7*p + 
        (18711. + 58644.*cosl1p3^2)*l1^6*p^2 + 
        cosl1p3*(-76788. - 75168.*cosl1p3^2)*l1^5*p^3 + 
        (111771. + 65529.*cosl1p3^2 + 11745.*cosl1p3^4)*l1^4*p^4 + 
        cosl1p3*(-237636. - 13878.*cosl1p3^2)*l1^3*p^5 + 
        (213900. + 31671.*cosl1p3^2)*l1^2*p^6 - 202428.*cosl1p3*l1*p^7 + 
        114324.*p^8) + cosl1p3*(2187.*l1^9 + 3645.*l1^7*p^2 + 
        86553.*l1^5*p^4 + 186492.*l1^3*p^6 + 81564.*l1*p^8)) + 
    cosl1p1*(2592.*l1^7*p^3 - 972.*cosl1p2^7*l1^6*p^4 + 16968.*l1^5*p^5 + 
      21140.*l1^3*p^7 + 6852.*l1*p^9 + cosl1p2^6*l1^5*p^3*
       (16038.*l1^2 - 3402.*cosl1p3*l1*p + 5184.*p^2) + 
      cosl1p2^5*l1^4*p^2*(-36450.*l1^4 + 47304.*cosl1p3*l1^3*p + 
        (-64044. - 3888.*cosl1p3^2)*l1^2*p^2 + 15552.*cosl1p3*l1*p^3 - 
        10206.*p^4) + cosl1p3^6*(-729.*l1^7*p^3 - 162.*l1^5*p^5) + 
      cosl1p2^4*l1^3*p*(20412.*l1^6 - 82944.*cosl1p3*l1^5*p + 
        (73899. + 40257.*cosl1p3^2)*l1^4*p^2 + 
        cosl1p3*(-149040. - 1215.*cosl1p3^2)*l1^3*p^3 + 
        (81819. + 12393.*cosl1p3^2)*l1^2*p^4 - 26541.*cosl1p3*l1*p^5 - 
        558.*p^6) + cosl1p3^5*(6804.*l1^8*p^2 + 9234.*l1^6*p^4 + 
        756.*l1^4*p^6) + cosl1p3^4*(-8748.*l1^9*p - 23598.*l1^7*p^3 - 
        11754.*l1^5*p^5 + 864.*l1^3*p^7) + cosl1p2^3*l1^2*
       (-2916.*l1^8 + 33048.*cosl1p3*l1^7*p + (-11988. - 42444.*cosl1p3^2)*
         l1^6*p^2 + cosl1p3*(122958. + 1296.*cosl1p3^2)*l1^5*p^3 + 
        (-42885. - 74952.*cosl1p3^2 + 972.*cosl1p3^4)*l1^4*p^4 + 
        cosl1p3*(149004. - 1620.*cosl1p3^2)*l1^3*p^5 + 
        (-22200. - 20979.*cosl1p3^2)*l1^2*p^6 - 3924.*cosl1p3*l1*p^7 + 
        17100.*p^8) + cosl1p3^3*(2187.*l1^10 + 6237.*l1^8*p^2 + 
        3312.*l1^6*p^4 + 2796.*l1^4*p^6 + 720.*l1^2*p^8) + 
      cosl1p3^2*(3807.*l1^9*p + 4104.*l1^7*p^3 - 23412.*l1^5*p^5 - 
        29764.*l1^3*p^7 - 6924.*l1*p^9) + 
      cosl1p3*(-972.*l1^10 - 1458.*l1^8*p^2 + 9936.*l1^6*p^4 + 
        16446.*l1^4*p^6 + 11950.*l1^2*p^8 + 3072.*p^10) + 
      cosl1p2*((486. + 1215.*cosl1p3^2)*l1^10 - 15552.*cosl1p3^3*l1^9*p + 
        (3564. - 1863.*cosl1p3^2 + 27864.*cosl1p3^4)*l1^8*p^2 + 
        cosl1p3*(-5184. - 44226.*cosl1p3^2 - 8748.*cosl1p3^4)*l1^7*p^3 + 
        (-648. - 10170.*cosl1p3^2 + 42525.*cosl1p3^4 + 243.*cosl1p3^6)*l1^6*
         p^4 + cosl1p3*(-47448. - 14544.*cosl1p3^2 - 1782.*cosl1p3^4)*l1^5*
         p^5 + (-27294. + 25116.*cosl1p3^2 + 1782.*cosl1p3^4)*l1^4*p^6 + 
        cosl1p3*(-55896. + 2412.*cosl1p3^2)*l1^3*p^7 + 
        (-15138. + 18018.*cosl1p3^2)*l1^2*p^8 - 20556.*cosl1p3*l1*p^9 + 
        3072.*p^10) + cosl1p2^2*l1*(-4293.*l1^8*p - 8586.*l1^6*p^3 + 
        972.*cosl1p3^5*l1^5*p^4 - 3318.*l1^4*p^5 - 896.*l1^2*p^7 - 
        13632.*p^9 + cosl1p3^4*(-16200.*l1^6*p^3 - 5265.*l1^4*p^5) + 
        cosl1p3^3*(19764.*l1^7*p^2 + 37503.*l1^5*p^4 - 3618.*l1^3*p^6) + 
        cosl1p3^2*(4374.*l1^8*p + 23571.*l1^6*p^3 + 60237.*l1^4*p^5 - 
          1818.*l1^2*p^7) + cosl1p3*(-2430.*l1^9 - 11826.*l1^7*p^2 - 
          47025.*l1^5*p^4 + 6024.*l1^3*p^6 + 34398.*l1*p^8))))*
   ((1. + 1.*k^6)*dtZA[(1 + k^6)^(1/6)]*RB[k^2, l1^2] + 
    (1 + k^6)*RBdot[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + 
    k^6*RB[k^2, l1^2]*(-50.*ZA[(1 + k^6)^(1/6)] + 
      50.*ZA[1.02*(1 + k^6)^(1/6)]))*
   ZA3[Sqrt[2/3]*Sqrt[l1^2 - cosl1p1*l1*p + p^2]]*
   ZA3[Sqrt[2/3]*Sqrt[l1^2 - (cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]]*
   ZA3[Sqrt[6*l1^2 - 6*(2*cosl1p1 + cosl1p2)*l1*p + 10*p^2]/3]*
   ZA3[Sqrt[6*l1^2 - 6*(2*cosl1p1 + 2*cosl1p2 + cosl1p3)*l1*p + 10*p^2]/3])/
  ((1. + k^6)*(l1^2 - 2.*cosl1p1*l1*p + p^2)*
   (l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*p + p^2)*
   (3.*l1^2 - 6.*cosl1p1*l1*p - 6.*cosl1p2*l1*p + 4.*p^2)*
   (RB[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + l1^2*ZA[l1])^2*
   (RB[k^2, l1^2 - 2*cosl1p1*l1*p + p^2]*ZA[(1 + k^6)^(1/6)] + 
    (l1^2 - 2.*cosl1p1*l1*p + p^2)*ZA[Sqrt[l1^2 - 2*cosl1p1*l1*p + p^2]])*
   (RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]*
     ZA[(1 + k^6)^(1/6)] + (l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*
       p + p^2)*ZA[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]])*
   (3.*RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2)*l1*p + (4*p^2)/3]*
     ZA[(1 + k^6)^(1/6)] + (3.*l1^2 - 6.*cosl1p1*l1*p - 6.*cosl1p2*l1*p + 
      4.*p^2)*ZA[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2)*l1*p + (4*p^2)/3]])), 
 (-0.04081632653061224*((1080. + 243.*cosl1p1^4 + 243.*cosl1p2^4 + 
      243.*cosl1p2^3*cosl1p3 - 1890.*cosl1p3^2 - 486.*cosl1p3^4 + 
      cosl1p1^3*(729.*cosl1p2 + 243.*cosl1p3) + 
      cosl1p2^2*(-567. - 243.*cosl1p3^2) + 
      cosl1p1^2*(-567. + 486.*cosl1p2^2 + 405.*cosl1p2*cosl1p3 - 
        243.*cosl1p3^2) + cosl1p2*(-1890.*cosl1p3 - 972.*cosl1p3^3) + 
      cosl1p1*(729.*cosl1p2^3 - 1890.*cosl1p3 + 405.*cosl1p2^2*cosl1p3 - 
        972.*cosl1p3^3 + cosl1p2*(-2916. - 810.*cosl1p3^2)))*l1^6 + 
    (-729.*cosl1p1^5 - 729.*cosl1p2^5 + cosl1p1^4*(-2916.*cosl1p2 - 
        1215.*cosl1p3) - 504.*cosl1p3 - 1215.*cosl1p2^4*cosl1p3 - 
      3348.*cosl1p3^3 + 972.*cosl1p3^5 + cosl1p2^3*(2592. + 243.*cosl1p3^2) + 
      cosl1p1^3*(2592. - 3645.*cosl1p2^2 - 3402.*cosl1p2*cosl1p3 + 
        243.*cosl1p3^2) + cosl1p2^2*(6210.*cosl1p3 + 3402.*cosl1p3^3) + 
      cosl1p2*(-3852. + 108.*cosl1p3^2 + 3402.*cosl1p3^4) + 
      cosl1p1^2*(-3645.*cosl1p2^3 + 6210.*cosl1p3 - 3402.*cosl1p2^2*cosl1p3 + 
        3402.*cosl1p3^3 + cosl1p2*(18576. + 2349.*cosl1p3^2)) + 
      cosl1p1*(-3852. - 2916.*cosl1p2^4 - 3402.*cosl1p2^3*cosl1p3 + 
        108.*cosl1p3^2 + 3402.*cosl1p3^4 + cosl1p2^2*
         (18576. + 2349.*cosl1p3^2) + cosl1p2*(22248.*cosl1p3 + 
          7452.*cosl1p3^3)))*l1^5*p + 
    (9144. + 486.*cosl1p1^6 + 486.*cosl1p2^6 + 1215.*cosl1p2^5*cosl1p3 - 
      11898.*cosl1p3^2 + 9666.*cosl1p3^4 - 486.*cosl1p3^6 + 
      cosl1p1^5*(2430.*cosl1p2 + 1215.*cosl1p3) + 
      cosl1p2^4*(-3159. + 486.*cosl1p3^2) + 
      cosl1p1^4*(-3159. + 4374.*cosl1p2^2 + 4698.*cosl1p2*cosl1p3 + 
        486.*cosl1p3^2) + cosl1p2^3*(-3807.*cosl1p3 - 2430.*cosl1p3^3) + 
      cosl1p2^2*(-4257. + 18171.*cosl1p3^2 - 4131.*cosl1p3^4) + 
      cosl1p2*(-18450.*cosl1p3 + 27108.*cosl1p3^3 - 2430.*cosl1p3^5) + 
      cosl1p1^3*(4860.*cosl1p2^3 - 3807.*cosl1p3 + 6561.*cosl1p2^2*cosl1p3 - 
        2430.*cosl1p3^3 + cosl1p2*(-31752. + 81.*cosl1p3^2)) + 
      cosl1p1^2*(-4257. + 4374.*cosl1p2^4 + 6561.*cosl1p2^3*cosl1p3 + 
        18171.*cosl1p3^2 - 4131.*cosl1p3^4 + cosl1p2^2*
         (-57510. - 1296.*cosl1p3^2) + cosl1p2*(-44307.*cosl1p3 - 
          8586.*cosl1p3^3)) + cosl1p1*(2430.*cosl1p2^5 - 18450.*cosl1p3 + 
        4698.*cosl1p2^4*cosl1p3 + 27108.*cosl1p3^3 - 2430.*cosl1p3^5 + 
        cosl1p2^3*(-31752. + 81.*cosl1p3^2) + cosl1p2^2*(-44307.*cosl1p3 - 
          8586.*cosl1p3^3) + cosl1p2*(-13968. + 24192.*cosl1p3^2 - 
          8586.*cosl1p3^4)))*l1^4*p^2 + 
    (648.*cosl1p1^5 + 648.*cosl1p2^5 + cosl1p1^4*(12366.*cosl1p2 - 
        2349.*cosl1p3) - 5760.*cosl1p3 - 2349.*cosl1p2^4*cosl1p3 - 
      4824.*cosl1p3^3 - 324.*cosl1p3^5 + cosl1p2^3*
       (23949. - 18549.*cosl1p3^2) + cosl1p1^3*(23949. + 34182.*cosl1p2^2 + 
        14148.*cosl1p2*cosl1p3 - 18549.*cosl1p3^2) + 
      cosl1p2^2*(57618.*cosl1p3 - 26406.*cosl1p3^3) + 
      cosl1p2*(-25200. + 25146.*cosl1p3^2 - 11178.*cosl1p3^4) + 
      cosl1p1^2*(34182.*cosl1p2^3 + 57618.*cosl1p3 + 33318.*cosl1p2^2*
         cosl1p3 - 26406.*cosl1p3^3 + cosl1p2*(88785. - 37557.*cosl1p3^2)) + 
      cosl1p1*(-25200. + 12366.*cosl1p2^4 + 14148.*cosl1p2^3*cosl1p3 + 
        25146.*cosl1p3^2 - 11178.*cosl1p3^4 + cosl1p2^2*
         (88785. - 37557.*cosl1p3^2) + cosl1p2*(128772.*cosl1p3 - 
          49140.*cosl1p3^3)))*l1^3*p^3 + 
    (13716. - 17172.*cosl1p1^4 - 17172.*cosl1p2^4 + 
      cosl1p1^3*(-78840.*cosl1p2 - 34677.*cosl1p3) - 
      34677.*cosl1p2^3*cosl1p3 - 10236.*cosl1p3^2 + 4734.*cosl1p3^4 + 
      cosl1p2^2*(1464. - 8595.*cosl1p3^2) + 
      cosl1p1^2*(1464. - 123336.*cosl1p2^2 - 121473.*cosl1p2*cosl1p3 - 
        8595.*cosl1p3^2) + cosl1p2*(-22596.*cosl1p3 + 13644.*cosl1p3^3) + 
      cosl1p1*(-78840.*cosl1p2^3 - 22596.*cosl1p3 - 121473.*cosl1p2^2*
         cosl1p3 + 13644.*cosl1p3^3 + cosl1p2*(-132. - 24480.*cosl1p3^2)))*
     l1^2*p^4 + (21738.*cosl1p1^3 + 21738.*cosl1p2^3 - 3112.*cosl1p3 + 
      36030.*cosl1p2^2*cosl1p3 - 588.*cosl1p3^3 + 
      cosl1p1^2*(67914.*cosl1p2 + 36030.*cosl1p3) + 
      cosl1p2*(-18052. + 13704.*cosl1p3^2) + 
      cosl1p1*(-18052. + 67914.*cosl1p2^2 + 74760.*cosl1p2*cosl1p3 + 
        13704.*cosl1p3^2))*l1*p^5 + (4672. - 6744.*cosl1p1^2 - 
      13488.*cosl1p1*cosl1p2 - 6744.*cosl1p2^2 - 7272.*cosl1p1*cosl1p3 - 
      7272.*cosl1p2*cosl1p3 - 528.*cosl1p3^2)*p^6)*
   ((1. + 1.*k^6)*dtZA[(1 + k^6)^(1/6)]*RB[k^2, l1^2] + 
    (1 + k^6)*RBdot[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + 
    k^6*RB[k^2, l1^2]*(-50.*ZA[(1 + k^6)^(1/6)] + 
      50.*ZA[1.02*(1 + k^6)^(1/6)]))*
   ZA3[Sqrt[2/3]*Sqrt[l1^2 - (cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]]*
   ZA3[Sqrt[6*l1^2 - 6*(2*cosl1p1 + 2*cosl1p2 + cosl1p3)*l1*p + 10*p^2]/3]*
   ZA4[Sqrt[3*l1^2 - 3*(cosl1p1 + cosl1p2)*l1*p + 5*p^2]/Sqrt[6]])/
  ((1. + k^6)*(l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*p + p^2)*
   (3.*l1^2 - 6.*cosl1p1*l1*p - 6.*cosl1p2*l1*p + 4.*p^2)*
   (RB[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + l1^2*ZA[l1])^2*
   (RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]*
     ZA[(1 + k^6)^(1/6)] + (l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*
       p + p^2)*ZA[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]])*
   (3.*RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2)*l1*p + (4*p^2)/3]*
     ZA[(1 + k^6)^(1/6)] + (3.*l1^2 - 6.*cosl1p1*l1*p - 6.*cosl1p2*l1*p + 
      4.*p^2)*ZA[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2)*l1*p + (4*p^2)/3]])), 
 (0.02040816326530612*((-120. - 27.*cosl1p1^4 + 54.*cosl1p2^4 + 
      cosl1p1^3*(-27.*cosl1p2 - 81.*cosl1p3) + 108.*cosl1p2^3*cosl1p3 + 
      63.*cosl1p3^2 - 27.*cosl1p3^4 + cosl1p1^2*(63. + 27.*cosl1p2^2 - 
        45.*cosl1p2*cosl1p3 - 54.*cosl1p3^2) + 
      cosl1p2^2*(210. + 27.*cosl1p3^2) + cosl1p2*(210.*cosl1p3 - 
        27.*cosl1p3^3) + cosl1p1*(108.*cosl1p2^3 + 324.*cosl1p3 + 
        90.*cosl1p2^2*cosl1p3 - 81.*cosl1p3^3 + 
        cosl1p2*(210. - 45.*cosl1p3^2)))*l1^6 + 
    (27.*cosl1p1^5 - 108.*cosl1p2^5 + 312.*cosl1p3 - 270.*cosl1p2^4*cosl1p3 - 
      18.*cosl1p3^3 + 27.*cosl1p3^5 + cosl1p1^4*(81.*cosl1p2 + 
        108.*cosl1p3) + cosl1p2^3*(-1776. - 162.*cosl1p3^2) + 
      cosl1p1^3*(-18. + 27.*cosl1p2^2 + 234.*cosl1p2*cosl1p3 + 
        135.*cosl1p3^2) + cosl1p2^2*(-2664.*cosl1p3 + 27.*cosl1p3^3) + 
      cosl1p2*(624. - 924.*cosl1p3^2 + 81.*cosl1p3^4) + 
      cosl1p1^2*(-162.*cosl1p2^3 - 126.*cosl1p3 - 27.*cosl1p2^2*cosl1p3 + 
        135.*cosl1p3^3 + cosl1p2*(-924. + 198.*cosl1p3^2)) + 
      cosl1p1*(312. - 270.*cosl1p2^4 - 396.*cosl1p2^3*cosl1p3 - 
        126.*cosl1p3^2 + 108.*cosl1p3^4 + cosl1p2^2*
         (-2664. - 27.*cosl1p3^2) + cosl1p2*(-1992.*cosl1p3 + 
          234.*cosl1p3^3)))*l1^5*p + (-1072. - 27.*cosl1p1^5*cosl1p2 + 
      54.*cosl1p2^6 + 6372.*cosl1p2^3*cosl1p3 + 162.*cosl1p2^5*cosl1p3 + 
      402.*cosl1p3^2 - 99.*cosl1p3^4 + cosl1p1^4*(-99. - 54.*cosl1p2^2 - 
        108.*cosl1p2*cosl1p3) + cosl1p2^4*(3186. + 135.*cosl1p3^2) + 
      cosl1p2^2*(1576. + 3579.*cosl1p3^2 - 54.*cosl1p3^4) + 
      cosl1p2*(1576.*cosl1p3 + 393.*cosl1p3^3 - 27.*cosl1p3^5) + 
      cosl1p1^3*(-630.*cosl1p3 - 153.*cosl1p2^2*cosl1p3 + 
        cosl1p2*(393. - 135.*cosl1p3^2)) + cosl1p1^2*(402. + 135.*cosl1p2^4 + 
        72.*cosl1p2^3*cosl1p3 - 1062.*cosl1p3^2 + 
        cosl1p2^2*(3579. - 144.*cosl1p3^2) + cosl1p2*(609.*cosl1p3 - 
          135.*cosl1p3^3)) + cosl1p1*(162.*cosl1p2^5 + 876.*cosl1p3 + 
        306.*cosl1p2^4*cosl1p3 - 630.*cosl1p3^3 + 
        cosl1p2^3*(6372. + 72.*cosl1p3^2) + cosl1p2^2*(6588.*cosl1p3 - 
          153.*cosl1p3^3) + cosl1p2*(1576. + 609.*cosl1p3^2 - 
          108.*cosl1p3^4)))*l1^4*p^2 + 
    (99.*cosl1p1^4*cosl1p2 - 1164.*cosl1p2^5 + 1404.*cosl1p3 - 
      2910.*cosl1p2^4*cosl1p3 - 294.*cosl1p3^3 + 
      cosl1p1^3*(-294. - 285.*cosl1p2^2 + 630.*cosl1p2*cosl1p3) + 
      cosl1p2^3*(-7448. - 2130.*cosl1p3^2) + 
      cosl1p2^2*(-11172.*cosl1p3 - 285.*cosl1p3^3) + 
      cosl1p2*(2808. - 4312.*cosl1p3^2 + 99.*cosl1p3^4) + 
      cosl1p1^2*(-2130.*cosl1p2^3 - 798.*cosl1p3 - 153.*cosl1p2^2*cosl1p3 + 
        cosl1p2*(-4312. + 1062.*cosl1p3^2)) + 
      cosl1p1*(1404. - 2910.*cosl1p2^4 - 3792.*cosl1p2^3*cosl1p3 - 
        798.*cosl1p3^2 + cosl1p2^2*(-11172. - 153.*cosl1p3^2) + 
        cosl1p2*(-8456.*cosl1p3 + 630.*cosl1p3^3)))*l1^3*p^3 + 
    (-1364. + 204.*cosl1p1^3*cosl1p2 + 5028.*cosl1p2^4 + 
      10056.*cosl1p2^3*cosl1p3 + 406.*cosl1p3^2 + 
      cosl1p1^2*(406. + 5232.*cosl1p2^2 + 468.*cosl1p2*cosl1p3) + 
      cosl1p2^2*(1816. + 5232.*cosl1p3^2) + 
      cosl1p2*(1816.*cosl1p3 + 204.*cosl1p3^3) + 
      cosl1p1*(10056.*cosl1p2^3 + 764.*cosl1p3 + 10320.*cosl1p2^2*cosl1p3 + 
        cosl1p2*(1816. + 468.*cosl1p3^2)))*l1^2*p^4 + 
    (-2316.*cosl1p1^2*cosl1p2 - 4632.*cosl1p2^3 + 772.*cosl1p3 - 
      6948.*cosl1p2^2*cosl1p3 + cosl1p1*(772. - 6948.*cosl1p2^2 - 
        4632.*cosl1p2*cosl1p3) + cosl1p2*(1544. - 2316.*cosl1p3^2))*l1*p^5 + 
    (-392. + 1176.*cosl1p1*cosl1p2 + 1176.*cosl1p2^2 + 1176.*cosl1p2*cosl1p3)*
     p^6)*((1. + 1.*k^6)*dtZA[(1 + k^6)^(1/6)]*RB[k^2, l1^2] + 
    (1 + k^6)*RBdot[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + 
    k^6*RB[k^2, l1^2]*(-50.*ZA[(1 + k^6)^(1/6)] + 
      50.*ZA[1.02*(1 + k^6)^(1/6)]))*
   ZA3[Sqrt[2/3]*Sqrt[l1^2 - cosl1p2*l1*p + p^2]]*
   ZA3[Sqrt[2/3]*Sqrt[l1^2 - (cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]]*
   ZA4[Sqrt[l1^2 - (cosl1p1 + 2*cosl1p2 + cosl1p3)*l1*p + 2*p^2]/Sqrt[2]])/
  ((1. + k^6)*(l1^2 - 2.*cosl1p2*l1*p + p^2)*
   (l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*p + p^2)*
   (RB[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + l1^2*ZA[l1])^2*
   (RB[k^2, l1^2 - 2*cosl1p2*l1*p + p^2]*ZA[(1 + k^6)^(1/6)] + 
    (l1^2 - 2.*cosl1p2*l1*p + p^2)*ZA[Sqrt[l1^2 - 2*cosl1p2*l1*p + p^2]])*
   (RB[k^2, l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + p^2]*
     ZA[(1 + k^6)^(1/6)] + (l1^2 + (-2.*cosl1p1 - 2.*cosl1p2 - 2.*cosl1p3)*l1*
       p + p^2)*ZA[Sqrt[l1^2 - 2*(cosl1p1 + cosl1p2 + cosl1p3)*l1*p + 
        p^2]])), 
 (3*(-3*(-1304 + 54*cosl1p1^4 - 27*cosl1p2^4 - 81*cosl1p2^3*cosl1p3 - 
      399*cosl1p3^2 - 27*cosl1p3^4 + 108*cosl1p1^3*(cosl1p2 + cosl1p3) + 
      3*cosl1p1^2*(-262 + 9*cosl1p2^2 + 30*cosl1p2*cosl1p3 + 9*cosl1p3^2) - 
      3*cosl1p2^2*(133 + 18*cosl1p3^2) - 3*cosl1p2*cosl1p3*
       (4 + 27*cosl1p3^2) - 3*cosl1p1*(262*cosl1p2 + 9*cosl1p2^3 + 
        262*cosl1p3 + 15*cosl1p2^2*cosl1p3 + 15*cosl1p2*cosl1p3^2 + 
        9*cosl1p3^3))*l1^2 - 9*(cosl1p2 + cosl1p3)*(872 + 282*cosl1p1^2 + 
      151*cosl1p2^2 + 20*cosl1p2*cosl1p3 + 151*cosl1p3^2 + 
      282*cosl1p1*(cosl1p2 + cosl1p3))*l1*p + 
    16*(382 + 57*cosl1p1^2 - 9*cosl1p2^2 - 75*cosl1p2*cosl1p3 - 9*cosl1p3^2 + 
      57*cosl1p1*(cosl1p2 + cosl1p3))*p^2)*
   (RBdot[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + RB[k^2, l1^2]*
     (dtZA[(1 + k^6)^(1/6)] + (50.*k^6*(-ZA[(1 + k^6)^(1/6)] + 
         ZA[1.02*(1 + k^6)^(1/6)]))/(1 + k^6)))*
   ZA4[Sqrt[3*l1^2 - 3*(cosl1p2 + cosl1p3)*l1*p + 5*p^2]/Sqrt[6]]^2)/
  (196*(3*l1^2 - 6*(cosl1p2 + cosl1p3)*l1*p + 4*p^2)*
   (RB[k^2, l1^2]*ZA[(1 + k^6)^(1/6)] + l1^2*ZA[l1])^2*
   (3*RB[k^2, l1^2 - 2*(cosl1p2 + cosl1p3)*l1*p + (4*p^2)/3]*
     ZA[(1 + k^6)^(1/6)] + (3*l1^2 - 6*(cosl1p2 + cosl1p3)*l1*p + 4*p^2)*
     ZA[Sqrt[l1^2 - 2*(cosl1p2 + cosl1p3)*l1*p + (4*p^2)/3]]))}
