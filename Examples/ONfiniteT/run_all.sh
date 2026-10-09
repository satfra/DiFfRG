#!/bin/bash

cd build
./CG -ss /output/name=CG_out
./DG -ss /output/name=DG_out
./LDG -ss /output/name=LDG_out
./KT -ss /output/name=KT_out
./KT_sigma -ss /output/name=KT_sigma_out
