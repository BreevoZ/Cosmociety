# Convection Scan

| run | ok | p | Dconv max | nabla_ad | conv frac | regions | convective regions | core? | env? | Tc/Ts | step | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 8.000 | 5.00e-04 | 0.400 | 0.390 | 1 | 0.616-1.000 | no | yes | 11.759 | 79255 | 2.00e-07 |
| 2 | yes | 8.000 | 1.00e-03 | 0.400 | 0.400 | 1 | 0.606-1.000 | no | yes | 10.291 | 88283 | 2.00e-07 |
| 3 | yes | 10.000 | 5.00e-04 | 0.400 | 0.470 | 1 | 0.535-1.000 | no | yes | 11.799 | 99518 | 2.00e-07 |
| 4 | yes | 10.000 | 1.00e-03 | 0.400 | 0.480 | 1 | 0.525-1.000 | no | yes | 10.333 | 112009 | 2.00e-07 |

## Notes

- `convective regions` lists separate contiguous radial convection zones.
- `core?` means convection touches the innermost grid cells.
- `env?` means convection reaches the outer envelope/surface.
- `Tc/Ts` is center-to-surface temperature contrast.
