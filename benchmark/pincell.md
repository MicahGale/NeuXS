### Reflective pincell problem description

This pin-cell model is used for cross-section lookup and neutron transport
in a thermal reactor. Geometry and dimensions are representative of a
typical UO2 fuel rod with Zircaloy cladding and light water moderator.

### Properties of the model

| Region | Nuclide | A |    Density[atoms/(barn·cm)] | Temperature | Fissionable |
|---|---:|---:|----------------------------:|---:|---:|
| Fuel | U235 | 235 |                     1.15e-6 | 250.0 | true |
| Fuel | U238 | 238 |                     5.45e-5 | 250.0 | true |
| Fuel | O16 | 16 |                     9.79e-4 | 250.0 | false |
| Gas | C12 | 12 |                     5.02e-8 | 250.0 | false |
| Cladding | Zr90 | 90 |                     2.84e-4 | 250.0 | false |
| Cladding | Sn120 | 120 |                     2.26e-6 | 250.0 | false |
| Cladding | Fe56 | 56 |                     2.15e-6 | 250.0 | false |
| Cladding | Cr52 | 52 |                     1.16e-6 | 250.0 | false |
| Moderator | H1 | 1 |                     2.99e-2 | 250.0 | false |
| Moderator | O16 | 16 |                     9.34e-4 | 250.0 | false |