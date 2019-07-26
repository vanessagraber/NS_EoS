# Neutron Star Parameters as a Function of Density

This repository provides code that allows the calculation of relevant neutron star parameters as a function of density 
for a given set of Skyrme parameters t<sub>0</sub>, t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>, t<sub>4</sub>
x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, x<sub>4</sub>, &sigma;, and &sigma;<sub>2</sub>.

## Background Physics

To determine the composition of the nuclear matter inside the neutron star core, we follow 
[Chamel (2008)](https://academic.oup.com/mnras/article/388/2/737/977911)
and solve four equations related to baryon conservation, charge neutrality, beta equilibrium and muon production rate 
for a given set of Skyrme parameters. Implemented in `ns_eos/equilibrium_comp.py`, 
this provides number densities, particle fractions, Fermi wave numbers, 
Landau effective masses and dynamic effective masses related to entrainment 
(see [Chammel and Haensel (2006)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.73.045802) 
for details) for a given baryon density.
To determine the properties of the superfluids (specifically their coherence lengths), 
we employ the gap parametrisation introduced in 
[Andersson et al. (2005)](https://www.sciencedirect.com/science/article/abs/pii/S0375947405010572?via%3Dihub) 
with parameters for the singlet proton and triplet neutron gap given in [Ho et al. (2015)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.91.015806).
Details of the gap calculation can be found in `ns_eos/gap_parametrisation.py`.


## Getting Started

These instructions will provide you with a copy of the project and help you get it up and running on your local machine.
The repo contains an environment file that can be installed by running
```
$ conda env create -f environment.yaml
```
We recommend working within this environment when using the code. To install the package run
```
$ python setup.py install
```
To see if the installation has been successful, we provide an example file. Running 
```
$ python examples/generating_eos_data.py
```
will create .txt files for six different sample equations of states (NRAPR, SLy4, LNS, SQMC700, Ska35s20, Sk&chi;450)
in `examples/data/`, providing a range of parameters as a function of baryon density.

Note that by default, an instance of the `EquationOfState` class will be created based on the Skyrme parameters 
of the NRAPR equation of state. If you would like to choose a different equation of state,
specify the corresponding Skyrme parameters t<sub>0</sub>, t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>, t<sub>4</sub>
x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, x<sub>4</sub>, &sigma;, and &sigma;<sub>2</sub>
 as shown in `examples/generating_eos_data.py`.

 
## Tests

All methods and functions in this repository have been tested. 
Tests can be found [here](https://github.com/vanessagraber/NS_EoS/tree/master/ns_eos/tests).
Running
```
$ python setup.py test
```
will run the tests and output a coverage report.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


