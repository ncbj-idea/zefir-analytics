# Zefir Analytics

The Zefir Analytics module was created by the authors of PyZefir for relatively inexpensive processing or 
conversion of raw data into user-friendly condition, such as a report or a set of graphs. Also, this module is a set 
of computational methods used in the endpoints of the Zefir Backend repository.

## Setup Development Environment

Install repository from global pip index:
```bash
pip install zefir-analytics
```

### Make setup

Check if make is already installed
```bash
make --version
```
If not, install make
```bash
sudo apt install make 
```

## Make stages

Install virtual environment and all dependencies
```bash
make install
```
Run linters check (black, pylama)
```bash
make lint
```
Run unit and fast integration tests (runs lint stage before)
```bash
Make unit
```
Run integration tests (runs lint and unit stages before)
```bash
make test
```
Remove temporary directories such as .venv, .mypy_cache, .pytest_cache etc.
```bash
make clean
```
## Available methods in Zefir Engine objects
* source_params:
  * get_generation_sum
  * get_dump_energy_sum
  * get_load_sum
  * get_installed_capacity
  * get_generation_demand
  * get_fuel_usage
  * get_capex_opex
  * get_emission
* aggregated_consumer_params:
  * get_fractions
  * get_n_consumers
  * get_yearly_energy_usage
  * get_total_yearly_energy_usage
  * get_fractions
* lbs_params:
  * get_lbs_fraction
  * get_lbs_capacity
* line_params:
  * get_flow
  * get_transmission_fee
