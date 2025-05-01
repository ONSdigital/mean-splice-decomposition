# GEKS decomposition code

This repo contains the code that accompanies the paper ["Practical decomposition of mean-spliced on published indices"](https://unece.org/sites/default/files/2025-04/geneva%20paper.pdf) by T Liu. It allows a multiplicative decomposition of the GEKS-Törnqvist index, breaking each elementary aggregate down based on product ID. When provided with groupings column it can calculate decompositions across multiple elementary aggregates at once. 

## GEKS_decomp_functions.py

The core code. GEKS_T_contrib, contrib_table, recursive_contrib, and init_window_decomp calculate the multiplicative decomposition of the GEKS. read_dominicks_data, write_master_contributions, and read_master_contributions are for reading in Dominick's data as well as reading and writing the contribution files.

## GEKS_initial.py

This script calculates the decomposition of the initial window, with the window length determining the number of periods to include in the index window and threshold setting the minimum acceptable number of bilateral pairs required to calculate the index. 1 to window length - 1 will result in a 'partial GEKS' if some bilateral pairs are incalculable, while setting it to window length will require a full GEKS be calculated.

## GEKS_arbitrary_periods.py

Having run GEKS_initial, GEKS_arbitrary_periods extends the index decomposition values out by window length periods, performing a splice on the published index. Note that the window length and threshold also need to be set here as well; it would be unconventional to set them to values different from the initial window.

## data

No data are provided, though read_dominicks_data is provided to help process Dominick's scanner data. The schema for the data to read into the decomposition functions is:s

  consumption_segment_code: str, grouping column
  
  region_code: str, grouping column
  
  retailer: str, grouping column

  product_id: str, alphanumetric code to identify specific products

  period: datetime64[ns, UTC], used to identify distinct time periods

  price: float, price per unit of specific product

  quantity: float, quantity sold of speicfic product

## Licence
Unless stated otherwise, the codebase is released under the MIT License. This covers both the codebase and any sample code in the documentation. The documentation is © Crown copyright and available under the terms of the Open Government 3.0 licence.

## Acknowledgements
[This project structure is based on the `govcookiecutter` template](https://github.com/best-practice-and-impact/govcookiecutter)
project. Guidance on using the govcookiecutter can be found on [this youtube video](https://www.youtube.com/watch?v=N7_d3k3uQ_M) and in the [documentation here](https://dataingovernment.blog.gov.uk/2021/07/20/govcookiecutter-a-template-for-data-science-projects/).


