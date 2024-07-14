# hik-thermo-dashboard
A webbased dashboard application to visualize and analyze thermal information from a 96-well plates data format.

## Project Motivation
This project aims to demonstrate a proof of concept data acquisition ecosystem for a high-throughput and low-cost thermal gradient system based on 96-well plate architecture. This system can be applied for a wide range of applications in studying the thermochemical (e.g., thermal decomposition) and thermophysical properties of materials. The thermal gradient heating and cooling system is developed by [Dr Richard Thompson](https://www.durham.ac.uk/staff/r-l-thompson/) research unit at the Durham University, United Kingdom. The proposed data acquisition ecosystem consist of two major parts: data extraction program and a webbased dashboard system.

This github repositatry is the second part of the purposed ecosystem that is used to visualized the extracted data from the thermal infrared camera deployed in this proof of concept system. This [thermal dashboard web application](https://hik-thermo-dashboard.onrender.com) is depolyed and lived on Render on Free instance. It is accesssible from all platforms with internet connection.

The example data format that is supported by this visualization system is provided in [sample-data folder](/sample-data/).

## Development Tools
- Programming Language: Python
- Framework: [Dash](https://dash.plotly.com/), Bootstrap
- Libraries: Pandas, NumPy, SciPy, Plotly, Dash Bootstrap Components
- Web Deployment: [Render](https://render.com/)
