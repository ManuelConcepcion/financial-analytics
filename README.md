# Financial Analytics Project
Isabella, Etienne, Irene, Manuel  
PLE+BDBA  

## Contents of this Repo
**data_download.py**: A refactored script for downloading Binance 5-minute data for the cryptocurrencies, providing methods to download extra data when necessary.  
**data_resampling.py**: A script containing a class and methods to resample data in some frequency to data in some other frequency, measured in minutes.  
**create_model.py**: A script containing a class and methods to take resampled data, fit a model, and evaluate that model given some financial parameters.  
**finance_pipeline.py**: A pipeline for evaluating the Cartesian product of lists of financial parameters, to find the most profitable.  
**finance_pipeline_utils.py**: A supporting script containing functions to calculate some financial features.  
  
**finance_pipeline_results.csv**: A csv file of the results of each iteration in finance_pipeline.py.  
**EDA.- GRAPHS.ipynb**: A visual analysis of some of the financial metrics utilized throughout the project.  
**Model Training Pipeline Final.ipynb**: A demonstration of the full pipeline for the best-performing financial parameters.  
  
## Canva Presentation Link
[Presentation Link](https://www.canva.com/design/DAF2ONNNOVo/JVmN7AcgFe_6efnBpul-Tg/edit?utm_content=DAF2ONNNOVo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
