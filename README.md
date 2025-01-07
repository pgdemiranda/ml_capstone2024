# FIPE Table - Historical Vehicle Price Data
## Overview
The FIPE Table's primary function is to calculate the average prices of vehicles (cars, motorcycles, and trucks) for each model and version across Brazil. 
It serves as the main reference for negotiations or valuations, offering consumers guidance on what constitutes a fair buying or selling price for a given vehicle.

The values are estimated based on a survey of vehicle prices and are updated monthly. 
The prices reflect the average values being negotiated in the market, with adjustments depending on variables such as the vehicle's condition, manufacturer, year, model, region, optional features, color, and other relevant factors.

In addition to being a benchmark for retailers and dealers, the FIPE Table is used by insurance companies to calculate indemnities in cases of theft or total loss, as well as for insurance policy pricing and tax calculations, such as the IPVA (Vehicle Property Tax).


1. The FIPE Table expresses average prices for cash transactions in resale markets targeting individual consumers (natural persons) across Brazil. It is intended as a negotiation or valuation parameter. Actual transaction prices may vary depending on the region, vehicle condition, color, accessories, or other factors influencing supply and demand for a specific vehicle.
2. The vehicle year refers to the model year, excluding vehicles designed for professional or special purposes.
3. Prices are expressed in BRL (Brazilian reais) for the referenced month and year.

### What is FIPE?
The Fundação Instituto de Pesquisas Econômicas (FIPE) is a private, non-profit organization established in 1973. It supports public and private educational and research institutions, particularly the Department of Economics at FEA-USP (São Paulo University). 
FIPE is recognized for its significant contributions to education, research, project development, and the creation of economic and financial indicators.

### <span style="color:red">**Disclaimer for the evaluation of this project**</span>
We tried to follow the steps from each of the lessons, but we made some modifications because I was looking for an opportunity to experiment with different technologies and steps, and the project turned out to be the ideal opportunity for that. 
Below are the most important modifications:

1. Package Management
This project uses Poetry as the package manager instead of Pipenv. Consequently, the project includes a `pyproject.toml` file and a `poetry.lock` file, replacing the traditional `Pipfile` and `Pipfile.lock` files used with Pipenv.

2. Feature Transformation
Similar to the approach taken in the Midterm project, instead of using the `DictVectorizer` class to transform feature-value mappings into vectors, the `Pipeline` class was used. This approach allows for the application of various encoding methods while maintaining a structured and organized transformation workflow for the dataset.

3. Hyperparameter Tuning and Cross-Validation
All parameter evaluation and cross-validation were conducted using the `GridSearchCV` function from Scikit-Learn. While the instructions remained consistent with standard practices, the implementation leveraged this single, versatile method to streamline the process.

## Dataset
The dataset provides a historical record of vehicle prices extracted from the official FIPE (Fundação Instituto de Pesquisas Econômicas) Table through its public consultation platform: [FIPE Vehicle Prices](). The dataset for this project was downloaded from [Tabela Fipe - Histórico de Preços Kaggle Competition](https://www.kaggle.com/datasets/franckepeixoto/tabela-fipe?select=tabela-fipe-historico-precos.csv) and contains 

### Numerical Features
|   **Feature**   |                    **Description**                 |
|-----------------|----------------------------------------------------|
| `anoModelo`     |          The manufacturing year of the vehicle     |
| `mesReferencia` | The month in which the average price was published |
| `anoReferencia` | The year in which the average price was published  |

### Categorical Features
| **Feature**     | **Description**                                           |  
|-----------------|-----------------------------------------------------------|  
| `codigoFipe`    | Internal code maintained by FIPE to identify the vehicle. |  
| `marca`         | The name of the vehicle's brand.                          |  
| `modelo`        | The specific model of the vehicle.                        |

### Target Feature
| **Feature** | **Description**                     |  
|-------------|-------------------------------------|  
| `valor`     | The average price of the vehicle.   |  

## Notebooks  
As this is an iterative process, different notebooks were produced to train various models and experiment with different approaches.  

### Notebook1  
A notebook detailing the process of data preparation, exploration, training, and evaluation can be found in [notebook1](./final_project/notebook1.ipynb). It includes:  

1. **Data Reading and Cleaning**: The dataset was loaded from a local `.csv` file, and headers and strings were transformed into snake case.  

2. **Feature Engineering**: A new feature was created for this project: `classificacao_marca`, which categorizes the mean values of `valor` by the `marca` feature. After this, the data was split into train, test, and validation sets.  

3. **Exploratory Data Analysis**:  
   Central tendency and data dispersion measures were analyzed. Categorical data was explored by examining the ratio of the target variable (`valor`). For numerical features, correlation with the target variable was evaluated, and a heatmap of these relationships was plotted.  

4. **Feature Selection**:  
   To enhance the understanding of the relationships in the data, a permutation importance study was performed using a LightGBM model to analyze the mean and standard deviation of feature importance. Additionally, SHAP values were used to assess the impact of features on the target variable (`valor`). Based on these analyses, the features `marca`, `anoModelo`, and `anoReferencia` were selected.  

5. **Training Models**:  
   Different linear models were trained: linear regression, as well as ridge and lasso regularizations.  

6. **Model Tuning**:  
   The final ridge model was tuned and cross-validated with `GridSearchCV`.  

7. **Model Serving**:  
   The final model was serialized using the `pickle` library and stored in the [models](./final_project/models/) directory.

## Scripts

## Dockerfile

## Deployment

## Conclusion
