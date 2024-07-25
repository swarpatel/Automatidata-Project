# **Automatidata project**

Welcome to the Automatidata Project!

You have just started as a data professional in a fictional data consulting firm, Automatidata. Their client, the New York City Taxi and Limousine Commission (New York City TLC), has hired the Automatidata team for its reputation in helping their clients develop data-based solutions.

The team is still in the early stages of the project. Previously, you were asked to complete a project proposal by your supervisor, DeShawn Washington. You have received notice that your project proposal has been approved and that New York City TLC has given the Automatidata team access to their data. To get clear insights, New York TLC's data must be analyzed, key variables identified, and the dataset ensured it is ready for analysis.

# Inspect and analyze data

**The purpose** of this project is to investigate and understand the data provided.
  
**The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings. 
<br/>  
*This activity has three parts:*

**Part 1:** Understand the situation 
* Prepare to understand and organize the provided taxi cab dataset and information.

**Part 2:** Understand the data

* Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities.

* Compile summary information about the data to inform next steps.

**Part 3:** Understand the variables

* Use insights from your examination of the summary data to guide deeper investigation into specific variables.



# **Identify data types and relevant variables using Python**


<img src="images/Pace.png" width="100" height="100" align=left>

# **PACE stages**


<img src="images/Plan.png" width="100" height="100" align=left>


## PACE: **Plan**

### **Task 1. Understand the situation**

#### How can you best prepare to understand and organize the provided taxi cab information? 

●	Who is your audience for this project? 

    New York City Taxi and Limousine Commission.

●	What are you trying to solve or accomplish? And, what do you anticipate the impact of this work will be on the larger needs of the client? 

    The estimation of taxi fares based on relevant variables.

●	What questions need to be asked or answered?

    What is the condition of the provided dataset? 
    What variables will be the most useful? 
    Are there trends within the data that can provide insight? 
    What steps can I take to reduce the impact of bias?

●	What resources are required to complete this project? 

    Project dataset, Python notebook, and input from stakeholders.

●	What are the deliverables that will need to be created over the course of this project? 

    Dataset scrubbed for exploratory data analysis, visualizations, statistical model, regression analysis and/or machine learning model.

<img src="images/Analyze.png" width="100" height="100" align=left>

## PACE: **Analyze**

### **Task 2a. Build dataframe**
















Create a pandas dataframe for data learning, and future exploratory data analysis (EDA) and statistical activities.


```python
#Import libraries and packages listed above
import numpy as np
import pandas as pd

# Load dataset into dataframe
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
print("done")
```

    done
    

### **Task 2b. Understand the data - Inspect the data**

Consider the following two questions:

**Question 1:** When reviewing the `df.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?

**Question 2:** When reviewing the `df.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values?


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24870114</td>
      <td>2</td>
      <td>03/25/2017 8:55:43 AM</td>
      <td>03/25/2017 9:09:47 AM</td>
      <td>6</td>
      <td>3.34</td>
      <td>1</td>
      <td>N</td>
      <td>100</td>
      <td>231</td>
      <td>1</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.76</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>16.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35634249</td>
      <td>1</td>
      <td>04/11/2017 2:53:28 PM</td>
      <td>04/11/2017 3:19:58 PM</td>
      <td>1</td>
      <td>1.80</td>
      <td>1</td>
      <td>N</td>
      <td>186</td>
      <td>43</td>
      <td>1</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>4.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>20.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>106203690</td>
      <td>1</td>
      <td>12/15/2017 7:26:56 AM</td>
      <td>12/15/2017 7:34:08 AM</td>
      <td>1</td>
      <td>1.00</td>
      <td>1</td>
      <td>N</td>
      <td>262</td>
      <td>236</td>
      <td>1</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.45</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38942136</td>
      <td>2</td>
      <td>05/07/2017 1:17:59 PM</td>
      <td>05/07/2017 1:48:14 PM</td>
      <td>1</td>
      <td>3.70</td>
      <td>1</td>
      <td>N</td>
      <td>188</td>
      <td>97</td>
      <td>1</td>
      <td>20.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.39</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>27.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30841670</td>
      <td>2</td>
      <td>04/15/2017 11:32:20 PM</td>
      <td>04/15/2017 11:49:03 PM</td>
      <td>1</td>
      <td>4.37</td>
      <td>1</td>
      <td>N</td>
      <td>4</td>
      <td>112</td>
      <td>2</td>
      <td>16.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>17.80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23345809</td>
      <td>2</td>
      <td>03/25/2017 8:34:11 PM</td>
      <td>03/25/2017 8:42:11 PM</td>
      <td>6</td>
      <td>2.30</td>
      <td>1</td>
      <td>N</td>
      <td>161</td>
      <td>236</td>
      <td>1</td>
      <td>9.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.06</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>12.36</td>
    </tr>
    <tr>
      <th>6</th>
      <td>37660487</td>
      <td>2</td>
      <td>05/03/2017 7:04:09 PM</td>
      <td>05/03/2017 8:03:47 PM</td>
      <td>1</td>
      <td>12.83</td>
      <td>1</td>
      <td>N</td>
      <td>79</td>
      <td>241</td>
      <td>1</td>
      <td>47.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>9.86</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>59.16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>69059411</td>
      <td>2</td>
      <td>08/15/2017 5:41:06 PM</td>
      <td>08/15/2017 6:03:05 PM</td>
      <td>1</td>
      <td>2.98</td>
      <td>1</td>
      <td>N</td>
      <td>237</td>
      <td>114</td>
      <td>1</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.78</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>19.58</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8433159</td>
      <td>2</td>
      <td>02/04/2017 4:17:07 PM</td>
      <td>02/04/2017 4:29:14 PM</td>
      <td>1</td>
      <td>1.20</td>
      <td>1</td>
      <td>N</td>
      <td>234</td>
      <td>249</td>
      <td>2</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>9.80</td>
    </tr>
    <tr>
      <th>9</th>
      <td>95294817</td>
      <td>1</td>
      <td>11/10/2017 3:20:29 PM</td>
      <td>11/10/2017 3:40:55 PM</td>
      <td>1</td>
      <td>1.60</td>
      <td>1</td>
      <td>N</td>
      <td>239</td>
      <td>237</td>
      <td>1</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.75</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>16.55</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22699 entries, 0 to 22698
    Data columns (total 18 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   Unnamed: 0             22699 non-null  int64  
     1   VendorID               22699 non-null  int64  
     2   tpep_pickup_datetime   22699 non-null  object 
     3   tpep_dropoff_datetime  22699 non-null  object 
     4   passenger_count        22699 non-null  int64  
     5   trip_distance          22699 non-null  float64
     6   RatecodeID             22699 non-null  int64  
     7   store_and_fwd_flag     22699 non-null  object 
     8   PULocationID           22699 non-null  int64  
     9   DOLocationID           22699 non-null  int64  
     10  payment_type           22699 non-null  int64  
     11  fare_amount            22699 non-null  float64
     12  extra                  22699 non-null  float64
     13  mta_tax                22699 non-null  float64
     14  tip_amount             22699 non-null  float64
     15  tolls_amount           22699 non-null  float64
     16  improvement_surcharge  22699 non-null  float64
     17  total_amount           22699 non-null  float64
    dtypes: float64(8), int64(7), object(3)
    memory usage: 3.1+ MB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>VendorID</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.269900e+04</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
      <td>22699.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.675849e+07</td>
      <td>1.556236</td>
      <td>1.642319</td>
      <td>2.913313</td>
      <td>1.043394</td>
      <td>162.412353</td>
      <td>161.527997</td>
      <td>1.336887</td>
      <td>13.026629</td>
      <td>0.333275</td>
      <td>0.497445</td>
      <td>1.835781</td>
      <td>0.312542</td>
      <td>0.299551</td>
      <td>16.310502</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.274493e+07</td>
      <td>0.496838</td>
      <td>1.285231</td>
      <td>3.653171</td>
      <td>0.708391</td>
      <td>66.633373</td>
      <td>70.139691</td>
      <td>0.496211</td>
      <td>13.243791</td>
      <td>0.463097</td>
      <td>0.039465</td>
      <td>2.800626</td>
      <td>1.399212</td>
      <td>0.015673</td>
      <td>16.097295</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.212700e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-120.000000</td>
      <td>-1.000000</td>
      <td>-0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.300000</td>
      <td>-120.300000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.852056e+07</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>1.000000</td>
      <td>114.000000</td>
      <td>112.000000</td>
      <td>1.000000</td>
      <td>6.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>8.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.673150e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.610000</td>
      <td>1.000000</td>
      <td>162.000000</td>
      <td>162.000000</td>
      <td>1.000000</td>
      <td>9.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.350000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>11.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.537452e+07</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.060000</td>
      <td>1.000000</td>
      <td>233.000000</td>
      <td>233.000000</td>
      <td>2.000000</td>
      <td>14.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>2.450000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>17.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.134863e+08</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>33.960000</td>
      <td>99.000000</td>
      <td>265.000000</td>
      <td>265.000000</td>
      <td>4.000000</td>
      <td>999.990000</td>
      <td>4.500000</td>
      <td>0.500000</td>
      <td>200.000000</td>
      <td>19.100000</td>
      <td>0.300000</td>
      <td>1200.290000</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Reviewing df.info() Output

Variable Types:

* The dataset has a total of 18 columns.
* There are 8 columns of type float64, 7 columns of type int64, and 3 columns of type object.

Null Values:

* None of the columns have null values. Each column has 22,699 non-null entries.

Column Types:

* tpep_pickup_datetime and tpep_dropoff_datetime are of type object, which means they are stored as strings. They will need to be converted to datetime objects for any time series analysis.
* store_and_fwd_flag is also of type object, which likely contains categorical data.

Data Integrity:

* The Unnamed: 0 column seems to be an unnecessary index column that can be dropped.
* All other columns are numeric, except for the datetime and categorical columns mentioned.

### 2. Reviewing df.describe() Output

Potential Issues:

* passenger_count: The minimum value is 0, which is questionable since there should be at least one passenger in a taxi trip.
* trip_distance: The minimum value is 0, indicating trips with no distance covered. This could be due to data entry errors or trips that were canceled or not properly recorded.
* There are negative values present in some columns (e.g., fare_amount and total_amount have a minimum of -120.0 and -120.3, respectively). This is highly unusual and suggests data entry errors.
* payment_type: The maximum value is 4, indicating 4 different types of payment methods.
* total_amount: The maximum value is 1200.29, which seems extremely high for a taxi ride, suggesting potential outliers or data errors.

### Summary:

* There are several data quality issues in the dataset, such as negative values in fare-related columns and zero values in columns where they are not logically expected.
* The datetime columns need conversion from object type to datetime type.
* Further investigation and cleaning are necessary to address the anomalies and ensure the accuracy of the analysis.

</br>

### **Task 2c. Understand the data - Investigate the variables**

Sort and interpret the data table for two variables:`trip_distance` and `total_amount`.

**Answer the following three questions:**

**Question 1:** Sort your first variable (`trip_distance`) from maximum to minimum value, do the values seem normal?

**Question 2:** Sort by your second variable (`total_amount`), are any values unusual?

**Question 3:** Are the resulting rows similar for both sorts? Why or why not?


```python
# Sort the data by trip distance from maximum to minimum value

df_sorted_trip_distance = df.sort_values(by=['trip_distance'],ascending=False)
df_sorted_trip_distance.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9280</th>
      <td>51810714</td>
      <td>2</td>
      <td>06/18/2017 11:33:25 PM</td>
      <td>06/19/2017 12:12:38 AM</td>
      <td>2</td>
      <td>33.96</td>
      <td>5</td>
      <td>N</td>
      <td>132</td>
      <td>265</td>
      <td>2</td>
      <td>150.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>150.30</td>
    </tr>
    <tr>
      <th>13861</th>
      <td>40523668</td>
      <td>2</td>
      <td>05/19/2017 8:20:21 AM</td>
      <td>05/19/2017 9:20:30 AM</td>
      <td>1</td>
      <td>33.92</td>
      <td>5</td>
      <td>N</td>
      <td>229</td>
      <td>265</td>
      <td>1</td>
      <td>200.01</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>51.64</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>258.21</td>
    </tr>
    <tr>
      <th>6064</th>
      <td>49894023</td>
      <td>2</td>
      <td>06/13/2017 12:30:22 PM</td>
      <td>06/13/2017 1:37:51 PM</td>
      <td>1</td>
      <td>32.72</td>
      <td>3</td>
      <td>N</td>
      <td>138</td>
      <td>1</td>
      <td>1</td>
      <td>107.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.50</td>
      <td>16.26</td>
      <td>0.3</td>
      <td>179.06</td>
    </tr>
    <tr>
      <th>10291</th>
      <td>76319330</td>
      <td>2</td>
      <td>09/11/2017 11:41:04 AM</td>
      <td>09/11/2017 12:18:58 PM</td>
      <td>1</td>
      <td>31.95</td>
      <td>4</td>
      <td>N</td>
      <td>138</td>
      <td>265</td>
      <td>2</td>
      <td>131.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>131.80</td>
    </tr>
    <tr>
      <th>29</th>
      <td>94052446</td>
      <td>2</td>
      <td>11/06/2017 8:30:50 PM</td>
      <td>11/07/2017 12:00:00 AM</td>
      <td>1</td>
      <td>30.83</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>23</td>
      <td>1</td>
      <td>80.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>18.56</td>
      <td>11.52</td>
      <td>0.3</td>
      <td>111.38</td>
    </tr>
    <tr>
      <th>18130</th>
      <td>90375786</td>
      <td>1</td>
      <td>10/26/2017 2:45:01 PM</td>
      <td>10/26/2017 4:12:49 PM</td>
      <td>1</td>
      <td>30.50</td>
      <td>1</td>
      <td>N</td>
      <td>132</td>
      <td>220</td>
      <td>1</td>
      <td>90.50</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>19.85</td>
      <td>8.16</td>
      <td>0.3</td>
      <td>119.31</td>
    </tr>
    <tr>
      <th>5792</th>
      <td>68023798</td>
      <td>2</td>
      <td>08/11/2017 2:14:01 PM</td>
      <td>08/11/2017 3:17:31 PM</td>
      <td>1</td>
      <td>30.33</td>
      <td>2</td>
      <td>N</td>
      <td>132</td>
      <td>158</td>
      <td>1</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>14.64</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>73.20</td>
    </tr>
    <tr>
      <th>15350</th>
      <td>77309977</td>
      <td>2</td>
      <td>09/14/2017 1:44:44 PM</td>
      <td>09/14/2017 2:34:29 PM</td>
      <td>1</td>
      <td>28.23</td>
      <td>2</td>
      <td>N</td>
      <td>13</td>
      <td>132</td>
      <td>1</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>4.40</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>62.96</td>
    </tr>
    <tr>
      <th>10302</th>
      <td>43431843</td>
      <td>1</td>
      <td>05/15/2017 8:11:34 AM</td>
      <td>05/15/2017 9:03:16 AM</td>
      <td>1</td>
      <td>28.20</td>
      <td>2</td>
      <td>N</td>
      <td>90</td>
      <td>132</td>
      <td>1</td>
      <td>52.00</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>11.71</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>70.27</td>
    </tr>
    <tr>
      <th>2592</th>
      <td>51094874</td>
      <td>2</td>
      <td>06/16/2017 6:51:20 PM</td>
      <td>06/16/2017 7:41:42 PM</td>
      <td>1</td>
      <td>27.97</td>
      <td>2</td>
      <td>N</td>
      <td>261</td>
      <td>132</td>
      <td>2</td>
      <td>52.00</td>
      <td>4.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>5.76</td>
      <td>0.3</td>
      <td>63.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sort the data by total amount and print the top 20 values

df_sorted_total_amount = df.sort_values(['total_amount'], ascending=False)['total_amount']
df_sorted_total_amount.head(20)
```




    8476     1200.29
    20312     450.30
    13861     258.21
    12511     233.74
    15474     211.80
    6064      179.06
    16379     157.06
    3582      152.30
    11269     151.82
    9280      150.30
    1928      137.80
    10291     131.80
    6708      126.00
    11608     123.30
    908       121.56
    7281      120.96
    18130     119.31
    13621     115.94
    13359     111.95
    29        111.38
    Name: total_amount, dtype: float64




```python
# Sort the data by total amount and print the bottom 20 values

df_sorted_total_amount.tail(20)
```




    14283      0.31
    19067      0.30
    10506      0.00
    4402       0.00
    22566      0.00
    5722       0.00
    1646      -3.30
    314       -3.80
    18565     -3.80
    5758      -3.80
    10281     -4.30
    5448      -4.30
    4423      -4.30
    8204      -4.80
    20317     -4.80
    11204     -5.30
    14714     -5.30
    20698     -5.80
    17602     -5.80
    12944   -120.30
    Name: total_amount, dtype: float64



### 1. Sorting by trip_distance

When sorting the trip_distance variable from maximum to minimum, the values seem generally plausible, with the longest trip distances around 33.96 miles. 

### 2. Sorting by total_amount

When sorting the total_amount variable, several unusual values appear:

1200.29: This is an extremely high fare and could indicate an error.
450.30: Also quite high, but could be a legitimate fare for a long trip or an unusual circumstance.
Several values near or exactly at 0.00 (Rows 10506, 4402, 22566, 5722): These indicate no fare was recorded, possibly due to trip cancellations or data errors.
Negative values (e.g., -120.30 in Row 12944): These are clearly erroneous, as fares should not be negative.

### 3. Comparison of Sorted Rows

The most expensive rides are not necessarily the longest ones.

Summary:

* The sorted values for trip_distance appear generally plausible but should be contextualized geographically.
* The sorted values for total_amount include several anomalies that need further investigation.
* The differences between the sorted results highlight the importance of examining multiple variables and ensuring data integrity.

</br>

According to the data dictionary, the payment method was encoded as follows:

1 = Credit card  
2 = Cash  
3 = No charge  
4 = Dispute  
5 = Unknown  
6 = Voided trip


```python
# How many of each payment type are represented in the data?

df['payment_type'].value_counts()
```




    payment_type
    1    15265
    2     7267
    3      121
    4       46
    Name: count, dtype: int64




```python
# What is the average tip for trips paid for with credit card?

avg_cc_tip = df[df['payment_type']==1]['tip_amount'].mean()
print('Avg. credit Card tip:', avg_cc_tip)

# What is the average tip for trips paid for with cash?
```

    Avg. credit Card tip: 2.7298001965280054
    


```python
# How many times is each vendor ID represented in the data?

df['VendorID'].value_counts()
```




    VendorID
    2    12626
    1    10073
    Name: count, dtype: int64




```python
# What is the mean total amount for each vendor?

df.groupby(['VendorID']).mean(numeric_only=True)[['total_amount']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_amount</th>
    </tr>
    <tr>
      <th>VendorID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>16.298119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.320382</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter the data for credit card payments only
credit_card_payments = df[df['payment_type']==1]

# Filter the credit-card-only data for passenger count only
credit_card_payments['passenger_count'].value_counts()
```




    passenger_count
    1    10977
    2     2168
    5      775
    3      600
    6      451
    4      267
    0       27
    Name: count, dtype: int64




```python
# Calculate the average tip amount for each passenger count (credit card payments only)

credit_card_payments.groupby(['passenger_count']).mean(numeric_only=True)[['tip_amount']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tip_amount</th>
    </tr>
    <tr>
      <th>passenger_count</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.610370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.714681</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.829949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.726800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.607753</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.762645</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.643326</td>
    </tr>
  </tbody>
</table>
</div>



<img src="images/Construct.png" width="100" height="100" align=left>

## PACE: **Construct**

**Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project. 




<img src="images/Execute.png" width="100" height="100" align=left>

## PACE: **Execute**


### **Given your efforts, what can you summarize for DeShawn and the data team?**

*Note for Learners: Your notebook should contain data that can address Luana's requests. Which two variables are most helpful for building a predictive model for the client: NYC TLC?*

After examining the dataset, it's clear that the two variables most useful for creating a predictive model for taxi ride fares are total_amount and trip_distance. These variables provide a comprehensive view of a taxi cab ride.

**Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
