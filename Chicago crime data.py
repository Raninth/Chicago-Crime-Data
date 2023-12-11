#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import tick customization tools
import matplotlib.ticker as mticks


# In[7]:


get_ipython().system('pip')


# In[12]:


file_path = "Crimes_-_2001_to_Present.csv"
data = pd.read_csv(file_path)


# In[14]:


data.head()


# In[16]:


data['Date'] = pd.to_datetime(data['Date'])


# In[17]:


data.info()


# **1) Comparing Police Districts**
# - Which district had the most crimes in 2022?
# - Which had the least?
# 

# In[20]:


#Which district had the most crimes in 2022?
# Filter data for the year 2022
crimes_2022 = data[data['Year'] == 2022]


# In[22]:


district_crime_counts = crimes_2022['District'].value_counts()
district_crime_counts


# In[23]:


# District with the most crimes in 2022
district_most_crimes = district_crime_counts.idxmax()
most_crimes_count = district_crime_counts.max()

# District with the least crimes in 2022
district_least_crimes = district_crime_counts.idxmin()
least_crimes_count = district_crime_counts.min()


# In[24]:


district_least_crimes


# In[25]:


least_crimes_count


# In[26]:


district_most_crimes


# In[27]:


most_crimes_count


# In[28]:


print(f"District with the most crimes in 2022: {district_most_crimes} with {most_crimes_count} crimes")
print(f"District with the least crimes in 2022: {district_least_crimes} with {least_crimes_count} crimes")


# In[29]:


sorted_districts = district_crime_counts.sort_values(ascending=False)

# Plotting the top districts with the most crimes
plt.figure(figsize=(10, 6))
sorted_districts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Districts with the Most Crimes')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[30]:


sorted_districts = district_crime_counts.sort_values(ascending=True)

# Plotting the districts with the lowest crimes
plt.figure(figsize=(10, 6))
sorted_districts.head(10).plot(kind='bar', color='lightgreen')
plt.title('Districts with the lowest Crimes')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# **2) Comparing Months:**
# - What months have the most crime? What months have the least?
# - Are there any individual crimes that do not follow this pattern? If so, which crimes?
# 

# In[33]:


data['Month'] = data['Date'].dt.month


# In[35]:


monthly_crime_counts = data['Month'].value_counts().sort_index().reset_index()
monthly_crime_counts.columns = ['Month', 'Crime Count']


# In[36]:


monthly_crime_counts


# In[37]:


month_most_crime = monthly_crime_counts[monthly_crime_counts['Crime Count'] == monthly_crime_counts['Crime Count'].max()]
month_least_crime = monthly_crime_counts[monthly_crime_counts['Crime Count'] == monthly_crime_counts['Crime Count'].min()]


# In[38]:


print("Month(s) with the most crimes:")
print(month_most_crime)

print("\nMonth(s) with the least crimes:")
print(month_least_crime)


# In[39]:


plt.figure(figsize=(10, 6))
plt.bar(monthly_crime_counts['Month'], monthly_crime_counts['Crime Count'], color='skyblue')
plt.xlabel('Month')
plt.ylabel('Crime Count')
plt.title('Crime Counts per Month')
plt.xticks(range(1, 13))  # Setting x-axis ticks for months (1 to 12)
plt.grid(axis='y')
plt.show()


# In[42]:


crime_counts_by_type_month = data.groupby(['Primary Type', 'Month']).size().reset_index(name='Count')
crime_anomalies = crime_counts_by_type_month[crime_counts_by_type_month['Month'].isin(monthly_crime_counts['Month']) & 
                                            (crime_counts_by_type_month['Count'] < crime_counts_by_type_month['Count'].quantile(0.25))]


# In[43]:


print("\nCrimes that do not follow the general pattern:")
print(crime_anomalies)


# In[44]:


plt.figure(figsize=(10, 6))
for crime_type in crime_anomalies['Primary Type'].unique():
    subset = crime_anomalies[crime_anomalies['Primary Type'] == crime_type]
    plt.plot(subset['Month'], subset['Count'], marker='o', label=crime_type)

plt.xlabel('Month')
plt.ylabel('Crime Count')
plt.title('Anomalies: Crimes with Lower Counts in High-Crime Months')
plt.xticks(range(1, 13))  # Setting x-axis ticks for months (1 to 12)
plt.legend()
plt.grid()
plt.show()


# **Topic 3) Crimes Across the Years:**
# 
# - Is the total number of crimes increasing or decreasing across the years?
# - Are there any individual crimes that are doing the opposite (e.g., decreasing when overall crime is increasing or vice-versa)?

# In[47]:


data['Year'] = data['Date'].dt.year

# Group data by year and count the number of crimes
crime_counts_by_year = data['Year'].value_counts().sort_index().reset_index()
crime_counts_by_year.columns = ['Year', 'Crime Count']


# In[48]:


crime_counts_by_year


# In[50]:


plt.figure(figsize=(10, 6))
plt.plot(crime_counts_by_year['Year'], crime_counts_by_year['Crime Count'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Total Crime Count')
plt.title('Total Crimes Across Years')
plt.grid()
plt.show()


# **Is the total number of crimes increasing or decreasing across the years?
# - it is decreasing

# In[52]:


overall_trend = 'Increasing' if crime_counts_by_year['Crime Count'].iloc[-1] > crime_counts_by_year['Crime Count'].iloc[0] else 'Decreasing'

# Function
def get_individual_crime_trends(data):
    individual_crime_trends = {}
    for crime_type, crime_type_data in data.groupby('Primary Type'):
        crime_counts_by_year_type = crime_type_data['Year'].value_counts().sort_index().reset_index()
        crime_counts_by_year_type.columns = ['Year', 'Crime Count']
        trend = 'Increasing' if crime_counts_by_year_type['Crime Count'].iloc[-1] > crime_counts_by_year_type['Crime Count'].iloc[0] else 'Decreasing'
        if trend != overall_trend:
            individual_crime_trends[crime_type] = trend
    return individual_crime_trends


individual_crime_trends = get_individual_crime_trends(data)

# Displaying overall trend
print(f"Overall trend of total crimes: {overall_trend}")
print("\nIndividual crimes with opposite trends:")
for crime, trend in individual_crime_trends.items():
    print(f"{crime}: {trend}")


# In[ ]:




