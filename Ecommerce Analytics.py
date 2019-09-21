
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

df=pd.read_excel('Online Retail.xlsx')
df.head()


# In[2]:


#QUESTION1

#Region Specific reports on Revenue
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate']).dt.strftime('%Y/%m')
df
df['Revenue']=df['Quantity']*df['UnitPrice']
df.Revenue.sum()
df1=df.groupby('Country')['Revenue'].sum()
df1=df1.reset_index()
print(df1)


# In[3]:


plt.figure(figsize=(15,10))
sns.barplot(y="Country",x="Revenue",data=df1)


# In[59]:


#Revenue Trend of top 10 Countries
ax=df1.sort_values(by=['Revenue'],ascending=False)
df2=ax.head(10)
plt.figure(figsize=(10,5))
sns.barplot(x="Country",y="Revenue",data=df2)
plt.xticks(rotation='60')


# In[5]:


df1.to_csv('RevenueReport.csv')


# In[6]:


#QUESTION 2

#Individual Product Earnings
product_revenue=df.groupby(['StockCode','Description'])['Revenue'].sum()

product_revenue=product_revenue[product_revenue>0]
product_revenue=product_revenue.reset_index()
product_revenue


# In[7]:


#QUESTION 3

#top performing products
bx=product_revenue.sort_values(by=['Revenue'],ascending=False)
bx 


# In[8]:


#top 20 best performing products 
b=bx.head(25)
b


# In[9]:


#Plot Products vs revenue
plt.figure(figsize=(15,10))
g=sns.barplot(y="Description",x="Revenue",data=b)


# In[10]:


#QUESTION 4

#10 products customer ignore
i=product_revenue.sort_values(by=['Revenue'],ascending=True)
i.head(10)


# In[11]:


df3=df.groupby('InvoiceDate')['Revenue'].sum()
df3=df3.reset_index()
df3


# In[12]:


#QUESTION 5

#Revenue trend in different time span
plt.figure(figsize=(185,60))
fy=sns.factorplot(x="InvoiceDate",y="Revenue",data=df3,size=8)
plt.xticks(rotation='60')


# In[13]:


#RFM ANALYSIS

dfn1=df.copy()
dfn=dfn1.dropna(axis=0,subset=['CustomerID'],inplace=False)
dfn['InvoiceDate']=pd.to_datetime(dfn['InvoiceDate']).dt.strftime('%Y-%m-%d')
dfn.info()


# In[14]:


dfn['InvoiceDate'].max()
now = dt.date(2011,12,10)
now=pd.to_datetime(now)
print(now)


# In[63]:


dfn['InvoiceDate']=pd.to_datetime(dfn['InvoiceDate'])
dfn['Recency'] =(now-dfn['InvoiceDate']).dt.days
dfn.head(10)


# In[62]:


#Recency
recency= dfn.groupby(by='CustomerID', as_index=False)['Recency'].min()
recency.columns = ['CustomerID','recency']
recency.sort_values(by=['recency'],ascending=True)
recency.head()


# In[61]:


dfn_copy = dfn.copy()
dfn_copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
#Frequency of purchases
frequency = dfn_copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
frequency.columns = ['CustomerID','Frequency']
frequency.sort_values(by=['Frequency'],ascending=False)
frequency.head()


# In[60]:


#Monetary
dfn['Revenue']=dfn['Quantity']*dfn['UnitPrice']
monetary= dfn.groupby('CustomerID')['Revenue'].sum()
monetary=monetary.reset_index()
monetary.head()


# In[64]:


#Merging Recency,Frequency and Monetary
temp= recency.merge(frequency,on='CustomerID')
rfm = temp.merge(monetary,on='CustomerID')


# In[65]:


#CustomerID as index
rfm=rfm.set_index('CustomerID')
rfm.head(10)


# In[66]:


quantiles = rfm.quantile(q=[0.25,0.5,0.75])
quantiles
quantiles.to_dict()


# In[67]:


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1  


# In[68]:


seg_rfm = rfm.copy()
seg_rfm['Rscore'] = seg_rfm['recency'].apply(RScore, args=('recency',quantiles,))
seg_rfm['FScore'] = seg_rfm['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
seg_rfm['MScore'] = seg_rfm['Revenue'].apply(FMScore, args=('Revenue',quantiles,))
seg_rfm.head()


# In[69]:


seg_rfm['RFMScore'] = seg_rfm.Rscore.map(str) + seg_rfm.FScore.map(str) + seg_rfm.MScore.map(str)
seg_rfm.sort_values(by='RFMScore')
seg_rfm.head()


# In[25]:


#QUESTION 6

#Loyal Customers
seg_rfm[seg_rfm['FScore']==1].sort_values('Frequency', ascending=False).head(30)


# In[26]:


#QUESTION 7
print("Best Customers: ",len(seg_rfm[seg_rfm['RFMScore']=='111']))
print('Loyal Customers: ',len(seg_rfm[seg_rfm['FScore']==1]))
print("Big Spenders: ",len(seg_rfm[seg_rfm['MScore']==1]))
print('Almost Lost: ', len(seg_rfm[seg_rfm['RFMScore']=='433']))
print('Lost Customers: ',len(seg_rfm[seg_rfm['RFMScore']=='411']))
print('Lost Cheap Customers: ',len(seg_rfm[seg_rfm['RFMScore']=='344']))


# In[27]:


#QUESTION 8

#Customers at risk
seg_rfm[seg_rfm['RFMScore']=='344'].sort_values('RFMScore', ascending=False).head(30)


# In[28]:


#QUESTION 9

#customers who try new products
seg_rfm[seg_rfm['RFMScore']=='111'].sort_values('RFMScore', ascending=False).head(30)


# In[29]:


#QUESTION 10
#inactive customers 
seg_rfm[seg_rfm['RFMScore']=='444'].sort_values('Frequency', ascending=False).head(30)


# In[70]:


#QUESTION 11

#NEW CUSTOMERS
fn=df.copy()
fn.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
fn=fn.dropna(axis=0,subset=['CustomerID'],inplace=False)
fn.head()


# In[31]:


dec=fn[fn['InvoiceDate']=="2010/12"]
jan=fn[fn['InvoiceDate']=="2011/01"]
feb=fn[fn['InvoiceDate']=="2011/02"]
mar=fn[fn['InvoiceDate']=="2011/03"]
apr=fn[fn['InvoiceDate']=="2011/04"]
may=fn[fn['InvoiceDate']=="2011/05"]
june=fn[fn['InvoiceDate']=="2011/06"]
july=fn[fn['InvoiceDate']=="2011/07"]
aug=fn[fn['InvoiceDate']=="2011/08"]
sep=fn[fn['InvoiceDate']=="2011/09"]
octb=fn[fn['InvoiceDate']=="2011/10"]
nov=fn[fn['InvoiceDate']=="2011/11"]
dec1=fn[fn['InvoiceDate']=="2011/12"]

dec=dec.CustomerID
dec=dec.reset_index()
jan=jan.CustomerID
jan=jan.reset_index()
feb=feb.CustomerID
feb=feb.reset_index()
mar=mar.CustomerID
mar=mar.reset_index()
apr=apr.CustomerID
apr=apr.reset_index()
may=may.CustomerID
may=may.reset_index()
june=june.CustomerID
june=june.reset_index()
july=july.CustomerID
july=july.reset_index()
aug=aug.CustomerID
aug=aug.reset_index()
sep=sep.CustomerID
sep=sep.reset_index()
octb=octb.CustomerID
octb=octb.reset_index()
nov=nov.CustomerID
nov=nov.reset_index()
dec1=dec1.CustomerID
dec1=dec1.reset_index()


# In[32]:


#New Customers in January
g = []
for item in jan['CustomerID']:
    if item not in list(dec['CustomerID']):
        g.append(item)
print(len(g))


# In[33]:


frame=[dec,jan]
pp=pd.concat(frame)


# In[34]:


#New Customers in Feb
h= []
for item in feb['CustomerID']:
    if item not in list(pp['CustomerID']):
        h.append(item)
print(len(h))


# In[35]:


frame=[dec,jan,feb]
ppp=pd.concat(frame)


# In[36]:


#New Customers in Mar
i= []
for item in mar['CustomerID']:
    if item not in list(ppp['CustomerID']):
        i.append(item)
print(len(i))


# In[37]:


frame=[dec,jan,feb,mar]
p4=pd.concat(frame)


# In[38]:


#New Customers in Apr
j= []
for item in apr['CustomerID']:
    if item not in list(p4['CustomerID']):
        j.append(item)
print(len(j))


# In[39]:


frame=[dec,jan,feb,mar,apr]
p5=pd.concat(frame)


# In[40]:


#New Customers in May
k= []
for item in may['CustomerID']:
    if item not in list(p5['CustomerID']):
        k.append(item)
print(len(k))


# In[41]:


frame=[dec,jan,feb,mar,apr,may]
p6=pd.concat(frame)


# In[42]:


#New Customers in June
l= []
for item in june['CustomerID']:
    if item not in list(p6['CustomerID']):
        l.append(item)
print(len(l))


# In[43]:


frame=[dec,jan,feb,mar,apr,may,june]
p7=pd.concat(frame)


# In[44]:


#New Customers in July
m= []
for item in july['CustomerID']:
    if item not in list(p7['CustomerID']):
        m.append(item)
print(len(m))


# In[45]:


frame=[dec,jan,feb,mar,apr,may,june,july]
p8=pd.concat(frame)


# In[46]:


#New Customers in August
n= []
for item in aug['CustomerID']:
    if item not in list(p8['CustomerID']):
        n.append(item)
print(len(n))


# In[47]:


frame=[dec,jan,feb,mar,apr,may,june,july,aug]
p9=pd.concat(frame)


# In[48]:


#New Customers in September
o= []
for item in sep['CustomerID']:
    if item not in list(p9['CustomerID']):
        o.append(item)
print(len(o))


# In[49]:


frame=[dec,jan,feb,mar,apr,may,june,july,aug,sep]
p10=pd.concat(frame)


# In[50]:


#New Customers in october
p= []
for item in octb['CustomerID']:
    if item not in list(p10['CustomerID']):
        p.append(item)
print(len(p))


# In[51]:


frame=[dec,jan,feb,mar,apr,may,june,july,aug,sep,octb]
p11=pd.concat(frame)


# In[52]:


#New Customers in November
q= []
for item in nov['CustomerID']:
    if item not in list(p11['CustomerID']):
        q.append(item)
print(len(q))


# In[53]:


frame=[dec,jan,feb,mar,apr,may,june,july,aug,sep,octb,nov]
p12=pd.concat(frame)


# In[54]:


#New Customers in December
r= []
for item in dec1['CustomerID']:
    if item not in list(p12['CustomerID']):
        r.append(item)
print(len(r))


# In[73]:


#QUESTION 12

#MARKET BASKET ANALYSIS
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df=dfn1.copy()
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df.head()


# In[72]:


basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
basket.head()


# In[71]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets.head()


# In[76]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequent_itemsets.head()


# In[77]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()


# In[78]:


rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
#antecedents are items which customers bought
#consequents are items which customers tends to buy

