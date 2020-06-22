#importing necessary libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import gc
#------------------------------Importing the Dataset(From Keggle)--------------------------------------------#
df = pd.read_csv('Dataset.csv',header = 0, names= ['ts', 'src', 'dst', 'port', 'bytes'])
df.info()

df.head()
#------------------------------------------Data Preprocessing------------------------------------------------#
df.head()def is_internal(s):
  return s.str.startswith(('12.', '13.', '14.'))
  
df['src_int'] = is_internal(df['src'])
df['dst_int'] = is_internal(df['dst'])
df['ts']      = pd.to_datetime(df.ts, unit='ms')
df['hour']    = df.ts.dt.hour.astype('uint8')
df['minute']  = df.ts.dt.minute.astype('uint8')
df['port']    = df['port'].astype('uint8')
df.head()

all_ips = set(df['src'].unique()) | set(df['dst'].unique())
print('Unique src:', df['src'].nunique())
#Unique src: 5970

print('Unique dst:', df['dst'].nunique())
#Unique dst: 5999

print('Total Unique IPs:', len(all_ips))
#Total Unique IPs: 5999

ip_type = pd.CategoricalDtype(categories=all_ips)
df['src'] = df['src'].astype(ip_type)
df['dst'] = df['dst'].astype(ip_type)
gc.collect()
df.info()
#-------------------------------------------------Data Analysis------------------------------------------------#

blacklist_ips = []    #contains IP's with higher outbound traffic and timed out IPs(higer Activity per hour)
answers = []          #contains Ip's with the corresponding port number, this list will be used to later tune the algorithm
src_bytes_out = df[df['src_int'] & ~df['dst_int']]\
.groupby('src')\.bytes.sum()\.pipe(lambda x: x[x > 0])\.sort_values(ascending=False)
src_bytes_out.to_frame().head()

src_bytes_out.head(10)\
    .sort_values()\
    .plot.barh(title='Top 10 high outbound traffic srcs')\
    .set_xlabel('total outbound bytes')
    
    ax = src_bytes_out\
  .plot.hist(bins=50, title='Outbound traffic per src')

ax.set_xlabel('total outbound bytes')
_ = ax.axvline(src_bytes_out.iloc[0], linestyle='--')
plt.text(src_bytes_out.iloc[0], 100, '13.37.84.125', rotation=90, horizontalalignment='right')

blacklist_ips.append('13.37.84.125')
answers.append('13.37.84.125')
df.groupby('hour').size()\
  .plot.bar(title='Activity per hour')\
  .set_ylabel('Connection counts')
  
off_hours_activity = df[
    ~df['src'].isin(blacklist_ips)          # Not including previous answers
    & df['src_int'] & ~df['dst_int']        # Outbound
    & (df['hour'] >= 0) & (df['hour'] < 16) # Off hours
].groupby('src')\
  .bytes.sum()\
  .sort_values(ascending=False)\
  .where(lambda x: x > 0)

off_hours_activity.head()
ax = src_bytes_out\
  .plot.hist(bins=50, title='Outbound traffic per src')

ax.set_xlabel('total outbound bytes')
_ = ax.axvline(src_bytes_out.loc['12.55.77.96'], color='k', linestyle='--')
plt.text(src_bytes_out.loc['12.55.77.96'], 100, '12.55.77.96', rotation=90, horizontalalignment='right')

blacklist_ips.append('12.55.77.96')
answers.append('12.55.77.96')

src_port_bytes_df = df[
        ~df['src'].isin(blacklist_ips)     # Not including previous answers
        & df['src_int'] & ~df['dst_int']   # Outbound
    ].groupby(['src', 'port'])\
        .bytes.sum()\
        .reset_index()

ports = src_port_bytes_df['port'].unique()
print('Number of unique ports:', len(ports))

src_port_bytes_df[src_port_bytes_df.port == 113]

src_port_bytes_df.groupby('port')\
    .bytes.sum()\
    .sort_values(ascending=False)\
    .plot.bar(figsize=(16,4), rot=0, title="Outbound bytes per port")\
    .set_ylabel('Total outbound bytes')
fig, axs = plt.subplots(ncols=3, nrows=3, sharey=True, figsize=(12,6))

for idx, p in enumerate(src_port_bytes_df.port.head(9)):
    src_port_bytes_df[src_port_bytes_df.port == p]\
        .bytes.plot.hist(title='Distribution for port {}'.format(p), ax = axs[idx % 3][idx // 3])\
        .set_xlabel('total outbound bytes')
plt.tight_layout()
     src_port_bytes_df\
      .groupby('port')\
      .apply(lambda x: np.max((x.bytes - x.bytes.mean()) / x.bytes.std()))\
      .sort_values(ascending=True)\
      .tail(10)\
      .plot.barh(title='Top z-score value per port')\
      .set_xlabel('Max z-score')
 src_124 = src_port_bytes_df\
  .pipe(lambda x: x[x['port'] == 124])\
  .sort_values('bytes', ascending=False).head(1)

src_124
ax = src_port_bytes_df[src_port_bytes_df.port == 124]\
    .bytes.plot.hist(bins=50, title='Distribution of outbound data usage for port 124')

ax.set_xlabel('total outbound bytes')
_ = ax.axvline(src_124.iloc[0, 2], linestyle='--')
plt.text(src_124.iloc[0, 2], 100, '12.30.96.87', rotation=90, horizontalalignment='right')
blacklist_ips.append('12.30.96.87')
answers.append('124')

df[~df['src_int']]\
  .drop_duplicates(('src', 'port'))\
  .groupby('port').size()\
  .sort_values()\
  .head()
df[~df['src_int'] & (df['port'] == 113)][['src', 'dst', 'port']]
df[(df['src'] == '15.104.76.58') & (df['dst'] == '12.45.104.32')]\
    [['src', 'dst', 'port']]
 answers.append('113')
 #-----------------------------------------------------Algorithm-------------------------------------------------#
 import networkx
from networkx.algorithms.approximation.clique import large_clique_size 
from collections import Counter

internal_edges_all = df[
  df['src_int'] & df['dst_int']
].drop_duplicates(['src', 'dst', 'port'])
internal_ports = internal_edges_all.port.unique()

port_upper_bounds = []
for p in internal_ports:
    internal_edges = internal_edges_all\
        .pipe(lambda x: x[x['port'] == p])\
        .drop_duplicates(['src', 'dst'])

    edges = set()
    for l, r in zip(internal_edges.src, internal_edges.dst):
        k = min((l, r), (r, l))
        edges.add(k)
    
    degrees = Counter()
    for (l, r) in edges:
        degrees[l] += 1
        degrees[r] += 1
    
    max_clique_size = 0
    min_degrees = len(degrees)
    for idx, (node, degree) in enumerate(degrees.most_common()):
        min_degrees = min(min_degrees, degree)
        if min_degrees >= idx:
            max_clique_size = max(max_clique_size, idx+1)
        if min_degrees < max_clique_size:
            break
            
    port_upper_bounds.append((p, max_clique_size + 1))
    
 port_upper_bounds.sort(key = lambda x: -x[-1])
port_upper_bounds[:5]

max_port = 0
curr_max_clique = 0
for p, max_clique_upper_bound in port_upper_bounds:
    if curr_max_clique > max_clique_upper_bound: break
    
    internal_edges = internal_edges_all\
        .pipe(lambda x: x[x['port'] == p])\
        .drop_duplicates(['src', 'dst'])
  
    internal_nodes = set(internal_edges.src) | set(internal_edges.dst)
    G = networkx.Graph()
    G.add_nodes_from(internal_nodes)
    for l, r in zip(internal_edges.src, internal_edges.dst):
        G.add_edge(l, r)        
        
    _size = large_clique_size(G) 
    if curr_max_clique < _size:
        curr_max_clique = _size
        max_port = p
  print('Port {} has approx. max clique size {}'.format(max_port, curr_max_clique))
answers.append(str(max_port))


single_dst = df[~df['src_int'] & df['dst_int']]\
    .drop_duplicates(['src', 'dst'])\
    .src.value_counts()\
    .pipe(lambda x: x[x == 1])\
    .index

print('Count of "little reason" src:', len(single_dst))
  
#----------------------------------------------------Tunning the Algorithm-----------------------------------------#
df[~df['src_int'] & df['dst_int']]\
    .pipe(lambda x: x[x.src.isin(single_dst)])\
    .drop_duplicates(['src', 'dst'])\
    .groupby('dst').size()\
    .where(lambda x: x > 0).dropna()
    
 df[~df['src_int'] & df['dst_int']]\
  .pipe(lambda x: x[x.src.isin(single_dst)])\
  .drop_duplicates(['src', 'dst'])\
  .head()
  
blacklist_ips.append('14.45.67.46')
answers.append('14.45.67.46')



df[
    df['src_int'] & df['dst_int']
    & (df['dst'] == '14.45.67.46')
    & (df['port'] == 27)
].drop_duplicates('src')
blacklist_ips.append('14.51.84.50')
answers.append('14.51.84.50')
periodic_callbacks = df[df['src_int'] & ~df['dst_int']]\
  .drop_duplicates(['dst', 'minute'])\
  .groupby('dst').size()\
  .pipe(lambda x: x[(x > 0) & (x <= 4)])\
  .sort_values()

periodic_callbacks

answers.append('51')

  df[df.dst.isin(periodic_callbacks.index)]\
    .ts.diff()\
    .dt.total_seconds()\
    .plot.hist(bins=50)
    
 dst_counts = df[df['src_int'] & df['dst_int']]\
    .drop_duplicates(['src', 'dst'])\
    .groupby('src').size()\
    .sort_values(ascending=False)
dst_counts.head()

blacklist_ips.append('13.42.70.40')
answers.append('13.42.70.40')

# Getting internal only connections
int_df = df[df['src_int'] & df['dst_int']]\
    .pipe(lambda x: x[~x.src.isin(blacklist_ips)])\
    .drop_duplicates(('src', 'dst', 'port'))
    
print('Unique dsts')
int_df\
  .drop_duplicates(['src', 'dst'])\
  .groupby('src').size()\
  .sort_values(ascending=False).head()
 
 print('Predicted IP & Ports')
int_df\
  .drop_duplicates(['src', 'port'])\
  .groupby('src').size()\
  .sort_values(ascending=False).head()
  #-----------------------------------------Predicted Outputs-----------------------------------------------------------------------#
  Predicted IP & Ports
src
14.43.40.85     45
14.49.102.62    45
13.57.72.73     45
14.30.26.123    45
13.37.85.29     45
dtype: int64

