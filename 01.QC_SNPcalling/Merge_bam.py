#!/usr/bin/env python
#!coding=utf-8
import sys
import re
import pandas as pd
import os
import json
import json

result_list = os.listdir('./report')
result_dict = {}
merge_result_dict = {}
for i in result_list:
    if i.endswith('.json'):
        with open('./report/' + i, 'r') as f:
            result_dict[i] = json.load(f)
for k,v in result_dict.items():
    key = k.split('.')[0]
    merge_result_dict[key] = {'Before_QC': v['summary']['before_filtering']['total_bases'] + merge_result_dict.get(key, {'total_bases': 0}).get('total_bases', 0),
    'After_QC': v['summary']['after_filtering']['total_bases'] + merge_result_dict.get(key, {'total_bases': 0}).get('total_bases', 0),
    'Before_QC>Q30': v['summary']['before_filtering']['q30_rate'] + merge_result_dict.get(key, {'q30_rate': 0}).get('q30_rate', 0),
    'After_>Q30': v['summary']['after_filtering']['q30_rate'] + merge_result_dict.get(key, {'q30_rate': 0}).get('q30_rate', 0),
}

df_ = pd.DataFrame(merge_result_dict)

flist = sys.argv[1]
outfile = sys.argv[2]

tmpfile = outfile + ".tmp"
F = open(flist)
O = open(tmpfile,"w")


lines = F.readlines()
fn = 0
for l in lines:
    l = l.strip()
    if l:
        fn += 1
    L = open(l)
    n = 0
    head = []
    stats = []
    lis = L.readlines()
    if len(lis) <= 3:
        continue
    for dst in lis:
        dst = dst.strip()
        if not dst:
            continue
        n += 1
        if n == 3:
            file = re.sub("## Files : ","",dst)
            namepref = re.sub(r"\.bam","",os.path.basename(file))
            continue
        if re.compile(r"^#").findall(dst):
            continue

        (cn,value) = re.split(r"\t",dst)
        cn = re.sub(r"^\s+","",cn)
        head.append(cn)
        stats.append(value)
    vline = namepref + "\t" + "\t".join(stats)
    if fn == 1:
        O.write("SampleName")
        head = "\t".join(head)
        O.write(head + "\n")

    O.write(vline + "\n")

O.close()
df = pd.read_csv(tmpfile,sep="\t")
df1 = df[["[Target] Average depth(rmdup)","[Target] Coverage (>0x)"]]
df1 = df1.T
# df1.to_excel(outfile)
df2=pd.concat([df_,df1])
print(df2)
df2.to_excel(outfile)
os.remove(tmpfile)
