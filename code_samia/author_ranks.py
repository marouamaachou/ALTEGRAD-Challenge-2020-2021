



f = open("../data/author_papers.txt","r")
papers_set = set()
d = {}
for l in f:
    auth_paps = [paper_id.strip() for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    d[l.split(":")[0]] = len(auth_paps)
f.close()



df = open("../data/author_ranks.csv","w")
for w in sorted(d, key=d.get, reverse=True):
    df.write(w+","+ str(d[w])+"\n")
df.close()