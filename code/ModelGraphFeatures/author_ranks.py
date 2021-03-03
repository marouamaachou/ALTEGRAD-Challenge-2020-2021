import os


def author_ranks(path_to_data="ModelGraphFeatures\\data", output_file="author_ranks.csv"):
    """ write a csv file with the rank associated to each author """

    f = open("author_papers.txt","r")
    papers_set = set()
    d = {}
    for l in f:
        auth_paps = [paper_id.strip() for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
        d[l.split(":")[0]] = len(auth_paps)
    f.close()

    try:
        df = open(path_to_data + "\\" + output_file, "w")
    except FileNotFoundError:
        os.mkdir(path_to_data)
        df = open(path_to_data + "\\" + output_file, "w")
    df.write('authorID,rank\n')
    for w in sorted(d, key=d.get, reverse=True):
        df.write(w+","+ str(d[w])+"\n")
    df.close()