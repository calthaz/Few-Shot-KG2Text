import json
import codecs

filename = ['SAOKE_DATA.json']

for fn in filename:
    fin = open(fn, "r", encoding="utf-8")
    data = []
    for line in fin:
        data.append(json.loads(line.strip()))
    fin.close()
    print(data[0])
    fout = codecs.open(fn[:-5] + "_peek.json", "w", "utf-8")

    fout.write(json.dumps(data[0:200], ensure_ascii=False, sort_keys=True, indent=4))
    fout.close()    #for d in data:
    #    for k, v in d.items(): 
    #        #v = bytes(v, "ascii")
    #        print(k, v)
    #    break

    fout = codecs.open(fn[:-5] + "_natural.txt", "w", "utf-8")
    
    new_data = []
    for d in data:
        if(len(d["natural"])<20):
            continue
        valid = False
        for l in d["logic"]:
            if(l["subject"] != "_"):
                valid = True
                break
        if (valid):
            new_d = {
                "natural": d["natural"],
                "triples": [],
                "accepted": False,
            }
            fout.write(d["natural"]+"\n")
            new_data.append(new_d)

    for x in range(100):
        fready = codecs.open(fn[:-5] + "_ready-"+str(x)+".json", "w", "utf-8")
        fready.write(json.dumps(new_data[x*100:(x+1)*100], ensure_ascii=False, sort_keys=True, indent=4))
        fready.close()

    #fready = codecs.open(fn[:-5] + "_ready-"+str(x+1)+".json", "w", "utf-8")
    #fready.write(json.dumps(new_data[(x+1)*100:], ensure_ascii=False, sort_keys=True, indent=4))
    #fready.close()
    #fout.close()

# sanwen?
# https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master/relation_extraction/Validation
# finance 
# https://github.com/thunlp/Chinese_NRE/blob/master/data/FinRE/test.txt
