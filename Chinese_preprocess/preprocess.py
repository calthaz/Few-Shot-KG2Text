import json
import re
import unidecode
import codecs
import random

filename = ['annotated/SAOKE_DATA_ready-0-annotated.json', 
            'annotated/SAOKE_DATA_ready-1-annotated.json', 
            'annotated/SAOKE_DATA_ready-2-annotated.json',
            'annotated/SAOKE_DATA_ready-3-annotated.json']

preprocessed_data = []
for fn in filename:
    print("INFO loading file", fn)
    fin = codecs.open(fn, "r", "utf-8")
    data = json.load(fin)
    #print (len(data))
    fin.close()
    '''
    {"accepted":true,
    "natural":"微博内容不仅有哈士奇的照片，而且还配有生动的解说文字。",
    "triples":[{"ent1":"微博内容","rel":"有","ent2":"哈士奇的照片"},
    {"ent1":"微博内容","rel":"有","ent2":"生动的解说文字"}]}
    '''
    '''
    {
        "triples": [
            [
                "Ajoblanco",
                "region",
                "Andalusia"
            ],
            [
                "Ajoblanco",
                "country",
                "Spain"
            ],
            [
                "Ajoblanco",
                "mainIngredients",
                "Bread, almonds, garlic, water, olive oil"
            ]
        ],
        "target": "From PATIENT_3 , PATIENT_1 , AGENT_1 is made with PATIENT_2 .",
        "target_txt": "From Andalusia , Spain , Ajoblanco is made with bread , almonds , garlic , water and olive oil .",
        "ner2ent": {
            "PATIENT_3": "Andalusia",
            "PATIENT_1": "Spain",
            "AGENT_1": "Ajoblanco",
            "PATIENT_2": "Bread, almonds, garlic, water, olive oil"
        }
    },
    '''
    for d in data:
        if d['accepted']:
            entities = set()
            natural_txt = d['natural']
            new_triples = []
            for triple in d['triples']:
                if (str(triple['ent1']).isdigit() or str(triple['ent2']).isdigit() 
                    or str(triple['rel']).isdigit()):
                    print("INFO:", natural_txt, "extra/missing relations")
                else:
                    entities.add(triple['ent1'])
                    entities.add(triple['ent2'])
                    new_triples.append(triple)
            ent_list = list(entities)
            
            mask2ent = {}
            for index, ent in enumerate(ent_list):
                mask2ent["ENTITY_"+str(index)] = ent
            #print(mask2ent)

            pointer = 0
            masked_text = ""
            while pointer < len(natural_txt):
                found = False
                for index, ent in enumerate(ent_list):
                    if (natural_txt[pointer:pointer+len(ent)] == ent):
                        masked_text += "ENTITY_"+str(index)
                        pointer += len(ent)
                        found = True
                        break
                if not found:
                    masked_text += natural_txt[pointer]
                    pointer += 1 
            new_data = {"triples": new_triples,
                      "target": masked_text,
                      "target_txt": natural_txt,
                      "ner2ent": mask2ent}
            preprocessed_data.append(new_data)

#split
random.seed(20)
random.shuffle(preprocessed_data)
total_num = len(preprocessed_data)
train_set = preprocessed_data[:int(total_num*0.8)]
valid_set = preprocessed_data[int(total_num*0.8):int(total_num*0.9)]
test_set = preprocessed_data[int(total_num*0.9):]

fready = codecs.open("preprocessed/train.json", "w", "utf-8")
fready.write(json.dumps(train_set, ensure_ascii=False, sort_keys=True, indent=4))
fready.close()

fready = codecs.open("preprocessed/valid.json", "w", "utf-8")
fready.write(json.dumps(valid_set, ensure_ascii=False, sort_keys=True, indent=4))
fready.close()

fready = codecs.open("preprocessed/test.json", "w", "utf-8")
fready.write(json.dumps(test_set, ensure_ascii=False, sort_keys=True, indent=4))
fready.close()
