import json
import re
import unidecode
import codecs
import random
import spacy
import networkx as nx
#from transformers import RobertaTokenizer, BartTokenizer, BertTokenizer
#from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
bert_tokenizer.bos_token="<s>"
bert_tokenizer.eos_token="</s>"
bert_tokenizer.mask_token="[MASK]"
#model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")x

class NLP:
    def __init__(self):
        self.nlp = spacy.load('zh_core_web_sm', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        #text = ' '.join(text.split())
        #if lower:
        #    text = text
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        #print(toks)
        return ' '.join(toks)


nlp = NLP()


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t)
    new_d = " ".join(new_d)
    return new_d


def get_nodes(n):
    #n = unidecode.unidecode(n.strip())#.lower()
    n = n.replace('-', ' ')
    n = n.replace('_', ' ')
    n = nlp.word_tokenize(n)

    return n


def get_relation(n):
    n = unidecode.unidecode(n.strip().lower())
    n = n.replace('-', ' ')
    n = n.replace('_', ' ')
    n = nlp.word_tokenize(n)

    return n


def get_text(txt, lower=True):
    if lower:
        txt = txt.lower()
    txt = unidecode.unidecode(txt.strip())
    txt = txt.replace('-', ' ')
    txt = nlp.word_tokenize(txt)

    return txt


def BFS_nx(graph, s):
    queue = [s]
    seen = [s]
    node_seq = []
    while queue:
        vertex = queue.pop(0)
        #adj_nodes = graph[vertex].item()
        for w in list(graph.neighbors(vertex)):
            if w not in seen:
                queue.append(w)
                seen.append(w)
        node_seq.append(vertex)
    return node_seq

def BFS(graph, s):
    queue = [s]
    seen = [s]
    node_seq = []
    while queue:
        vertex = queue.pop(0)
        adj_nodes = graph[vertex]
        for w in adj_nodes:
            if w not in seen:
                queue.append(w)
                seen.append(w)
        node_seq.append(vertex)
    return node_seq


def DFS(graph, s):
    stack = [s]
    seen = [s]
    node_seq = []
    while stack:
        vertex = stack.pop()
        adj_nodes = graph[vertex]
        for w in adj_nodes:
            if w not in seen:
                stack.append(w)
                seen.append(w)
        node_seq.append(vertex)
    return node_seq

print(get_nodes("我很快乐"))

#bert_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
#bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# print(tokenizer.decoder_start_token_id)
# exit(0)

filename = ['../Chinese_preprocess/preprocessed/train.json', 
            '../Chinese_preprocess/preprocessed/valid.json', 
            '../Chinese_preprocess/preprocessed/test.json']

for fn in filename:
    fin = codecs.open(fn, "r", "utf-8")
    data = json.load(fin)
    print ("INFO data set length", fn, len(data))
    fin.close()

    fout = codecs.open(fn[:-5] + "_processed.json", "w", "utf-8")
    for d in data:
        new_dict = dict()

        # -------WebNLG dataset------
        valid = True
        ner_dict = {}
        ren_dict = {}
        for k, v in d['ner2ent'].items():
           en = v #get_nodes(v)

           if en == "":
               valid = False
           ner_dict[k] = en
           ren_dict[en] = k
        new_dict['ner2ent'] = ner_dict
        new_dict['ent2ner'] = ren_dict
        # -------WebNLG dataset------

        if not valid:
            continue

        temp = []
        serialization = []
        for tri in d['triples']:
            h =  tri ['ent1']#get_nodes(tri[0])
            t =  tri ['ent2']#get_nodes(tri[2])
            r =  tri ['rel']#camel_case_split(get_relation(tri[1]))
            new_t = [h, r, t]
            temp.append(new_t)
            serialization.extend(["<Head>", h, "<Relation>", r, "<Tail>", t])
        new_dict['triples'] = temp
        new_dict['triples_serialization'] = serialization

        #tokens = []
        #for token in d['target'].split():
        #    if token.isupper() and '_' in token:
        #        tokens.append(token)
        #    else:
        #        tokens.append(token.lower())
        new_dict['target'] = d['target'] #get_text(' '.join(tokens), lower=False)

        #print("before checking nodes exists in ner2ent")
        reg_str = r'ENTITY_[0-9]+'
        list_of_masks = re.findall(reg_str, new_dict['target'])
        #print(list_of_masks)
        
        try:
            tokens = []
            nodes = []

            pointer = 0
            while pointer < len(new_dict['target']):
                found = False
                for index, mask in enumerate(list_of_masks):
                    if (new_dict['target'][pointer:pointer+len(mask)] == mask):
                        tokens.append(new_dict['ner2ent'][mask])
                        if new_dict['ner2ent'][mask] not in nodes:
                            nodes.append(new_dict['ner2ent'][mask])
                        pointer += len(mask)
                        found = True
                        break
                if not found:
                    tokens.append(new_dict['target'][pointer])
                    pointer += 1 
            new_dict['target_txt'] = (''.join(tokens))
        except KeyError:
            continue

        new_dict['plm_output'] = bert_tokenizer.tokenize(new_dict['target_txt'])
        
        #print(new_dict['target_txt'])
        #print(new_dict['plm_output'])
        #exit()
        
        test_output = []
        pointer = []
        #------------------
        text_pointer = 0
        plm_pointer = 0
        while plm_pointer < len(new_dict['plm_output']):
            found = False
            for index, mask in enumerate(list_of_masks):
                if (new_dict['target'][text_pointer:text_pointer+len(mask)] == mask):
                    ent = new_dict['ner2ent'][mask]
                    ent = bert_tokenizer.tokenize(ent)
                    test_output.extend(ent)
                    plm_pointer += len(ent)
                    text_pointer += len(mask)
                    pointer.extend([1] * len(ent))
                    found = True
                    break
            if not found:
                test_output.extend(new_dict['plm_output'][plm_pointer])
                text_pointer += len(new_dict['plm_output'][plm_pointer])
                plm_pointer += 1
                pointer.extend([0])
        #-------------------
        #print(test_output)
        #print(new_dict['plm_output'])
        if not (len(pointer) == len(new_dict['plm_output'])):
            print("ERROR: The length of pointer and output are not equal!")
            print("test", test_output)
            print("plm", new_dict['plm_output'])
            exit()
            
        #assert test_output == new_dict['plm_output'],  "The test output and plm output are not equal!"
        if not (test_output == new_dict['plm_output']):
            print("WARNING: The test output and plm output are not equal!")
            print("test", test_output)
            print("plm", new_dict['plm_output'])
            # exit()


        new_dict['pointer'] = pointer

        # adject = dict() ??
        for t in new_dict['triples']:
            if t[0] not in nodes:
                nodes.append(t[0])
            if t[2] not in nodes:
                nodes.append(t[2])

        new_dict['nodes'] = nodes

        edges = [[], []]
        types = []
        for t in new_dict['triples']:
            hid = new_dict['nodes'].index(t[0])
            tid = new_dict['nodes'].index(t[2])
            edges[0].append(hid)
            edges[1].append(tid)
            types.append(t[1])
            #edges[1].append(hid)
            #edges[0].append(tid)
            #types.append(t[1]+"<-1>") #TODO should not have symmetry?? 
        new_dict['edges'] = edges
        new_dict['types'] = types

        # ------------ bfs ----------------------
        G = nx.DiGraph()
        for t in new_dict['triples']:
            G.add_node(t[0])
            G.add_node(t[2])
            G.add_edge(t[0], t[2], attr={'edge_type': t[1]})
        node_list = []
        for x in range(len(new_dict['triples'])):
            if (new_dict['triples'][x][0] not in node_list):
                source = new_dict['triples'][x][0] 
                nodes_from_source = list(nx.dfs_tree(G, source=new_dict['triples'][x][0]).nodes())
                for ns in nodes_from_source:
                    if(ns not in node_list):
                        node_list.append(ns)
        assert len(node_list) == len(nodes), \
            "traversal gives non-equal nodes sorted: {}, original: {}".format(
                json.dumps(node_list,  ensure_ascii=False), json.dumps(nodes,  ensure_ascii=False))
        new_dict['sorted_node_list'] = node_list
        # bfs_edges = [[], []]
        # bfs_types = []
        # for t in new_dict['triples']:
        #     hid = new_dict['bfs_nodes'].index(t[0])
        #     tid = new_dict['bfs_nodes'].index(t[2])
        #     bfs_edges[0].append(hid)
        #     bfs_edges[1].append(tid)
        #     bfs_types.append(t[1])
        #     bfs_edges[1].append(hid)
        #     bfs_edges[0].append(tid)
        #     bfs_types.append(t[1])
        # new_dict['bfs_edges'] = bfs_edges
        # new_dict['bfs_types'] = bfs_types
        #
        # new_dict['dfs_nodes'] = new_dict['nodes']
        #
        # dfs_edges = [[], []]
        # dfs_types = []
        # for t in new_dict['triples']:
        #     hid = new_dict['dfs_nodes'].index(t[0])
        #     tid = new_dict['dfs_nodes'].index(t[2])
        #     dfs_edges[0].append(hid)
        #     dfs_edges[1].append(tid)
        #     dfs_types.append(t[1])
        #     dfs_edges[1].append(hid)
        #     dfs_edges[0].append(tid)
        #     dfs_types.append(t[1])
        # new_dict['dfs_edges'] = dfs_edges
        # new_dict['dfs_types'] = dfs_types
        #
        # new_dict['shuffle_nodes'] = nodes
        # random.shuffle(new_dict['shuffle_nodes'])
        #
        # shuffle_edges = [[], []]
        # shuffle_types = []
        # for t in new_dict['triples']:
        #     hid = new_dict['shuffle_nodes'].index(t[0])
        #     tid = new_dict['shuffle_nodes'].index(t[2])
        #     shuffle_edges[0].append(hid)
        #     shuffle_edges[1].append(tid)
        #     shuffle_types.append(t[1])
        #     shuffle_edges[1].append(hid)
        #     shuffle_edges[0].append(tid)
        #     shuffle_types.append(t[1])
        # new_dict['shuffle_edges'] = shuffle_edges
        # new_dict['shuffle_types'] = shuffle_types

        word_nodes = [bert_tokenizer.tokenize(node) for node in new_dict['sorted_node_list'] ] #new_dict['nodes']
        new_dict['split_nodes'] = [nd for nodes in word_nodes for nd in nodes]

        start = 0
        split2start = {}
        for idx in range(len(word_nodes)):
            split2start[idx] = start
            start += len(word_nodes[idx])

        split_edges = [[], []]
        split_types = []
        pairs = []
        relations = []
        for tri in new_dict['triples']:
            h, r, t = bert_tokenizer.tokenize(tri[0]), tri[1], bert_tokenizer.tokenize(tri[2])
            hidx = word_nodes.index(h)
            tidx = word_nodes.index(t)
            pairs.append([[split2start[hidx], split2start[hidx] + len(h) - 1],
                          [split2start[tidx], split2start[tidx] + len(t) - 1]])
            relations.append(r)
            for i, hn in enumerate(word_nodes[hidx]):
                for j, tn in enumerate(word_nodes[tidx]):
                    split_edges[0].append(split2start[hidx] + i)
                    split_edges[1].append(split2start[tidx] + j)
                    split_types.append(r)
                    #split_edges[1].append(split2start[hidx] + i)
                    #split_edges[0].append(split2start[tidx] + j)
                    #split_types.append(r+"<-1>") #TODO should not have symmetry?? 
        new_dict['split_edges'] = split_edges
        new_dict['split_types'] = split_types
        new_dict['pairs'] = pairs
        new_dict['relations'] = relations

        assert len(new_dict['pairs']) == len(new_dict['relations']), "the length of pairs and relations are not equal"

        target_tokens = new_dict['target'].split()

        order2ent = {}
        used_ner = set()
        new_target_tokens = []
        order = 1
        #--------------------
        text_pointer = 0
        while text_pointer < len(new_dict['target']):
            found = False
            for index, mask in enumerate(list_of_masks):
                if (new_dict['target'][text_pointer:text_pointer+len(mask)] == mask):
                    if mask not in used_ner:
                        new_target_tokens.append('[MASK]')
                        ent = new_dict['ner2ent'][mask]
                        used_ner.add(mask)
                        order2ent[order] = ent
                        order += 1
                    else:
                        ent = new_dict['ner2ent'][mask]
                        new_target_tokens.append(ent)
                    text_pointer += len(mask)
                    found = True
                    break
            if not found:
                new_target_tokens.append(new_dict['target'][text_pointer])
                text_pointer += 1
        #-------------------------------

        teacher_cloze_tokens = []
        last_is_extra = False
        extra_count = 0
        labels = []
        order = 1
        masked_teacher_tokens = []
        for idx, token in enumerate(new_target_tokens):
            if token == '[MASK]':
                ent = order2ent[order]
                teacher_cloze_tokens.extend(bert_tokenizer.tokenize(ent))
                masked_teacher_tokens.append('[MASK]')
                last_is_extra = False
                order += 1
            else:
                if not last_is_extra:
                    extra_token = 'extra'+str(extra_count)
                    extra_count += 1
                    teacher_cloze_tokens.append(extra_token)
                    masked_teacher_tokens.append(extra_token)
                    last_is_extra = True
                    labels.append(extra_token)
                labels.append(token)
                
                
        labels = bert_tokenizer.tokenize(''.join(labels))
        new_dict['teacher_cloze_tokens'] = teacher_cloze_tokens
        new_dict['labels'] = labels

        #print(labels)
        #print(teacher_cloze_tokens)
        #exit()


        target_tokens = ["<s>"] + bert_tokenizer.tokenize(''.join(masked_teacher_tokens)) + ["</s>"]#

        positions = [[0] * len(bert_tokenizer.tokenize(ent)) for ent in new_dict['sorted_node_list']]#nodes
        masked_target_tokens = []
        new_target_tokens = []
        order = 1
        # target token = ['<s>', '[MASK]', '的', '[MASK]', '其', '[MASK]', '[MASK]', '，', '[MASK]', '也', '[MASK]', '，', '[MASK]', '。', '</s>']
        #"这 些 变 化 <extra0> 消 费 者 心 理 <extra1>消费者心理<extra2>新变化<extra3>新特点"
        #cloze "粗粒及不等粒结构extra0石材extra2外观效果较差extra3力学性能extra4不均匀extra5质量稍差extra6"
        #labels extra0的extra2其extra3，extra4也extra5，extra6。
        last_is_extra = False
        for idx, token in enumerate(target_tokens):
            if token == '[MASK]':
                ent = order2ent[order]
                ent_len = len(bert_tokenizer.tokenize(ent))
                start = len(masked_target_tokens)
                ent_idx = new_dict['sorted_node_list'].index(ent) #new_dict['nodes'].index(ent)
                positions[ent_idx] = list(range(start, start + ent_len))
                masked_target_tokens.extend(['[MASK]'] * ent_len)
                #new_target_tokens.extend(bert_tokenizer.tokenize(ent))
                last_is_extra = False
                order += 1
            else:
                #new_target_tokens.append(token)
                if not last_is_extra:
                    masked_target_tokens.append('extra_mask')   
                    last_is_extra = True

        positions = [p for pos in positions for p in pos]
        new_dict['positions'] = positions

        #print('teacher_cloze', new_dict['teacher_cloze_tokens'])
        #print('masked_target_tokens', masked_target_tokens)
        #print('positions', new_dict['positions'] )
        #print('split_nodes', new_dict['split_nodes'])
        
        #new_dict['description'] = new_target_tokens
        #new_dict['masked_description'] = masked_target_tokens

        assert len(new_dict['split_nodes']) == len(new_dict['positions'])

        
        all_node_ranges = []
        for pair in pairs:
            all_node_ranges.append(pair[0])
            all_node_ranges.append(pair[1])
        all_node_ranges = list(all_node_ranges)
        all_node_ranges.sort()
        #print(all_node_ranges)
        res = []
        for i in all_node_ranges:
            if i not in res:
                res.append(i)
        #print(res)
        prediction_tokens = []
        for idx, r in enumerate(res):
            prediction_tokens.extend(range(r[0], r[1]+1))
            prediction_tokens.append(-idx)

        new_dict['prediction_tokens'] = prediction_tokens
        # print(json.dumps(new_dict, ensure_ascii=False)

        #print(len(new_dict['split_nodes']))
        fout.write(json.dumps(new_dict, ensure_ascii=False) + "\n")
        
    fout.close()
