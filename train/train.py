BUCKET = "gs://chinesekg2text-bucket"
import os
import torch
import time
import numpy as np
import pickle
from torch import nn
from logging import getLogger
from train.data import Vocab, NLP, S2SDataset
from train.utils import build_optimizer, init_seed, init_logger, init_device, read_configuration, collate_fn_graph_text, \
    format_time
from train.module import GraphEncoder, GraphReconstructor, GraphPointer
#from transformers import BartTokenizer, BartForConditionalGeneration, BertModel, BertTokenizer
#from transformers import AutoTokenizer, AutoModelForMaskedLM, BertLMHeadModel
from transformers import BertTokenizer, T5EncoderModel, T5ForConditionalGeneration, Text2TextGenerationPipeline
from torch.utils.data import Dataset, DataLoader
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer, file_utils
import argparse
import matplotlib.pyplot as plt



def compute_kd_loss(node_embeddings, desc_embeddings, node_masks, kd_masks):
    #print(node_embeddings.size())
    #print(desc_embeddings.size())
    assert node_embeddings.size() == desc_embeddings.size()
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(node_embeddings, desc_embeddings)
    loss = loss.mean(dim=-1)
    masks = node_masks * kd_masks
    loss = loss.masked_select(masks).mean()
    return loss


def compute_ce_loss(logits, labels, masks):
    ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.reshape_as(labels)
    loss = loss.masked_select(masks).mean()
    return loss

def plot_grad(model, model_name):
    layers = []
    ave_grads = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if(p.grad is None):
                ave_grads.append(0)
            else:
                #if(model_name=="plm"):
                    #print(n, p.grad.abs().mean())
                #ave_grads.append(p.grad.abs().mean())
                ave_grads.append(torch.linalg.norm(p.grad))

    plt.figure(figsize=(18, 4))
    plt.plot(ave_grads, "o-", alpha=0.3)
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow for "+ model_name)
    plt.grid(True)
    plt.show()

def print_norm(model, name):
    print("--------------{}-------------".format(name))
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            if(p.grad is None):
                print(n, "None")
            else:
                print(n, torch.linalg.norm(p.grad))
    print("--------------end {}-------------".format(name))
def print_var_grad(grad):
    print("--------------var_grad-------------")
    print(grad[0, :, 0])
    print(grad[1, :, 0])
    print("--------------end var_grad-------------")

def print_var_grad_norm(grad):
    print("--------------var_grad_norm-------------")
    print(torch.linalg.norm(grad))
    print("--------------end var_norm-------------")

def print_extra_loss(grad):
    print("-----------extra_loss grad--------------")
    print(grad)
    print("-----------end extra_loss grad--------------")

def run_train_batch(config, batch, teacher, student, plm, reconstructor, copyer,
                    plm_optimizer, external_optimizer, scaler, extra_input_ids, device):
    #print(batch)
    # S2SDataset: data = {"nodes": input_nodes, "edges": input_edges, "types": input_types, "outputs": output_ids,
    # "pointer": pointer, "pairs": input_pairs, "relations": relations, "positions": positions,
    # "descriptions": descriptions}
    # utils: return nodes, edges, types, node_masks, descriptions, description_masks, positions, relations, pairs, pair_masks, \
    #       outputs, output_masks, pointer, pointer_masks
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
        recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks, \
        predictions, prediction_masks = batch
    #------------------try to solve nan?
    #print(nodes)
    #assert not torch.any(torch.isnan(nodes))
    #flat_edges = [item for sublist in edges for pairs in sublist for item in pairs]
    #assert not np.any(np.isnan(np.asarray(flat_edges)))
    #flat_types = [item for sublist in types for item in sublist]
    #assert not np.any(np.isnan(np.asarray(flat_types)))
    #assert not torch.any(torch.isnan(kd_description))
    #assert not torch.any(torch.isnan(kd_positions))
    #assert not torch.any(torch.isnan(recon_relations))
    #assert not torch.any(torch.isnan(recon_positions))
    #assert not torch.any(torch.isnan(gen_outputs))
    #assert not torch.any(torch.isnan(pointer))

    #assert not torch.sum(node_masks)==0
    #assert not torch.sum(kd_description_masks)==0
    #assert not torch.sum(recon_masks)==0
    #assert not torch.sum(gen_masks)==0
    #assert not torch.sum(pointer_masks)==0

    kd_description = kd_description.to(device)
    kd_description_masks = kd_description_masks.to(device)
    # print("kd_description size", kd_description.size())
    #print("kd_description", kd_description)
    
    with torch.cuda.amp.autocast(enabled=config['use_amp']):
        output_dict = teacher(input_ids=kd_description,
                          attention_mask=kd_description_masks,
                          output_hidden_states=True,
                          return_dict=True)

    encoder_last_hidden_state = output_dict["hidden_states"][-1] #TODO I assume this is the last encoder state??
    positions = kd_positions.unsqueeze(-1).expand(-1, -1, encoder_last_hidden_state.size(-1)).to(device)
    #Hidden-states of the model at the output of each layer plus the initial embedding outputs
    
    teacher_embeddings = encoder_last_hidden_state
    teacher_embeddings = teacher_embeddings.detach()

    teacher_nodes_embeddings = torch.gather(encoder_last_hidden_state, dim=1, index=positions)
    teacher_nodes_embeddings = teacher_nodes_embeddings.detach()
    # print("teacher_embeddings size", teacher_embeddings.size())

    nodes = nodes.to(device)
    # print("nodes size", nodes.size())
    #print("nodes", nodes)
    with torch.cuda.amp.autocast(enabled=config['use_amp']):
        student_embeddings = student(nodes, edges, types)
    # student_embeddings.register_hook(print_var_grad_norm)

    node_masks = node_masks.to(device)
    kd_masks = torch.ne(kd_positions, 0).to(device)
    kd_loss = compute_kd_loss(student_embeddings, teacher_nodes_embeddings, node_masks, kd_masks)

    gen_outputs = gen_outputs.to(device)
    # print("gen_outputs size", gen_outputs.size())
    gen_masks = gen_masks.to(device)
    with torch.cuda.amp.autocast(enabled=config['use_amp']):
        output_dict = plm(input_ids=None,
                  inputs_embeds = teacher_embeddings,
                  attention_mask = kd_description_masks,#node_masks
                  
                  decoder_input_ids=gen_outputs[:, :-1],
                  decoder_attention_mask=gen_masks[:, :-1],
                  
                  output_hidden_states=True,
                  labels=gen_outputs[:, 1:].contiguous(),
                  return_dict=True)
    gen_loss = output_dict["loss"]

    #logits = output_dict['logits']
    #extra_logits = logits[:, :, extra_input_ids]
    #extra_logits.register_hook(print)
    extra_loss = torch.tensor(0.0) #(extra_logits-(extra_logits.detach()-1)).mean()
    #extra_loss.register_hook(print_extra_loss)

    decoder_input_embeddings = plm.get_input_embeddings()(gen_outputs[:, :-1])
    decoder_output_hiddens = output_dict["decoder_hidden_states"][-1]
    # decoder_output_hiddens.register_hook(print_var_grad_norm)
    decoder_output_hiddens = torch.nn.functional.normalize(decoder_output_hiddens)
    
    # assert not torch.any(torch.isnan(decoder_output_hiddens))
    # print("decoder_input_embeddings", decoder_input_embeddings[0, 0, 0])
    # print("decoder_output_hiddens", decoder_output_hiddens[0, 0, 0])

    # what to do with copying now??
    #pointer = pointer.to(device)
    #pointer_masks = pointer_masks.to(device)
    #with torch.cuda.amp.autocast(enabled=config['use_amp']):
    #    copy_prob = copyer(decoder_input_embeddings, decoder_output_hiddens, pointer[:, 1:])
    ## copy_prob.register_hook(print_var_grad_norm)
    #copy_loss = copy_prob.masked_select(pointer_masks[:, 1:]).mean()
    ##copy_loss.register_hook(print)
    copy_loss = torch.tensor(0.0)

    recon_positions = recon_positions.to(device)
    recon_relations = recon_relations.to(device)
    recon_masks = recon_masks.to(device)
    with torch.cuda.amp.autocast(enabled=config['use_amp']):
        rec_logits = reconstructor(recon_positions, output_dict["encoder_hidden_states"][-1])
    # rec_logits.register_hook(print_var_grad_norm)
    rec_loss = compute_ce_loss(rec_logits, recon_relations, recon_masks)

    loss = gen_loss*config["gen_weight"] + rec_loss*config["rec_weight"] + kd_loss*config["kd_weight"] + extra_loss*config["extra_weight"] + copy_loss*config["cp_weight"] #
    # 
    plm_optimizer.zero_grad()
    external_optimizer.zero_grad()
    # not needed with following line? loss.backward()
    scaler.scale(loss).backward()
    #------------------try to solve nan?
    # not needed with scalar step external_optimizer.step()
    # not needed with scalar step plm_optimizer.step()
    torch.nn.utils.clip_grad_norm_(plm.parameters(), 1)
    # torch.nn.utils.clip_grad_norm_(copyer.parameters(), 1)
    # plot_grad(teacher, "teacher")
    # plot_grad(student, "student")
    # plot_grad(plm, "plm")
    # plot_grad(reconstructor, "reconstructor")
    # plot_grad(copyer, "copyer")
    # print_norm(student, "student")
    # print_norm(plm, "plm")
    # print_norm(reconstructor, "reconstructor")
    # print_norm(copyer, "copyer")

    scaler.step(plm_optimizer)
    scaler.step(external_optimizer)
    scaler.update()
    #print("add update")


    return gen_loss.item(), rec_loss.item(), kd_loss.item(), copy_loss.item(), extra_loss.item()


def run_eval_batch(config, batch, teacher, student, plm, reconstructor, copyer, extra_input_ids, device):
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
        recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks, \
                predictions, prediction_masks = batch

    kd_description = kd_description.to(device)
    kd_description_masks = kd_description_masks.to(device)
    output_dict = teacher(input_ids=kd_description,
                          attention_mask=kd_description_masks,
                          output_hidden_states=True,
                          return_dict=True)
    encoder_last_hidden_state = output_dict["hidden_states"][-1]
    positions = kd_positions.unsqueeze(-1).expand(-1, -1, encoder_last_hidden_state.size(-1)).to(device)
    teacher_embeddings = torch.gather(encoder_last_hidden_state, dim=1, index=positions)
    teacher_embeddings = teacher_embeddings.detach()

    nodes = nodes.to(device)
    student_embeddings = student(nodes, edges, types)

    node_masks = node_masks.to(device)#node_masks.to(device)
    kd_masks = torch.ne(kd_positions, 0).to(device)
    kd_loss = compute_kd_loss(student_embeddings, teacher_embeddings, node_masks, kd_masks)

    extra_embeddings = plm.get_input_embeddings()(extra_input_ids)
    extra_embeddings = extra_embeddings.to(device)

    prediction_masks = prediction_masks.to(device)
    student_inserted_embeddings = []
    #print(predictions)
    for batch_idx, student_embedding in enumerate(student_embeddings):
        student_embedding_list = []
        for pidx in predictions[batch_idx]:
            if(pidx>0):
                student_embedding_list.append(student_embeddings[batch_idx, pidx])
            else:
                student_embedding_list.append(extra_embeddings[-pidx])
        student_embedding_stacked = torch.stack(student_embedding_list, 0)
        student_inserted_embeddings.append(student_embedding_stacked)
    student_inserted_embeddings = torch.stack(student_inserted_embeddings, 0)
    student_inserted_embeddings = student_inserted_embeddings.to(device)

    gen_outputs = gen_outputs.to(device)
    gen_masks = gen_masks.to(device)
    output_dict = plm(input_ids=None,
                      inputs_embeds=student_inserted_embeddings,
                      attention_mask=prediction_masks,
                      decoder_input_ids=gen_outputs[:, :-1],
                      decoder_attention_mask=gen_masks[:, :-1],
                      output_hidden_states=True,
                      labels=gen_outputs[:, 1:].contiguous(),
                      return_dict=True)
    gen_loss = output_dict["loss"]
    
    logits = output_dict['logits']
    extra_logits = logits[:, :, extra_input_ids]
    extra_loss = (extra_logits-(extra_logits.detach()-1)).mean()

    decoder_input_embeddings = plm.get_input_embeddings()(gen_outputs[:, :-1])
    decoder_output_hiddens = output_dict["decoder_hidden_states"][-1]
    #pointer = pointer.to(device)
    #pointer_masks = pointer_masks.to(device)
    #copy_prob = copyer(decoder_input_embeddings, decoder_output_hiddens, pointer[:, 1:])
    #copy_loss = copy_prob.masked_select(pointer_masks[:, 1:]).mean()
    copy_loss = torch.tensor(0.0)

    recon_positions = recon_positions.to(device)
    recon_relations = recon_relations.to(device)
    recon_masks = recon_masks.to(device)
    rec_logits = reconstructor(recon_positions, output_dict["encoder_hidden_states"][-1])
    rec_loss = compute_ce_loss(rec_logits, recon_relations, recon_masks)

    return gen_loss.item(), rec_loss.item(), kd_loss.item(), copy_loss.item(), extra_loss.item()


def train(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"], mode = config['mode'])
    vocabs["relation"] = Vocab(config["relation_vocab"], mode = config['mode'])

    logger.info("Build Teacher Model.")
    #teacher = BartForConditionalGeneration.from_pretrained(config["teacher_dir"])
    teacher = T5EncoderModel.from_pretrained(config["teacher_dir"])
    teacher.requires_grad = False
    for para in teacher.parameters():
        para.requires_grad = False
    teacher.to(device)
    #teacher.half()

    logger.info("Build Student Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),
                           config["gnn_layers"], config["embedding_size"], config["node_embedding"])
    student.to(device)
    #student.half()

    logger.info("Build PLM Model.")
    #bart_tokenizer = BartTokenizer.from_pretrained(config["plm_dir"])
    bert_tokenizer = BertTokenizer.from_pretrained(config["plm_dir"])
    bert_tokenizer.bos_token="<s>"
    bert_tokenizer.eos_token="</s>"
    bert_tokenizer.mask_token="[MASK]"

    extra_tokens = ["extra"+str(x) for x in range(100)]
    extra_input_ids = bert_tokenizer(extra_tokens, return_tensors="pt").input_ids[:, 1]
    extra_input_ids = extra_input_ids.to(device)
    #plm = BartForConditionalGeneration.from_pretrained(config["plm_dir"])
    plm = T5ForConditionalGeneration.from_pretrained(config["plm_dir"])
    plm.to(device)
    #plm.half()

    logger.info("Build Reconstructor Model.")
    reconstructor = GraphReconstructor(vocabs["relation"].size(), config["hidden_size"])
    reconstructor.to(device)
    #reconstructor.half()

    logger.info("Build Copy Model.")
    copyer = GraphPointer(config["embedding_size"], config["hidden_size"])
    copyer.to(device)
    #copyer.half()

    plm_parameters = [p for p in plm.parameters() if p.requires_grad]
    plm_optimizer = build_optimizer(plm_parameters, config["plm_learner"], config["plm_lr"], config)

    external_parameters = []
    for p in student.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    for p in reconstructor.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    for p in copyer.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    external_optimizer = build_optimizer(external_parameters, config["external_learner"], config["external_lr"], config)

    logger.info("Create training dataset.")
    train_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bert_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples=config["num_samples"], usage="train", mode=config["mode"]),
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)

    logger.info("Create validation dataset.")
    valid_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bert_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples="all", usage="valid", mode=config["mode"]),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)

    kd_losses = []
    gen_losses = []
    copy_losses = []
    rec_losses = []
    best_gen_loss = None
    #new
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
    torch.autograd.set_detect_anomaly(False)

    for epoch_idx in range(config["start_epoch"], config["epochs"]):
        with torch.cuda.amp.autocast(enabled=config['use_amp']):
            teacher.train()
            student.train()
            plm.train()
            reconstructor.train()
            copyer.train()
        train_gen_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            #new
            
            gen_loss, rec_loss, kd_loss, copy_loss, extra_loss = run_train_batch(config, batch, teacher, student, plm, reconstructor,
                                                                     copyer, plm_optimizer, external_optimizer, scaler, extra_input_ids, device)

            logger.info("Epoch {} batch {}: KD loss {}, Gen loss {} Rec loss {} Copy loss {} Extra loss {}.".format(epoch_idx,
                                                                                                      batch_idx,
                                                                                                      kd_loss,
                                                                                                      gen_loss,
                                                                                                      rec_loss,
                                                                                                      copy_loss,
                                                                                                      extra_loss))

            train_gen_loss += gen_loss
            kd_losses.append(kd_loss)
            gen_losses.append(gen_loss)
            rec_losses.append(rec_loss)
            copy_losses.append(copy_loss)

        train_gen_loss /= len(train_dataloader)
        train_ppl = np.exp(train_gen_loss)
        training_time = format_time(time.time() - t0)
        logger.info("\n\nEpoch {}: training generation loss {} perplexity {} time {}.\n".format(epoch_idx,
                                                                                                train_gen_loss,
                                                                                                train_ppl,
                                                                                                training_time))

        with torch.no_grad():
            teacher.eval()
            student.eval()
            plm.eval()
            reconstructor.eval()
            copyer.eval()
            valid_gen_loss = 0
            t0 = time.time()
            for batch in valid_dataloader:
                gen_loss, rec_loss, kd_loss, copy_loss, extra_loss = run_eval_batch(config, batch, teacher, student, plm,
                                                                        reconstructor, copyer, extra_input_ids, device)
                valid_gen_loss += gen_loss

            valid_gen_loss /= len(valid_dataloader)
            valid_ppl = np.exp(valid_gen_loss)
            valid_time = format_time(time.time() - t0)
            logger.info("\n\nEpoch {}: validation generation loss {} perplexity {} time {}.\n".format(epoch_idx,
                                                                                                      valid_gen_loss,
                                                                                                      valid_ppl,
                                                                                                      valid_time))

        if best_gen_loss is None or valid_gen_loss <= best_gen_loss:
            output_dir = '{}-{}'.format(config["dataset"], config["num_samples"])
            saved_path = os.path.join(config["model_save_path"], output_dir)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            # save pretrained language model
            model_to_save = plm.module if hasattr(plm, 'module') else plm
            model_to_save.save_pretrained(saved_path)
            bert_tokenizer.save_pretrained(saved_path)

            # save knowledge adapter
            torch.save(config, os.path.join(saved_path, 'training_configurations.bin'))
            torch.save({"student": student.state_dict(),
                        "reconstructor": reconstructor.state_dict(),
                        "copyer": copyer.state_dict()},
                       os.path.join(saved_path, 'external.bin'))
            logger.info("Save student, reconstructor, copyer and fine-tuned PLM model into {}.".format(saved_path))

            best_gen_loss = valid_gen_loss


def test(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"], config['mode'])
    vocabs["relation"] = Vocab(config["relation_vocab"], config['mode'])

    logger.info("Build Teacher Model.")
    #teacher = BartForConditionalGeneration.from_pretrained(config["teacher_dir"])
    teacher = T5EncoderModel.from_pretrained(config["teacher_dir"])
    teacher.requires_grad = False
    for para in teacher.parameters():
        para.requires_grad = False
    teacher.to(device)

    logger.info("Build Student Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),
                           config["gnn_layers"], config["embedding_size"], config["node_embedding"])
    student.load_state_dict(torch.load(config["external_model"])["student"])
    student.to(device)

    logger.info("Build PLM Model.")
    bert_tokenizer = BertTokenizer.from_pretrained(config["plm_dir"])#config["fine_tuned_plm_dir"]
    bert_tokenizer.bos_token="<s>"
    bert_tokenizer.eos_token="</s>"
    bert_tokenizer.mask_token="[MASK]"

    plm = T5ForConditionalGeneration.from_pretrained(config["plm_dir"])#config["fine_tuned_plm_dir"]
    plm.to(device)

    logger.info("Create testing dataset.")
    test_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bert_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples="all", usage="test", mode=config['mode']),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)

    student.eval()
    teacher.eval()
    plm.eval()
    idx = 0
    teacher_text = []
    generated_text = []
    reference_text = []

    extra_tokens = ["extra"+str(x) for x in range(100)]
    extra_input_ids = bert_tokenizer(extra_tokens, return_tensors="pt").input_ids[:, 1]
    extra_input_ids = extra_input_ids.to(device)
    extra_embeddings = plm.get_input_embeddings()(extra_input_ids)
    extra_embeddings = extra_embeddings.to(device)

    with torch.no_grad():
        for batch in test_dataloader:
            nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
                recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks, \
                predictions, prediction_masks = batch

            #-----------test with teacher--------------------
            #kd_description = torch.ones_like(kd_description)
            kd_description = kd_description.to(device)
            #kd_description_masks = torch.ones_like(kd_description_masks)
            kd_description_masks = kd_description_masks.to(device)

            print(kd_description.size())
            print(kd_description_masks.size())
            
            print(kd_description)
            print(kd_description_masks)
            
            
            output_dict = teacher(input_ids=kd_description,
                                attention_mask=kd_description_masks,
                                output_hidden_states=True,
                                return_dict=True)

            encoder_last_hidden_state = output_dict["hidden_states"][-1] #TODO I assume this is the last encoder state??
            teacher_embeddings = encoder_last_hidden_state.detach()
            #-----------------end test with teacher-------------------

            nodes = nodes.to(device)
            student_embeddings = student(nodes, edges, types)

            node_masks = prediction_masks.to(device)#node_masks.to(device)

            student_inserted_embeddings = []
            #print(predictions)
            for batch_idx, student_embedding in enumerate(student_embeddings):
                student_embedding_list = []
                for pidx in predictions[batch_idx]:
                    if(pidx>0):
                        student_embedding_list.append(student_embeddings[batch_idx, pidx])
                    else:
                        student_embedding_list.append(extra_embeddings[-pidx])
                student_embedding_stacked = torch.stack(student_embedding_list, 0)
                student_inserted_embeddings.append(student_embedding_stacked)
            student_inserted_embeddings = torch.stack(student_inserted_embeddings, 0)
            student_inserted_embeddings = student_inserted_embeddings.to(device)
            
            student_output_dict = file_utils.ModelOutput()
            setattr(student_output_dict , 'last_hidden_state', student_inserted_embeddings) 
            model_kwargs = {
                "encoder_outputs": student_output_dict #plm.get_encoder()(
                    #inputs_embeds=output_dict["hidden_states"][-1], #teacher_embeddings, #student_inserted_embeddings
                    #return_dict=True)
            }
            
            generated_ids = plm.generate(input_ids=None,
                                         attention_mask=node_masks,
                                         num_beams=4,
                                         max_length=config["max_seq_length"],
                                         early_stopping=True,**model_kwargs
                                        )
            #print(generated_ids.size())
            teacher_descriptions = bert_tokenizer.batch_decode(kd_description, skip_special_tokens=True)
            teacher_text.extend(teacher_descriptions)
            generated = bert_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated)
            reference = bert_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            generated_text.extend(generated)
            reference_text.extend(reference)

            idx += 1
            logger.info("Finish {}-th example.".format(idx))

    assert len(generated_text) == len(reference_text)
    saved_file = "{}-{}.res".format(config["dataset"], config["num_samples"])
    saved_file_path = os.path.join(config["output_dir"], saved_file)
    fout = open(saved_file_path, "w")
    for i in range(len(generated_text)):
        fout.write("Teacher text: " + teacher_text[i].strip() + "\n")
        fout.write("Generated text: " + generated_text[i].strip() + "\n")
        fout.write("Reference text: " + reference_text[i].strip() + "\n")
        fout.write("\n")
    fout.close()


def main():
    print("updated train code")
    parser = argparse.ArgumentParser()

    #config = read_configuration("config.yaml")
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--use_gpu', type=bool, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--state', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_samples', type=str, required=True)
    parser.add_argument('--reproducibility', type=bool, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--use_amp', type=bool, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--node_vocab', type=str, required=True)
    parser.add_argument('--relation_vocab', type=str, required=True)
    parser.add_argument('--node_embedding', type=str, required=True)
    parser.add_argument('--colab_data_dir', type=str, required=True)
    parser.add_argument('--colab_node_vocab', type=str, required=True)
    parser.add_argument('--colab_relation_vocab', type=str, required=True)
    parser.add_argument('--colab_node_embedding', type=str, required=True)
    parser.add_argument('--teacher_dir', type=str, required=True)
    parser.add_argument('--plm_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--colab_model_save_path', type=str, required=True)
    parser.add_argument('--vertex_model_save_path', type=str, required=True)
    parser.add_argument('--start_epoch', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--plm_learner', type=str, required=True)
    parser.add_argument('--plm_lr', type=float, required=True)
    parser.add_argument('--external_learner', type=str, required=True)
    parser.add_argument('--external_lr', type=float, required=True)
    parser.add_argument('--rec_weight', type=float, required=True)
    parser.add_argument('--kd_weight', type=float, required=True)
    parser.add_argument('--cp_weight', type=float, required=True)
    parser.add_argument('--extra_weight', type=float, required=True)
    parser.add_argument('--gen_weight', type=float, required=True)
    parser.add_argument('--gnn_layers', type=int, required=True)
    parser.add_argument('--embedding_size', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--eval_batch_size', type=int, required=True)
    parser.add_argument('--external_model', type=str, required=True)
    parser.add_argument('--fine_tuned_plm_dir', type=str, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--max_seq_length', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    config={}
    config['gpu_id']=args.gpu_id
    config['use_gpu']=args.use_gpu
    config['seed']=args.seed
    config['state']=args.state
    config['dataset']=args.dataset
    config['num_samples']=args.num_samples
    config['reproducibility']=args.reproducibility
    config['mode']=args.mode
    config['use_amp']=args.use_amp
    config['data_dir']=args.data_dir
    config['node_vocab']=args.node_vocab
    config['relation_vocab']=args.relation_vocab
    config['node_embedding']=args.node_embedding
    config['colab_data_dir']=args.colab_data_dir
    config['colab_node_vocab']=args.colab_node_vocab
    config['colab_relation_vocab']=args.colab_relation_vocab
    config['colab_node_embedding']=args.colab_node_embedding
    config['teacher_dir']=args.teacher_dir
    config['plm_dir']=args.plm_dir
    config['log_dir']=args.log_dir
    config['model_save_path']=args.model_save_path
    config['colab_model_save_path']=args.colab_model_save_path
    config['vertex_model_save_path']=args.vertex_model_save_path
    config['start_epoch']=args.start_epoch
    config['epochs']=args.epochs
    config['train_batch_size']=args.train_batch_size
    config['plm_learner']=args.plm_learner
    config['plm_lr']=args.plm_lr
    config['external_learner']=args.external_learner
    config['external_lr']=args.external_lr
    config['rec_weight']=args.rec_weight
    config['kd_weight']=args.kd_weight
    config['cp_weight']=args.cp_weight
    config['extra_weight']=args.extra_weight
    config['gen_weight']=args.gen_weight
    config['gnn_layers']=args.gnn_layers
    config['embedding_size']=args.embedding_size
    config['hidden_size']=args.hidden_size
    config['eval_batch_size']=args.eval_batch_size
    config['external_model']=args.external_model
    config['fine_tuned_plm_dir']=args.fine_tuned_plm_dir
    config['test_batch_size']=args.test_batch_size
    config['max_seq_length']=args.max_seq_length
    config['output_dir']=args.output_dir

    if config["mode"] == "train":
        train(config)
    if config['mode'] == "train_colab":
        config["data_dir"] = config["colab_data_dir"] 
        config['node_vocab'] = config['colab_node_vocab']
        config['relation_vocab'] = config['colab_relation_vocab']
        config['node_embedding'] = config['colab_node_embedding']
        config['model_save_path'] = config['colab_model_save_path']
        train(config)
    if config['mode'] == "train_vertex":
        config['model_save_path'] = config['vertex_model_save_path']
        train(config)
    else:
        test(config)


if __name__ == '__main__':
    main()
