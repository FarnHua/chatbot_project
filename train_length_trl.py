import os
import numpy as np
import random
import json
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from os.path import join
import re
from argparse import ArgumentParser

# from Emo_detector.detect_emotion import re_emo_score, prepare_model
from transformers import  GPT2Tokenizer
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from chat_load import post_set
from lsp_model.optim import Adam
import tensorflow as tf
### add for trl
from ppo import PPOTrainer
from gpt2 import GPT2HeadWithValueModel, respond_to_batch
import time
###


import string
from tqdm import tqdm

import wandb

wandb.login()
torch.manual_seed(100)
emo_dict = {
                "<afraid>": [-0.12, 0.79], 
                "<angry>": [-0.42, 0.79], 
                "<annoyed>": [-0.44, 0.66], 
                "<anticipating>": [0.32, 0.06], # expectant
                "<anxious>":[-0.72, -0.8] , 
                    "<apprehensive>": [-0.77, -0.6], 
                "<ashamed>": [-0.45, -0.5], 
                    "<caring>": [0.25, -0.5], # not worried 
                "<confident>": [0.51, -0.2], 
                "<content>": [0.82, -0.55], 
                    "<devastated>": [-0.8, -0.5], 
                "<disappointed>": [-0.8, -0.03], 
                "<disgusted>": [-0.67, 0.49], 
                "<embarrassed>": [-0.32, -0.6], # caring
                "<excited>": [0.7, 0.72], 
                    "<faithful>": [0.6, 0.2], 
                    "<furious>": [-0.7, 0.85], 
                    "<grateful>": [0.8, -0.08], 
                "<guilty>": [-0.4, -0.43], # feel guilt
                "<hopeful>": [0.62, -0.3], 
                "<impressed>": [0.38, -0.07], 
                "<jealous>": [-0.08, 0.56], 
                "<joyful>": [0.85, 0.15], 
                    "<lonely>": [-0.2, -0.8], 
                    "<nostalgic>": [-0.5, -0.2],  # sentimental
                    # "prepared": , # confident
                    "<proud>": [0.45, 0.07],
                "<sad>": [-0.82, -0.4], 
                    # "sentimental": ,# nostalgic
                "<surprised>": [0.42, 0.79],  # astonished
                    # "terrified": , # afraid
                    "<trusting>": [0.3, 0.2]
            }

# detect_model, detect_processor, emotion_tokenizer = prepare_model()
# bad_word = ["4r5e", "5h1t", "5hit", "a55", "anal", "anus", "ar5e", "arrse", "arse", "ass", "ass-fucker", "asses", "assfucker", "assfukka", "asshole", "assholes", "asswhole", "a_s_s", "b!tch", "b00bs", "b17ch", "b1tch", "ballbag", "balls", "ballsack", "bastard", "beastial", "beastiality", "bellend", "bestial", "bestiality", "bi+ch", "biatch", "bitch", "bitcher", "bitchers", "bitches", "bitchin", "bitching", "bloody", "blow job", "blowjob", "blowjobs", "boiolas", "bollock", "bollok", "boner", "boob", "boobs", "booobs", "boooobs", "booooobs", "booooooobs", "breasts", "buceta", "bugger", "bum", "bunny fucker", "butt", "butthole", "buttmuch", "buttplug", "c0ck", "c0cksucker", "carpet muncher", "cawk", "chink", "cipa", "cl1t", "clit", "clitoris", "clits", "cnut", "cock", "cock-sucker", "cockface", "cockhead", "cockmunch", "cockmuncher", "cocks", "cocksuck", "cocksucked", "cocksucker", "cocksucking", "cocksucks", "cocksuka", "cocksukka", "cok", "cokmuncher", "coksucka", "coon", "cox", "crap", "cum", "cummer", "cumming", "cums", "cumshot", "cunilingus", "cunillingus", "cunnilingus", "cunt", "cuntlick", "cuntlicker", "cuntlicking", "cunts", "cyalis", "cyberfuc", "cyberfuck", "cyberfucked", "cyberfucker", "cyberfuckers", "cyberfucking", "d1ck", "damn", "dick", "dickhead", "dildo", "dildos", "dink", "dinks", "dirsa", "dlck", "dog-fucker", "doggin", "dogging", "donkeyribber", "doosh", "duche", "dyke", "ejaculate", "ejaculated", "ejaculates", "ejaculating", "ejaculatings", "ejaculation", "ejakulate", "f u c k", "f u c k e r", "f4nny", "fag", "fagging", "faggitt", "faggot", "faggs", "fagot", "fagots", "fags", "fanny", "fannyflaps", "fannyfucker", "fanyy", "fatass", "fcuk", "fcuker", "fcuking", "feck", "fecker", "felching", "fellate", "fellatio", "fingerfuck", "fingerfucked", "fingerfucker", "fingerfuckers", "fingerfucking", "fingerfucks", "fistfuck", "fistfucked", "fistfucker", "fistfuckers", "fistfucking", "fistfuckings", "fistfucks", "flange", "fook", "fooker", "fuck", "fucka", "fucked", "fucker", "fuckers", "fuckhead", "fuckheads", "fuckin", "fucking", "fuckings", "fuckingshitmotherfucker", "fuckme", "fucks", "fuckwhit", "fuckwit", "fudge packer", "fudgepacker", "fuk", "fuker", "fukker", "fukkin", "fuks", "fukwhit", "fukwit", "fux", "fux0r", "f_u_c_k", "gangbang", "gangbanged", "gangbangs", "gaylord", "gaysex", "goatse", "God", "god-dam", "god-damned", "goddamn", "goddamned", "hardcoresex", "hell", "heshe", "hoar", "hoare", "hoer", "homo", "hore", "horniest", "horny", "hotsex", "jack-off", "jackoff", "jap", "jerk-off", "jism", "jiz", "jizm", "jizz", "kawk", "knob", "knobead", "knobed", "knobend", "knobhead", "knobjocky", "knobjokey", "kock", "kondum", "kondums", "kum", "kummer", "kumming", "kums", "kunilingus", "l3i+ch", "l3itch", "labia", "lmfao", "lust", "lusting", "m0f0", "m0fo", "m45terbate", "ma5terb8", "ma5terbate", "masochist", "master-bate", "masterb8", "masterbat*", "masterbat3", "masterbate", "masterbation", "masterbations", "masturbate", "mo-fo", "mof0", "mofo", "mothafuck", "mothafucka", "mothafuckas", "mothafuckaz", "mothafucked", "mothafucker", "mothafuckers", "mothafuckin", "mothafucking", "mothafuckings", "mothafucks", "mother fucker", "motherfuck", "motherfucked", "motherfucker", "motherfuckers", "motherfuckin", "motherfucking", "motherfuckings", "motherfuckka", "motherfucks", "muff", "mutha", "muthafecker", "muthafuckker", "muther", "mutherfucker", "n1gga", "n1gger", "nazi", "nigg3r", "nigg4h", "nigga", "niggah", "niggas", "niggaz", "nigger", "niggers", "nob", "nob jokey", "nobhead", "nobjocky", "nobjokey", "numbnuts", "nutsack", "orgasim", "orgasims", "orgasm", "orgasms", "p0rn", "pawn", "pecker", "penis", "penisfucker", "phonesex", "phuck", "phuk", "phuked", "phuking", "phukked", "phukking", "phuks", "phuq", "pigfucker", "pimpis", "piss", "pissed", "pisser", "pissers", "pisses", "pissflaps", "pissin", "pissing", "pissoff", "poop", "porn", "porno", "pornography", "pornos", "prick", "pricks", "pron", "pube", "pusse", "pussi", "pussies", "pussy", "pussys", "rectum", "retard", "rimjaw", "rimming", "s hit", "s.o.b.", "sadist", "schlong", "screwing", "scroat", "scrote", "scrotum", "semen", "sex", "sh!+", "sh!t", "sh1t", "shag", "shagger", "shaggin", "shagging", "shemale", "shi+", "shit", "shitdick", "shite", "shited", "shitey", "shitfuck", "shitfull", "shithead", "shiting", "shitings", "shits", "shitted", "shitter", "shitters", "shitting", "shittings", "shitty", "skank", "slut", "sluts", "smegma", "smut", "snatch", "son-of-a-bitch", "spac", "spunk", "s_h_i_t", "t1tt1e5", "t1tties", "teets", "teez", "testical", "testicle", "tit", "titfuck", "tits", "titt", "tittie5", "tittiefucker", "titties", "tittyfuck", "tittywank", "titwank", "tosser", "turd", "tw4t", "twat", "twathead", "twatty", "twunt", "twunter", "v14gra", "v1gra", "vagina", "viagra", "vulva", "w00se", "wang", "wank", "wanker", "wanky", "whoar", "whore", "willies", "willy", "xrated", "xxx"]
# bad_dict = {}
# for w in bad_word:
#     bad_dict[w] = 1

word_dict = {}

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                commonWords_id: list = None):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
            commonWords_id(list, optional): use commonwords to initialize
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte  
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte, n_tokens, 
                            random_range, initialize_from_vocab, commonWords_id))
        ## (10,768)
        
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True,
                             commonWords_id: list = None):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        
        if initialize_from_vocab:
            if commonWords_id != None:
                return self.wte.weight[commonWords_id].clone().detach()
            else:
                return self.wte.weight[:n_tokens].clone().detach()

        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # input_embedding = self.wte(tokens[:, self.n_tokens:]) ## (1,4,768)
        if tokens.size()[1] > 1:
            input_embedding = self.wte(tokens[:, self.n_tokens:]) ## (1,4,768)
            learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1) #torch.Size([1, 20, 768])
            return torch.cat([learned_embedding, input_embedding], 1) #torch.Size([1, 24, 768])
        else:
            return self.wte(tokens)



def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
       # print(values.shape)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits
temperature = 1 #2.2
top_k = 50        #50
top_p = 0.95
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_response(model, sentences, tokenizer, first_input):
    with torch.no_grad():

        sentences = [tokenizer.encode(x) for x in sentences]
        t = []
        for i in range(len(sentences)):
            t_0 = [0 for i in range(len(list(first_input[i])))]
            t_1 = [1 for i in range(len(list(sentences[i]))-1)]
            sentences[i] = list(first_input[i]) + list(sentences[i])

            t.append(t_0[:] + t_1[:])
        mask= []


        sentences = [x[:-1] for x in sentences]
        for i in range(len(sentences)):
            mask.append([1 for x in range(len(sentences[i]))])
        eos = [tokenizer.encoder["<|endoftext|>"]]

        prev_input = pad_sequence([torch.LongTensor(x) for x in sentences], batch_first=True, padding_value=0).to(device_0)
        mask = pad_sequence([torch.LongTensor(x) for x in mask], batch_first=True, padding_value=0).to(device_0)
        #_, past = model(prev_input, past=None, attention_mask=mask)
        _, past, v = model(prev_input, past_key_values=None, attention_mask=mask)
        prev_input = torch.LongTensor([[eos] * len(sentences)]).to(device_0)
        temp_sentence = [[] for i in range(len(sentences))]
        for i in range(128):
            #prev_input, past = model(prev_input, past=past)
            prev_input, past, v = model(prev_input, past_key_values=past)
            # prev_input, past = output['logits'], output['past_key_values']
            prev_input = prev_input.squeeze(0).squeeze(1)
            prev_input = prev_input / 0.7
            prev_input = torch.softmax(prev_input, dim=-1)

            prev_input = torch.multinomial(prev_input, num_samples=1)

            if i == 0:
                for j in range(len(sentences)):
                    temp_sentence[j].append(prev_input[j].item())
                continue
            flag = 1
            
            for j in range(len(sentences)):
                if temp_sentence[j][-1] != eos[0]: 
                    flag = 0
                    temp_sentence[j].append(prev_input[j].item())
            if flag == 1: break
    return [[tokenizer.decode(x).replace('<|endoftext|>', '')] for x in temp_sentence]

table_data = []

def train(model_train, inputs_id, mask, model_bot, tokenizer, ll, args, batch_size, n_tokens, batch):
    
    inputs_id = inputs_id.to(device_0) ## 8*29
    
    
    eos = [tokenizer.encoder["<|endoftext|>"]] ## [50256]

    mask = mask.to(device_0)
    
    
    prev_input, past, v = model_train(inputs_id, past_key_values=None, attention_mask=mask)
    inputs_id = inputs_id.to(device_0)
    mask = mask.to(device_0)
    
    prev_input = torch.LongTensor([[eos] * inputs_id.shape[0]]).squeeze(0).to(device_0) # (8,1)



    ######### all (8,) 
    temp_sentence = [[] for i in range(inputs_id.shape[0])]
    
    append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device_0)
    mask = torch.cat((mask, append), 1) 

    for i in range(40): # 40 words

        prev_input = prev_input.to(device_0)
        logits, past, v = model_train(prev_input, past_key_values=past)
        
        
        
        mask = torch.cat((mask, append), 1)
        # print(logits.size()) ## (1, 8, 1, 50257)
        logits = logits.squeeze(0).squeeze(1)
        # print(logits.size()) ## (8, 50257)
        logits = logits / temperature
        logits = torch.softmax(logits, dim=-1)

        ## prevent empty sentence
        # if i == 0:
        #     for j in range(inputs_id.shape[0]):
        #         logits[j][tokenizer.sep_token_id] = 0


        prev_input = torch.multinomial(logits[:], num_samples=1) #(8,1)

        if i == 0:
            ## if first word of chatbot
            for j in range(inputs_id.shape[0]):
                temp_sentence[j].append(prev_input[j].item())
            continue ## jump to second words
        flag = 1 ## to ascertain whether all sentence complete
        
        for j in range(inputs_id.shape[0]):
            if temp_sentence[j][-1] != eos[0]: 
                flag = 0
                temp_sentence[j].append(prev_input[j].item())
        if flag == 1: break
    decode_temp_sentence = [tokenizer.decode(x) for x in temp_sentence]

    
    

    eos = [tokenizer.encoder["<|endoftext|>"]]
    first_input = list(inputs_id.cpu().detach().numpy())
    for j in range(inputs_id.shape[0]):
        l = ll[j]
        first_input[j] = first_input[j][n_tokens:n_tokens+l+1]
        first_input[j][-1] = eos[0]
    inter_response = []
    if 'gpt' in args.inter:
        inter_response.extend(make_response(model_bot, decode_temp_sentence, tokenizer, first_input))
    
    if batch % 40 == 0:
        # print(f'batch: {batch}')
        inpu_t = [tokenizer.decode(x[n_tokens:]) for x in inputs_id]
        my_table = wandb.Table(columns=['batch','input', 'chatbot', 'inter']) 
        for i in range(inputs_id.shape[0]):
            table_data.append([batch,inpu_t[i] ,decode_temp_sentence[i].replace('<|endoftext|>', ''), inter_response[i][0]])
        for t in table_data:
            my_table.add_data(*t)
        # print('************** save to table **********')
        wandb.log({'generation table': my_table}, commit=False)


#################################################################################
#                                                                               #
#                   Reward                                                      #
#                                                                               #
#################################################################################

    sent_input = []
    for j in range(inputs_id.shape[0]*len(args.inter)):
        l = ll[j%inputs_id.shape[0]]
        sent_input.append([tokenizer.decode(inputs_id[j%inputs_id.shape[0]][n_tokens:].tolist()), decode_temp_sentence[j%inputs_id.shape[0]], inter_response[j][0]])
        
        if j == 0:
            query = tokenizer.decode(inputs_id[j%inputs_id.shape[0]][n_tokens:].tolist()).replace('[SEP]', '').replace('[CLS]', '').replace(' ', '')
            response = decode_temp_sentence[j%inputs_id.shape[0]].replace('[SEP]', '').replace('[CLS]', '').replace(' ', '')
            inter = inter_response[j][0].replace('[SEP]', '').replace('[CLS]', '').replace(' ', '')

    temp_score = []
    for sens in sent_input:           
        sen = (sens[0] + sens[1] + sens[2]).replace('[SEP]', '').replace('[CLS]', '').replace(' ', '')
        temp_score.append(len(sens[2].split()))
    # test_len = [len(s) for s in temp_sentence]
    score = np.array(temp_score) / len(args.inter)
    return temp_sentence, torch.Tensor(score), query, response, inter, np.sum(score) / inputs_id.size()[0]

def main():
    parser = ArgumentParser()
    parser.add_argument("--emotion", type=str, default=None)
    parser.add_argument("--writer", type=str, default="")
    parser.add_argument("--save", type=str, default="model/save/")
    parser.add_argument("--model", type=str, default='model/turn')
    parser.add_argument("--ra", type=float, default=3)
    parser.add_argument("--inter", type=str, default="gpt", nargs='+', required=True)
    parser.add_argument("--n_tokens", type=int, default=10)
    parser.add_argument("--sw", type=str, default=None)
    parser.add_argument("--len", type=str, default=None)
    parser.add_argument("--initial", type=str, default='vocab')
    args = parser.parse_args()

    if not args.emotion and not args.sw and not args.len :
        print('Please enter emotion or specific word')
        assert(0)
    else:
        if args.sw : 
            reward = 'sw'
        elif args.emotion : 
            reward = 'emotion'
        else :
            reward = 'length'
        # reward = 'sw' if args.sw else 'emotion' 

    wandb.init(
      # Set the project where this run will be logged
      project="chatbot_softprompt", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"{args.save}",
      entity="chatbot_ntu"
      )
    # Track hyperparameters and run metadata
    wandb.config.update(args)
    wandb.config.update({"seed":100, 
        'init_from_vocab': True if args.initial == 'vocab' else False})

    if not os.path.exists(f'./model/save/{args.save}'):
        os.makedirs(f'./model/save/{args.save}')

    config = wandb.config
    os.makedirs('model/' + args.model, exist_ok=True)
    #test
    

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    model_train = GPT2HeadWithValueModel(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    sep = tokenizer.sep_token_id

    ### setting  softprompt
    n_tokens = args.n_tokens
    initialize_from_vocab = config.init_from_vocab

    ## init from commonWords
    if initialize_from_vocab:
      random.seed(config.seed)
      with open("commonWords.txt") as f:
          words = f.read().splitlines()
          # print(args.n_tokens)
          random_num = random.sample(range(0, 3000), n_tokens)
          commonWords = words[random_num[0]]
          for i in range(1, n_tokens):
            commonWords = commonWords + ' ' + words[random_num[i]]
          commonWords_id = tokenizer.encode(commonWords)
          commonWords_id = commonWords_id[:n_tokens]
    else:
      commonWords_id = None

    initialize_from_vocab = config.init_from_vocab
    s_wte = SoftEmbedding(model_train.transformer.get_input_embeddings(), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=initialize_from_vocab,
                      commonWords_id=commonWords_id)

    model_train.transformer.set_input_embeddings(s_wte)

    parameters = list(model_train.transformer.parameters())
    # parameters_check = list(model_train.parameters())

    for x in parameters[1:]:
        x.requires_grad = False
    ###


    if 'gpt' in args.inter:
        model_bot = GPT2HeadWithValueModel(args.model)
        model_bot.to(device_0)
        model_bot.eval()
    

    ### init ppo trainer

    ppo_config = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "steps": 51200,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "n_tokens": n_tokens
    }
    ppo_trainer = PPOTrainer(model_train, model_bot, **ppo_config)
    wandb.config.update(ppo_config)
    
    wandb.config.update({'total_update_param': sum(p.numel() for p in model_train.parameters() if p.requires_grad)})
    ##

    model_train.to(device_0)
    

        
    
    batch_size = ppo_config['batch_size']
    fbs = ppo_config['forward_batch_size']
    post = post_set('data/train_raw.tsv', tokenizer, n_tokens)
    train_dataloader = DataLoader(post, batch_size=int(config['batch_size']/fbs), shuffle=True, num_workers=2)

    batch = 0
    temp_score = 0
    loss = 0
   
    test_score = 0
    for global_step in range(1):
        model_train.train()
        


        batches = []
        querys = []
        responses = []
        inters = []
        record_rewards = []

        for epoch in range(1):
            torch.cuda.empty_cache()
            logs = dict()
            game_data = dict()
            timing = dict()
            t0 = time.time()

            query_tensors = []
            response_tensors = []
            rewards = []
            i = 0
            avg_r = 0
            
           
            for inputs_id, mask, ll in tqdm(train_dataloader):
                
                response_ids, score, query, response, inter, avg_score = train(model_train, inputs_id, mask, model_bot, tokenizer, ll, args, fbs, n_tokens, batch)
                # query_tensors.append(torch.cat((inputs_id, torch.LongTensor([[sep] for x in range(inputs_id.shape[0])])), axis=-1))
                avg_r += avg_score
                query_tensors.append(inputs_id)
                for response_id in response_ids:
                    response_tensors.append(response_id[:, -1])

                    print("show_response:   ====================\n")
                    print(response_ids[:, -1])
                    assert(0)
                rewards.append(score)
                if i == 0:
                    querys.append(query)
                    responses.append(response)
                    inters.append(inter)
                    batches.append(batch)
                    record_rewards.append(score[0])
                    i = 1

                if (batch + 1) % 16 == 0:

                    game_data['batch'] = batch
                    game_data['query'] = querys
                    game_data['response'] = responses
                    game_data['inter'] = inters
                    game_data['reward'] = record_rewards

                    query_tensors = torch.cat(query_tensors).to(device_0)
                    response_tensors = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in response_tensors], padding='post', value=0)).to(device_0)
                    rewards = torch.cat(rewards).to(device_0)
                    stats, avg_loss = ppo_trainer.step(query_tensors, response_tensors, rewards)            
                    
                    # timing['time/epoch'] = time.time()-t0
                    # table_rows = [list(r) for r in zip(game_data['batch'], game_data['query'], game_data['response'], game_data['inter'], game_data['reward'])]
                    # logs.update({'game_log':wandb.Table(
                    #     columns=['epoch', 'query', 'response', 'inter', 'reward'],
                    #     rows=table_rows)})
                    # logs.update(timing)
                    # logs.update(stats)
                    # logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
                    # logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
                    # logs['env/reward_dist'] = rewards.cpu().numpy()
                    # wandb.log(logs)
                    wandb.log({"reward": avg_r / 16, "loss": avg_loss})


                    torch.cuda.empty_cache()
                    logs = dict()
                    game_data = dict()
                    timing = dict()
                    t0 = time.time()

                    query_tensors = []
                    response_tensors = []
                    rewards = []
                    i = 0
                    avg_r = 0
                
                if (batch + 1) % 100 == 0:
                    name = 'transformer.wte.learned_embedding' 
                    # idx = random.randint(1, len(parameters_check) - 1)
                    # param = list(model_train.parameters())
                    # check_valid(param[idx], parameters_check[idx])
                    torch.save(
                        {name: (model_train.transformer.state_dict()[name])},
                        join(f'model/save/{args.save}',
                                f'{args.save}_swe-{batch}.pkl'))
                    
                    torch.save(
                        {name: (model_train.v_head.state_dict())},
                        join(f'model/save/{args.save}',
                                f'{args.save}_value-{batch}.pkl')
                    )
                batch += 1


    wandb.finish()

if __name__ == "__main__":
    main()