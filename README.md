This project is modified from https://github.com/jacksukk/Chatbot-Project


We adapt the DialoGPT(https://github.com/microsoft/DialoGPT) model to be our chatbot model.
## Get started
### 1. example.ipynb

### 2. Clone the repository
```
git clone https://github.com/FarnHua/chatbot_project.git
```

#### Corpus
https://github.com/facebookresearch/EmpatheticDialogues.git

#### Train
```
python train_c.py --emotion <emotion> --writer <tensorboard writer>  
--save <save path> --model <pretrained model> --ra <ratio between 2 loss> 
--inter <interlocutor you want to interact> --sw <specific word>
```

#### Test
```
python test_c.py --model <model to test (.pkl)> --filename <output file> \ 
--inter <interlocutor you want to interact> \ 
--base_model <gpt2 or dialogpt> --len <length of prefix>
```

#### Evaluation 
```
python eval.py --filename <output file from test_c.py> \
--model_name <model_to_test(.pkl)> --len <length of prefix> \ 
--outputfilename <output csv file>
```

#### Emotion Detector
please download the following link and put it in './' directory.
```
https://drive.google.com/file/d/1FZu2HIadORIvGD5nJAIOG6NjgtXrNXrz/view?usp=sharing
```
