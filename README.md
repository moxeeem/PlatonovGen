# 📃 PlatonovGen
![img](https://github.com/moxeeem/PlatonovGen/blob/main/img.jpg)

## Project Description

This project features a CharRNN for generating texts in the style of the writer Andrei Platonov, along with a FastAPI service that implements this model.

## Table of Contents

- [📃 PlatonovGen](#-platonovgen)
  - [Project Description](#project-description)
  - [Table of Contents](#table-of-contents)
  - [Files](#files)
  - [Dataset](#dataset)
  - [CharRNN](#charrnn)
  - [How to Set Up the Project](#how-to-set-up-the-project)
  - [How to Use the Project](#how-to-use-the-project)
  - [Include Credits](#include-credits)
      - [Author](#author)
      - [Course](#course)
  - [License](#license)

## Files
- [rnn_training.ipynb](https://github.com/moxeeem/PlatonovGen/blob/main/rnn.ipynb) : Jupyter Notebook with RNN training
- [Platonov8.txt](https://github.com/moxeeem/PlatonovGen/blob/main/Platonov8.txt) : Dataset
- [/app](https://github.com/moxeeem/PlatonovGen/blob/main/app) : FastApi service files
- [/app/CharRNN.py](https://github.com/moxeeem/PlatonovGen/blob/main/app/CharRNN.py) : CharRNN model

## Dataset
The complete collection of Andrei Platonov's works (8 volumes) was used as a dataset for model training. 

[Андрей Платонов. Собрание сочинений. тт. 1-8, 2009-2011, Издательство: Время](https://archive.org/details/B-001-004-236/1/)


## CharRNN

This neural network is a character level recurrent neural network (CharRNN). It uses an LSTM layer with 3 layers, input size 152 (dictionary length), hidden size 1024 and dropout 0.5. The LSTM is followed by dropout 0.5 layer and full-connected layer with input size 1024 and output size 152 (character dictionary length).

The original text is tokenized and encoded using One-Hot Encoding.

**CharRNN**

| Layer  |        Type         |  Input Shape  | Output Shape | Parameters |
|--------|---------------------|---------------|--------------|------------|
| lstm   | LSTM                | (152,)         | (1024,)      | Num_layers=3, Batch_first=True, Dropout=0.5 |
| dropout| Dropout             | (1024,)        | (1024,)      | p=0.5, inplace=False |
| fc     | Linear              | (1024,)        | (152,)       | Bias=True  |

The following parameters are used to train the neural network:

| Parameter  |    Value   |
|------------|------------|
| epochs     | 20         |
| batch_size | 128        |
| seq_length | 100        |
| lr         | 0.001      |
| clip       | 5          |
| val_frac   | 0.1        |
| optimizer  | Adam       |
| criterion  | CrossEntropy|


In the end, we managed to get a model that can generate text with a style similar to Andrei Platonov and with good grammar, but with a lack of meaning.

---

Generation example:

<a href="https://i.postimg.cc/QxVd204w/2024-02-03-224158.png"><img src="https://i.postimg.cc/QxVd204w/2024-02-03-224158.png" width="75%"/></a>

## How to Set Up the Project

You can set up the project in two ways - using docker image or using virtual environment.

1. Using [docker image](https://hub.docker.com/repository/docker/moxeeeem/platonovgen/general)
   
   1. Download the image using command: 
      `docker pull moxeeeem/platonovgen`
   2. Run the image using command:
      `docker run -d -p 8000:8000 moxeeeem/platonovgen` 
   3. Go to http://localhost:8000/docs in your browser to see the service.
   

2. Using virtual environment
   
   1. Download the project files from github
   2. Create a virtual environment
      `python3 -m venv venv`
   3. Activate the virtual environment
      `source venv/bin/activate`
   4. Install requirements
      `pip install -r requirements.txt`
   5. Run the project
      `uvicorn main:app --reload`


## How to Use the Project

You can generate text in two ways - from request or from text file.

1. From request using `/generate_text/` endpoint
2. From text file using `/generate_text_from_file/` endpoint
   
   You can use *.txt or *.csv files.

## Include Credits

#### Author
Maxim Ivanov - [GitHub](https://github.com/moxeeem), [Telegram](https://t.me/fwznn_ql1d_8)

#### Course

This project was completed as part of the ["Рекуррентные сети в NLP и приложениях"](https://stepik.org/course/188632) course offered by [AI Education](https://stepik.org/users/628121134).


## License
This project is licensed under the MIT license. For more information, see the [LICENSE](/LICENSE) file.
