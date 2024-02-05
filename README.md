# 📃 PlatonovGen
[![readme.jpg](http://anopic.ag/NjU4eMabP9xSK5BcjENdmmoJND9PpUW4Xii4yZlb.jpg)](http://anopic.ag/NjU4eMabP9xSK5BcjENdmmoJND9PpUW4Xii4yZlb.jpg)

## Project Description

This project features a CharRNN for generating texts in the style of the writer Andrei Platonov, along with a FastAPI service that implements this model.

## Table of Contents

- [📃 PlatonovGen](#-platonovgen)
  - [Project Description](#project-description)
  - [Table of Contents](#table-of-contents)
  - [Files](#files)
  - [Dataset](#dataset)
  - [CharRNN](#charrnn)
  - [Web Service](#web-service)
  - [Include Credits](#include-credits)
      - [Author](#author)
      - [Course](#course)
  - [License](#license)

## Files
- [rnn_training.ipynb](https://github.com/moxeeem/PlatonovGen/blob/main/rnn.ipynb) : Jupyter Notebook with RNN training
- [rnn.net](https://github.com/moxeeem/PlatonovGen/blob/main/rnn.net) : File with trained RNN
- [Platonov8.txt](https://github.com/moxeeem/PlatonovGen/blob/main/Platonov8.txt) : Dataset


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

<a href="http://anopic.ag/pGiSH6Ifca1OhnAfXjiKaYxlxKJatZYWW3UZ9sof.jpg"><img src="http://anopic.ag/pGiSH6Ifca1OhnAfXjiKaYxlxKJatZYWW3UZ9sof.jpg" width="75%"/></a>

## Web Service
...


## Include Credits

#### Author
Maxim Ivanov - [GitHub](https://github.com/moxeeem), [Telegram](https://t.me/fwznn_ql1d_8)

#### Course

This project was completed as part of the ["Рекуррентные сети в NLP и приложениях"](https://stepik.org/course/188632) course offered by [AI Education](https://stepik.org/users/628121134).


## License
This project is licensed under the MIT license. For more information, see the [LICENSE](/LICENSE) file.
