# N3LP
C++ implementation for Neural Network-based NLP, such as LSTM machine translation!<br>
This project ONLY requires a template library for linear algebra, Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)<br>
<b>Please note that this project started just for fun as my hobby, but sometimes it can be used to develop state-of-the-art models!</b>

## Long Short-Term Memory (LSTM)
The LSTM implemented in this project employs a variant of the major LSTM's gate computation where previous cell states are used to compute input/output gates.
See [1, 2] for the simplified version of the LSTM implemented here.

[1] http://arxiv.org/abs/1410.4615<br>
[2] http://nlp.stanford.edu/pubs/tai-socher-manning-acl2015.pdf

## BlackOut sampling
BlackOut [3, 4] is an approximation method to softmax classification learning with the large number of classes.

[3] http://arxiv.org/abs/1511.06909<br>
[4] https://github.com/IntelLabs/rnnlm

## Layer Normalization
Layer Normalization [5] is a normalization method for deep neural networks and it can be easily applied to recurrent neural networks, such as LSTMs.

[5] http://arxiv.org/abs/1607.06450

## USAGE ##
1) modify the line in Makefile to use Eigen<br>
EIGEN_LOCATION=$$HOME/local/eigen_new #Change this line to use Eigen

2) run the command "make"

3) ./run the command "n3lp", and then the seq2seq model training starts (currently)

## Projects using N3LP ##
Feel free to tell me (hassy@logos.t.u-tokyo.ac.jp) if you are using N3LP or have any questions!
* A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks (Hashimoto et al., arXiv, 2016)<br>
Paper: https://arxiv.org/abs/1611.01587<br>

* Tree-to-Sequence Attentional Neural Machine Translation (Eriguchi et al., ACL, 2016)<br>
Paper: http://www.logos.t.u-tokyo.ac.jp/~eriguchi/paper/ACL2016/ACL2016.pdf<br>
Code: https://github.com/tempra28/tree2seq

* The UT-AKY/KAY systems at WAT 2016 (Eriguchi et al., WAT, 2016; Hashimoto et al., WAT 2016)<br>
Paper (UT-AKY): to apper<br>
Paper (UT-KAY): http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/wat2016/paper_kay.pdf<br>

## Contributors ##
* <a href="http://www.logos.t.u-tokyo.ac.jp/~hassy/">Kazuma Hashimoto</a> - Mainly developing this project
* <a href="http://www.logos.t.u-tokyo.ac.jp/~eriguchi/">Akiko Eriguchi</a> - Developing practical applications (e.g. <a href="https://github.com/tempra28/tree2seq">tree-to-sequence neural machine translation</a>)

## Licence ##
MIT licence