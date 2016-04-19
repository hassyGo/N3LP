# N3LP
C++ implementation for Neural Network-based NLP, such as LSTM machine translation!<br>
This project ONLY requires a template library for linear algebra, Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Long Short-Term Memory (LSTM)
The LSTM implemented in this project employs a variant of the major LSTM's gate computation where previous cell states are used to compute input/output gates.
See [1, 2] for the simplified version of the LSTM implemented here.

[1] http://arxiv.org/abs/1410.4615<br>
[2] http://nlp.stanford.edu/pubs/tai-socher-manning-acl2015.pdf

## BlackOut sampling
BlackOut [3, 4] is an approximation method to softmax classification learning with the large number of classes.

[3] http://arxiv.org/abs/1511.06909<br>
[4] https://github.com/IntelLabs/rnnlm

## USAGE ##
1) modify the line in Makefile to use Eigen<br>
EIGEN_LOCATION=$$HOME/local/eigen_new #Change this line to use Eigen

2) run the command "make"

3) ./run the command "n3lp", and then the seq2seq model training starts (currently)

## Projects using N3LP ##
Feel free to tell me (hassy@logos.t.u-tokyo.ac.jp) if you are using N3LP or have any questions!
* Tree-to-Sequence Attentional Neural Machine Translation  
Paper: http://arxiv.org/abs/1603.06075<br>
Code: https://github.com/tempra28/tree2seq
