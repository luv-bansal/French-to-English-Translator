# French-to-English-Translator
Using GRU with glove embedding transfer learning in TensorFlow framework I built a french to English translator. 
The architecture of model consists of two parts a) Encoder model and b)Decoder model
Encoder model takes input as French language and outputs the initial state for the Decoder model.  And Decoder takes input as partial english sentence and the output of the Encoder model as an initial state in GRUs and outputs the English sentence.
