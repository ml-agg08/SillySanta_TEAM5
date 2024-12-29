Silly Santa

Santa has to give gifts to kids all over the globe. There are alot of kids loving him, waiting for the gifts. But its too hard for santa to read all the long letters send by the kids. At the same time, he cant avoid their love. So our SillySanta helps him with that. It reads the long letters for him and says what the kiddo loves <3

We made use of a pre-trained language model, BERT for this project. It's already trained on a very large wikipedia texts, making it able to capture the contextual, syntatcical and grammatical meanings of each tokens. 
Initial hours went in learning the architecture of the model. We decided to use fine tuning over transfer learning. Hence weights in most of the layers are chnaged a bit, but to preserve its previosuly mentioned abilities we kept the learning rate very low.
Then added a layer at the end for classification, since the initial pretrained model has entities like person, organisation etc which are compeltely useless for our case. Unlike image segemntation, it is very difficult to annotate the custom dataset. We had to annotate each and every token manually, so we were forced to keep the custom data set for fine tuning very small. 
But still it performs fairly well. 

PLEASE NOTE:

Since we are finetuining a pre trained model like bert, the actual directory size if much much bigger  ( around 12GB).
Thus in repo we have included, the 
1. main.py which consists of the training logic, check it to see the custom datasets we used, how we added an extra layer, the annotations, epochs everthing.
2. app.py, here we use flask since we've to scale it to  a website, check it to see all the server logics..
3. index.html, for a very basic interface.

Also we've hosted the website using a tunneling technique, basically exposing our local host 5000 to public using ngork.

If you want to see the training process, I'll share the colab link below:
