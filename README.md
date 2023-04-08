# VisualStorytelling
My bachelor's thesis -- Visual Storytelling -- an encoder-decoder model consisting of both convolutional and attentional recurrent networks that transform a sequence of 5 images into a context-generative story in natural language (english).

# Base model architecture
<img width="1169" alt="image" src="https://user-images.githubusercontent.com/39880374/230730540-7636dac0-813f-4ded-9077-4fc21acc67db.png">

# Results
From the above-illustrated architecture, two models were trained and compared:
- Encoding the whole album's into one embedding vector (**CS**)
- Encoding each photo separately to its own embedding vector and transmitting them to the decoder sequentially (**CC**)

<img width="878" alt="image" src="https://user-images.githubusercontent.com/39880374/230730797-5461b65b-25fb-466e-bb34-4d5461eff89e.png">
<img width="856" alt="image" src="https://user-images.githubusercontent.com/39880374/230730736-f9cc3e09-d059-45f4-83de-696581cfe4ba.png">
<img width="885" alt="image" src="https://user-images.githubusercontent.com/39880374/230730821-24ec1883-a8e3-4c2b-b372-a2a37a6893e9.png">

<img width="693" alt="image" src="https://user-images.githubusercontent.com/39880374/230730857-26279b9a-fbb0-4c13-a887-c2af5f090cdb.png">

#### How overfitting, even though it increases METEOR and ROGUE score, harm the results
<img width="785" alt="image" src="https://user-images.githubusercontent.com/39880374/230731307-94df6d40-6b63-475c-b29c-94c092c4d5b0.png">
