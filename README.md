# Connect 4 AI Game 

## Controls
- Left Arrow 
- Right Arrow
- Return


## Installation
Using [git](https://git-scm.com/downloads), clone the following repository from github: <br>
`git clone https://github.com/boraacer/Connect-4.git`<br>

Go into project directory in a terminal and enter <br>
`pip3 install -r requirements.txt`
<br>
To then open the game, use the following command: <br>
`python3 play.py` <br>
This command will use the pretrained model on the cpu allowing maximum compatability


## Training
To train the AI use the following command <br>
`python3 train.py` <br>
You will most likely need to edit the contents of `train.py` to support your gpu to change the amount of epochs/episodes