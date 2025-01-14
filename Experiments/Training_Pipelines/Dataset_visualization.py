import ml_pipeline_utils
import matplotlib.pyplot as plt
import numpy as np

imgs = ml_pipeline_utils.Data_read_from_csv("data_source_list.csv")
img_list = imgs[0]
img_tags = imgs[1]

points_list = []

x = []
y = []
z = []

rock = 0
paper = 0
scissors = 0
ngg = 0

for tag in img_tags:
    if tag == "rock":
        rock += 1
    elif tag == "paper":
        paper += 1
    elif tag == "scissors":
        scissors += 1
    else: ngg += 1

total = rock+paper+scissors+ngg

print(rock)
print(paper)
print(scissors)
print(ngg)

y = np.array([rock, paper, scissors, ngg])
labels = [r'rock('+str(((rock/total)*100))+'%)', r'paper('+str(((paper/total)*100))+'%)', r'scissors('+str(((scissors/total)*100))+'%)',
          r'ngg('+str(((ngg/total)*100))+'%)']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(y, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")

# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.tight_layout()
plt.show()


x = ("rock", "paper", "scissors", "ngg")


plt.bar(x,y,align='center') # A bar chart
plt.xlabel('Gestures')
plt.ylabel('Dataset size')

plt.show()



