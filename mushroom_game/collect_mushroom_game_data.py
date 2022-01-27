### an example of a simple game
### you're shown pictures of mushrooms and have to decide whether to eat them or not. you get given rewards or punishments as feedback.
### Unfortunately, this code only runs on windows. 

from graphics import *
import random
import numpy as np
import msvcrt
import time

class TimeoutExpired(Exception):
    pass

def input_with_timeout(prompt, timeout, timer=time.monotonic): ### from https://stackoverflow.com/questions/15528939/python-3-timed-input
    sys.stdout.write(prompt)
    sys.stdout.flush()
    endtime = timer() + timeout
    result = []
    while timer() < endtime:
        if msvcrt.kbhit():
            result.append(msvcrt.getwche()) #XXX can it block on multibyte characters?
            if result[-1] == '\r':
                return ''.join(result[:-1])
        time.sleep(0.04) # just to yield to other processes/threads
    raise TimeoutExpired

def get_reward(stim,act):
    if stim == 0:
        if act == "go":
            r = get_r_prob_80()
        elif act == "nogo":
            r = 0      
    elif stim == 1:
        if act == "go":
            r = 0
        elif act == "nogo":
            r = get_r_prob_80()    
    elif stim == 2:
        if act == "go":
            r = get_loss_prob_80()
        elif act == "nogo":
            r = 0    
    elif stim == 3:
        if act == "go":
            r = 0
        elif act == "nogo":
            r = get_loss_prob_80()
    return r

def get_r_prob_80():
    rand = np.random.random()
    if rand < 0.8:
        return 1
    else:
        return 0
def get_loss_prob_80():
    rand = np.random.random()
    if rand < 0.8:
        return -1
    else:
        return 0


def showmushroom(win,file):
	
	mymush = Image(Point(160,160),file)
	mymush.draw(win)
	#win.getMouse() # Pause to view result
	#win.close()    # Close window when done
	return win

def get_reward_pic(reward):
	if reward==-1:
		file='sadface.png'
	elif reward==0:
		file='neutralface.png'
	elif reward==1:
		file='happyface.png'
	else:
		file='ddd'
		print('something went wrong with the reward value')
	return file

def showfeedback(reward):
	file=get_reward_pic(reward)
	win = GraphWin("feedback", 320, 320)
	feedback_face = Image(Point(160,160),file)
	
	feedback_face.draw(win)

	time.sleep(2)
	win.close()
	#win.getMouse() # Pause to view result
	#win.close()    # Close window when done
	return 0

Ntrials=10
mushroomfiles = ['redcirclemushroom.png','bluestarmushroom.png','orangesquaremushroom.png','greytrianglemushroom.png']

filename=input('Under what filename would you like to store your data?')
f = open(filename,'w')

f.write("s\t")
f.write("a\t")
f.write("r\n")

print("You will be shown a series of pictures of mushrooms. You need to decide whether to eat them or not. \
If you choose to eat a mushroom, press enter. If you choose not to eat a mushroom, do nothing. \
Due to bugginess in the code (sorry), once the game starts, please try not to press enter at any \
time other than when there's a mushroom on the screen, and press it at most once per mushroom.")
pause_before_begin = input("Press enter to begin.")
for i in range(Ntrials):
	s = random.randint(0,3)
	
	mush = mushroomfiles[s]
	win = GraphWin("Do you want to eat this mushroom?", 320, 320)
	#win = showmushroom(win,mush)
	mymush = Image(Point(160,160),mush)
	mymush.draw(win)
	
	act="not a valid action" # make the action invalid to start with

	try:
		inp = input_with_timeout("",2)
	except TimeoutExpired:
		act = "nogo"
		print("You did not eat the mushroom.")
	else:
		act="go"
		print("You ate the mushroom.")
	win.close()


	r = get_reward(s,act)
	#showfeedback(r)
	print('Your reward is ',r)
	file=get_reward_pic(r)
	get_reward_pic(r)
	
	win = GraphWin("feedback", 320, 320)
	feedback_face = Image(Point(160,160),file)
	
	feedback_face.draw(win)

	time.sleep(2)
	win.close()

	f.write(str(s))
	f.write('\t')
	f.write(act)
	f.write('\t')
	f.write(str(r))
	f.write('\n')
f.close()