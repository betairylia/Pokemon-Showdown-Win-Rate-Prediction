import urllib.request
from bs4 import BeautifulSoup

import sys
import time
from termcolor import colored, cprint

format = sys.argv[1]
print( '----- ' + colored(format, 'magenta') + ' -----')
pFile = open(format + ".txt", "a+")
pFile.close()

historyDict = {}

currentPage = 1
targetPage = 25

# Read in keys
rFile = open(format + ".txt", "r")
fileContent = [line.rstrip('\n') for line in rFile]
for i in range(len(fileContent)):
    if i % 4 == 0:
        historyDict[fileContent[i]] = True

for currentPage in range(1, targetPage + 1):
    print( '---   ' + colored("Page %d" % currentPage, 'magenta') + '   ---')
    url = "https://replay.pokemonshowdown.com/search/?format=" + format + "&page=" + str(currentPage)

    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

    req = urllib.request.Request(url=url,headers=headers)
    res = urllib.request.urlopen(req)

    html = res.read()
    soup = BeautifulSoup(html, features="lxml")

    links = soup.find_all('li')[8:]

    for i in range(len(links)):
        pFile = open(format + ".txt", "a+")

        replay_id = links[i].a['href'][1:].split('-')[1]
        if(replay_id in historyDict):
            print(colored("Skipping existing replay [" + replay_id + "]", 'red'))
            continue
        
        historyDict[replay_id] = True

        replay_url = "https://replay.pokemonshowdown.com" + links[i].a['href']
        replay_req = urllib.request.Request(url = replay_url, headers = headers)
        replay_res = urllib.request.urlopen(replay_req)

        replay_html = replay_res.read()
        replay_soup = BeautifulSoup(replay_html, features = "lxml")

        replay_log = replay_soup.find('script', {'class': 'log'}).text

        lines = replay_log.split('\n')
        
        p1Name = ""
        p2Name = ""

        p1Poke = []
        p2Poke = []

        win = 0

        for line in lines:
            keywords = line[1:].split("|")
            
            if keywords[0] == "player" and keywords[1] == "p1":
                p1Name = keywords[2]
            if keywords[0] == "player" and keywords[1] == "p2":
                p2Name = keywords[2]

            # check if it is 6v6
            if keywords[0] == "teamsize" and keywords[2] != "6":
                break
            
            if keywords[0] == "poke" and keywords[1] == "p1":
                p1Poke.append(keywords[2].split(',')[0])
            if keywords[0] == "poke" and keywords[1] == "p2":
                p2Poke.append(keywords[2].split(',')[0])

            if keywords[0] == "win":
                if keywords[1] == p1Name:
                    win = 1
                if keywords[1] == p2Name:
                    win = 2
        
        # If everything works correctly
        if win != 0:

            strP1 = colored("Player 1", "yellow") + "\n"
            strP2 = colored("Player 2", "cyan") + "\n"

            pFile.write(replay_id + "\n")

            # Write to file
            for pkmn in p1Poke:
                pFile.write(pkmn)
                pFile.write(",")

                strP1 += pkmn + ", "
            
            pFile.write("\n")

            for pkmn in p2Poke:
                pFile.write(pkmn)
                pFile.write(",")

                strP2 += pkmn + ", "
            
            pFile.write("\n")
            pFile.write(str(win) + "\n")

            print(colored("*******************", 'green'))
            print(strP1)
            print("=======")
            print(strP2)

            if win == 1:
                print(colored('Player 1 wins!', 'yellow'))
            if win == 2:
                print(colored('Player 2 wins!', 'cyan'))
            
            print(colored('( ' + format + ', ' + ("Page %d" % currentPage) + ' )', 'magenta'))

        pFile.close()

        # Wait some time
        time.sleep(.4)
