# IC_Algothon_24

## Description
Run data through our pipeline which has a form of momentum mechanism and determines weights using Markowitz mean-variance optimisation, optimising for 

We also ate Wingstop once (thank you Mango Habanero and Ranch).

## How to run
Download requirements and ensure passwords.csv is downloaded

Run the line of code below to start our API up

``` uvicorn api_db:app --reload```

Once this is up, just need to keep automation.py and api_db.py running.

## Automation
`api_db.py` automatically pulls information from the slack chat and writes this information to `passwords.csv`, definitely not perfect as it kind of gets screwed if there are more than 999 messages sent since the last password was sent but it works.

To automate submissions, we considered using a couple of methods
- requests (doesn't work because of user authentication)
- Selenium (didn't do this because dev time was slower than our chosen method)
- Google Forms API (you need to have designed the form iirc)
- JS Script run using TamperMonkey (works ok cus cookies bypass authentication and no captcha, needs Chrome installed)

Essentially the script sent into the Slack Chat allowed the form to be automatically submitted when the form was opened in a browser so I just used the batch file to open and close my browser tabs when we had submissions.

To run the code on a timer, could've used
- cron (UNIX but unfortunately I have a Windows PC)
- inbuilt Python timer module (not that reliable)
- batch files (good for this use case because it is easy to open Chrome from command line)

```
call .venv\Scripts\activate.bat

@echo off
set SLACK_BOT_TOKEN="put the bot token here"

call pip install -q -r requirements.txt
call python slack_pull_2.py
call python automation.py

start chrome "put the forms link here"

timeout 20 /nobreak REM not neccesary

taskkill /F /IM chrome.exe /T
```

## Shit we could've done better
- Used a better optimisation function rather than scipy.optimize.minimize for constraint reasons (SLSQP method needs a differentiable function but $\sum{|x|}$ is not)
- Distributed work better (different levels of experience meant that the work was pretty unevenly distributed)
- Analysed the dataset faster at the start and immediately hopped onto momentum/trend strats because they were CTAs
- Finished a better PnL tracker
- Slept more