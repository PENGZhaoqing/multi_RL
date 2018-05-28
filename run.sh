#!/usr/bin/env bash


/usr/bin/python2.7  run_dynamic.py --method ARMI --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
/usr/bin/python2.7  run_dynamic.py --method CommNet --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
/usr/bin/python2.7  run_dynamic.py --method IL --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.08 --dynamic True
/usr/bin/python2.7  run_dynamic.py --method VAIN --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
#/usr/bin/python2.7  run_dynamic.py --method LSTM --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
/usr/bin/python2.7  run_dynamic.py --method DIAL --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True

#/usr/bin/python2.7  run_dynamic.py --method ARMI --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
#/usr/bin/python2.7  run_dynamic.py --method CommNet --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
#/usr/bin/python2.7  run_dynamic.py --method IL --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.08 --dynamic True
#/usr/bin/python2.7  run_dynamic.py --method VAIN --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
#/usr/bin/python2.7  run_dynamic.py --method LSTM --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True
#/usr/bin/python2.7  run_dynamic.py --method DIAL --t-max 1 --max-agent-num 5 --game-name hunterworld --entropy-wt 0.05 --dynamic True

#shutdown -h now
