from datetime import datetime
from zoneinfo import ZoneInfo

def curr_time():
    currt = datetime.now(ZoneInfo('Europe/Kiev'))
    return currt.strftime('%Y-%m-%d %H:%M:%S')


def printshare(msg, logfile="training_log.txt", logfile_mode='a'):
    print(msg)

    with open(logfile, logfile_mode) as f:
        print(msg, file=f)