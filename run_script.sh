#!/bin/sh

function usage {
    echo "Usage: cmd [-i <ip of the worker to kill at the end of the running (if not set won't kill it)>] 
    [-w <window tier name>] 
    [-n <with noise-reduction>]" 

    exit $1;
}

WINDOW=window
CLEAN=false
options='w:nh'
while getopts $options option
do
    case $option in

        w  ) WINDOW=$OPTARG;;
        n  ) CLEAN=true;;
        h  ) usage; exit;;
        \? ) echo "Unknown option: -$OPTARG" >&2; exit 1;;
        :  ) echo "Missing option argument for -$OPTARG" >&2; exit 1;;
    esac
done


date > log.txt
python3 ./helpers/check_req.py >> run_log.txt
if [ $? -eq 1 ]; then
    echo "Missing requirements"
    exit
fi


sh run.sh $WINDOW $CLEAN| tee  run_log.txt



