#!/bin/bash

#set -x
echo "$1"
for file in $1
do 
    FILENAME=$file
    
    #grep ENDING 190710170717695300014_2.log | tail -n 1 | cut -d' ' -f5,6
    
    #STARTTIME=`grep run_start $FILENAME | head -n 1| cut -d' ' -f5,6,7`
    STARTTIME=`grep run_start $FILENAME |tail -n 1 |cut -d' ' -f5| cut -d',' -f1`
#    STARTTIME=$(date --date "$(grep "STARTING TIMING" $FILENAME | tail -n 1 | awk '{print $5,$6,$7}')" +%s )
    #ENDTIME=`grep run_stop $FILENAME | tail -n 1| cut -d' ' -f5,6,7`
    ENDTIME=`grep run_stop $FILENAME | tail -n 1 |cut -d' ' -f5| cut -d',' -f1`
#    ENDTIME=$(date --date "$(grep "ENDING TIMING" $FILENAME | tail -n 1 | awk '{print $5,$6,$7}')" +%s )
    if grep ROCm $FILENAME &>/dev/null
    then
        STARTTIME=`grep run_start $FILENAME | head -n 1| cut -d' ' -f2`
        ENDTIME=`grep run_stop $FILENAME | head -n 1| cut -d' ' -f2`
    fi
    
    EPOCH_NUM=$(grep eval_stop $FILENAME| tail -n 1 | awk '{print $17,$18}' | cut -d "}" -f1)    
    SOLVER_STEPS=$(grep SOLVER.STEPS $FILENAME | awk '{print $30,$31}' | head -n 1)
    BATCH_SIZE=$( grep -Po '(?<=SOLVER.IMS_PER_BATCH)\W*\K[^ ]*' $FILENAME | head -n 1 )
    BATCH_SIZE_SSD=$( grep batch-size $FILENAME | sed 's/=/ /g'| grep -Po '(?<=batch-size)\W*\K[^ ]*' | head -n 1  )
    BATCH_SIZE_MINIGO=$( grep '\-\-train_batch_size' $FILENAME|head -n 1 | sed 's/=/ /g'| grep -Po '(?<=batch_size)\W*\K[^ ]*' )
    BATCH_SIZE_TRANSFORMER=$( grep max-tokens $FILENAME|head -n 1 | grep -Po '(?<=max-tokens)\W*\K[^ ]*' )
    #BATCH_SIZE_GNMT=$( grep train-batch-size $FILENAME|head -n 1 | grep -Po '(?<=train-batch-size)\W*\K[^ ]*' ) #GNMT can use the SSD one
    BATCH_SIZE="$BATCH_SIZE_SSD""$BATCH_SIZE""$BATCH_SIZE_MINIGO""$BATCH_SIZE_TRANSFORMER""$BATCH_SIZE_GNMT"
    #date -d "2019-07-11 02:15:10" +"%s"
    STATUS=$( grep status $FILENAME | tail -n 1 | sed 's/:/ /g' |sed 's/"/ /g'| grep -Po '(?<=status)\W*\K[^ ]*'  )
    #STARTSEC=`date -d "$STARTTIME" +"%s"`
    #ENDSEC=`date -d "$ENDTIME" +"%s"`
    
    #SEC=$(echo $ENDSEC - $STARTSEC  |bc )
    #for minigo
    if grep run_stop>/dev/null $FILENAME
    then
            SEC=$(echo $ENDTIME - $STARTTIME  |bc )
    else
            SEC=$(sed -n -e 's/^.*timestamp//p' $FILENAME | cut -d":" -f2 | cut -d"}" -f1 | cut -d"," -f1 )
    fi

#    #for minigo - 20211013, minigo had been adjusted to as the same as others
#    if grep "beat target after">/dev/null $FILENAME
#    then
#	SEC=$(grep "beat target after" $FILENAME |awk '{print $6}'|cut -d's' -f1)
#    fi
    #for dlrm
    if grep "Hit target accuracy AUC">/dev/null $FILENAME
    then
	    #0.7
	    #SEC=$( grep "Hit target accuracy AUC" $FILENAME |awk '{ print $10 }'|cut -ds -f1 )
	    #1.0
	    SEC=$( grep "Hit target accuracy AUC" $FILENAME |awk '{ print $13 }'|cut -ds -f1 )
    fi
#    echo $SEC
    
    if [[ ! -z $SEC ]]
    then
        printf "FILENAME: $FILENAME "
        printf "$EPOCH_NUM "
        printf "$SOLVER_STEPS "
        printf "batch_size $BATCH_SIZE "
        printf "status:$STATUS "
        echo "scale=4; $SEC / 60000" | bc 2>/dev/null

    fi
    #for file in `ls 200*`; do ./results.sh $file; done | sort -n -k3
done

#for d in ./*/ ; do (cd "$d" && pwd && ./results.sh "200*"); done
#  ./results.sh "resnet50/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "minigo/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "dlrm/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "transformer/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "gnmt/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "maskrcnn/*" |cut -d' ' -f10 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "ssd/*" |sort -k10 -n |head -n 4| tail -n 3 |awk '{ sum += $10 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "ssd/*" |sort -k8 -n |head -n 4| tail -n 3 |awk '{ sum += $8 } END { if (NR > 0) print sum / NR }'
