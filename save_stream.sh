#!/bin/bash


# while :
# do
#     IP=(rtsp://admin:123456@192.168.123.235:554/stream1 \
#         rtsp://admin:123456@192.168.123.235:554/stream1 \
#         rtsp://admin:123456@192.168.123.235:554/stream2 \
#         rtsp://admin:123456@192.168.123.235:554/stream2 \
#     )
#     a=0
#     splite="_"
#     for ip in ${IP[*]}
#     do
#         a=$(($a+1))
#         date_time=`date "+%Y_%m_%d"`
#         SAVE_PATH="/home/pc001/e/lcn/stream/$a/${date_time}"
#         if [ ! -d ${SAVE_PATH} ];then mkdir -p ${SAVE_PATH}; fi
#         (ffmpeg -i $ip -codec:a aac -b:a 64k -codec:v libx264 -b:v 768k -r 15 -t 60 -s 1280x720 ${SAVE_PATH}/$a${splite}`date +%Y%m%d-%H%M`.mkv
#         )&
#     done
# done

# exit 0

splite="_"
basePath="/home/pc001/e/lcn/stream"

cat ip_list.config |

while read line
do
    ip=$line
    a=$(($a+1))
    date_time=`date "+%Y_%m_%d"`
    SAVE_PATH="$basePath/$a/${date_time}"
    if [ ! -d ${SAVE_PATH} ];then mkdir -p ${SAVE_PATH}; fi
    ffmpeg -i $ip -rtsp_transport http -codec:a aac -b:a 64k -codec:v libx264 -b:v 768k -r 15 -t 60 -s 1280x720 ${SAVE_PATH}/$a${splite}`date +%Y%m%d-%H%M`.mkv >/dev/null 2>&1 &
done

exit