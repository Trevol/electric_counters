SRC=/hdd/Datasets/counters
DST=/hdd/Datasets/
rsync -crvh $SRC trevol@192.168.0.108:$DST

#SRC=/hdd/Datasets/counters
#DST=/hdd/TMP
#rsync -crvh $SRC $DST