#!/bin/sh

hsh="./Hash -X 64 -Y 64 -Z 60 -l 6 -r requests.data -b 4"

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_LOG=./.ht.log
echo "# ht-size GPU-Time Occupancy" > ht-size-naive.data
echo "# ht-size GPU-Time Occupancy" > ht-size-staging.data
for htsize in $(seq 1 150) ; do
	# There seems to be an issue with the first run; I guess it takes a sec
	# for the GPU to get into "CUDA mode".  So run once unrecorded, to
	# prime CUDA.
	${hsh} -n -h ${htsize} &>/dev/null
	${hsh} -h ${htsize} &>/dev/null
	for i in 1 2 3 ; do
                ${hsh} -n -h ${htsize} &>/dev/null
                awk -S --assign variable="${htsize}" \
			-f parse-profile.awk .ht.log >> ht-size-naive.data

		${hsh} -h ${htsize} &>/dev/null
                awk -S --assign variable="${htsize}" \
			-f parse-profile.awk .ht.log >> ht-size-staging.data
	done
	progress=$(echo "(${htsize} / 150) * 100.00" | bc -l)
	/bin/echo -en "\r${progress}..."
done
echo "" # newline so that the above echos don't steal the line.
