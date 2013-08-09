#!/bin/sh

hsh="./Hash -X 64 -Y 64 -Z 60 -l 6 -r requests.data -b 8"

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_LOG=./.ht.log
export NAIVE="ht-size-naive.data"
export STAGING="ht-size-staging.data"
echo "# ht-size GPU-Time Occupancy" > ${NAIVE}
echo "# ht-size GPU-Time Occupancy" > ${STAGING}
for htsize in $(seq 25 5 150) ; do
	# There seems to be an issue with the first run; I guess it takes a sec
	# for the GPU to get into "CUDA mode".  So run once unrecorded, to
	# prime CUDA.
	${hsh} -n -h ${htsize} &>/dev/null
	${hsh} -h ${htsize} &>/dev/null
	for i in 1 2 3 ; do
		# make sure we don't get nonsense.
		rm -f ${COMPUTE_PROFILE_LOG}
                ${hsh} -n -h ${htsize} &>/dev/null
		if test $? -eq 0; then
			awk -S --assign variable="${htsize}" \
				-f parse-profile.awk \
				${COMPUTE_PROFILE_LOG} >> ${NAIVE}
		fi

		rm -f ${COMPUTE_PROFILE_LOG}
		${hsh} -h ${htsize} &>/dev/null
		if test $? -eq 0; then
			awk -S --assign variable="${htsize}" \
				-f parse-profile.awk \
				${COMPUTE_PROFILE_LOG} >> ${STAGING}
		fi
	done
	progress=$(echo "(${htsize} / 150) * 100.00" | bc -l)
	/bin/echo -en "\r${progress}..."
done
echo "" # newline so that the above echos don't steal the line.
gnuplot size-ht.gnu
