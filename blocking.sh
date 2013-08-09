#!/bin/sh

hsh="./Hash -X 64 -Y 64 -Z 60 -l 6 -r requests.data -h 10"

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_LOG=./.bf.log
export NAIVE="bf-naive.data"
export STAGING="bf-staging.data"
echo "# blocking-factor GPU-Time Occupancy" > ${NAIVE}
echo "# blocking-factor GPU-Time Occupancy" > ${STAGING}
for bf in $(seq 1 30) ; do
	progress=$(echo "(${bf} / 30) * 100.00" | bc -l)
	/bin/echo -en "\r${progress}..."
	# There seems to be an issue with the first run; I guess it takes a sec
	# for the GPU to get into "CUDA mode".  So run once unrecorded, to
	# prime CUDA.
	${hsh} -n -b ${bf} &>/dev/null
	${hsh} -b ${bf} &>/dev/null
	for i in 1 2 3 ; do
                ${hsh} -n -b ${bf} &>/dev/null
                awk -S --assign variable="${bf}" \
			-f parse-profile.awk \
			${COMPUTE_PROFILE_LOG} >> ${NAIVE}

		${hsh} -b ${bf} &>/dev/null
                awk -S --assign variable="${bf}" \
			-f parse-profile.awk \
			${COMPUTE_PROFILE_LOG} >> ${STAGING}
	done
done
echo "" # newline so that the above echos don't steal the line.
