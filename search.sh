#!/bin/sh
# varies how many elements we search in the global table before deciding the
# value is already there.

hsh="./Hash -h 200 -X 64 -Y 64 -Z 60 -l 6 -r requests.req -b 2"
glcf=$(pkg-config --cflags glib-2.0)
cmpl="nvcc -O3 ${glcf} -arch=sm_30 -c ht.cu -o ht.o"
glld=$(pkg-config --libs glib-2.0)
link="nvcc -O3 ${glcf} -arch=sm_30 ht.o opt.o -o Hash ${glld}"

function error {
	echo "$@"
	exit 1
}

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_LOG=./.iter.log
export BENCHMARK=".elems-to-search.data"
export MAX_SIZE=20
echo "# elems GPU-Time Occupancy" > ${BENCHMARK}
for iter in $(seq 30 -1 ${MAX_SIZE}) ; do
	${cmpl} -DELEMS_TO_SEARCH=${iter} || error "compile failed"
	${link} || error "link failed"

	for i in 1 2 ; do
		# make sure we don't get nonsense from a previous run.
		rm -f ${COMPUTE_PROFILE_LOG}
		${hsh} -n &>/dev/null
		if test $? -eq 0; then
			awk -S --assign variable="${iter}" \
				-f parse-profile.awk \
				${COMPUTE_PROFILE_LOG} >> ${BENCHMARK}
		else
			echo "wtf?!"
		fi
	done
	progress=$(echo "(${iter} / ${MAX_SIZE}) * 100.00" | bc -l)
	/bin/echo -en "\r${progress}..."
done
echo "" # newline so that the above echos don't steal the line.
