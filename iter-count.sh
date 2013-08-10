#!/bin/sh
# varies the iteration count

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
export BENCHMARK=".iter-count.data"
export MAX_SIZE=5
echo "# iter-count GPU-Time Occupancy" > ${BENCHMARK}
for iter in $(seq 1 1 ${MAX_SIZE}) ; do
	${cmpl} -DMAX_ITERS=${iter} || error "compile failed"
	${link} || error "link failed"

	for i in 1 2 3 ; do
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
