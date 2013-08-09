/* generates a sample 'requests' file based on what a probable ray request
 * pattern would be. */
package main
import(
	"flag"
	"fmt"
	_ "os"
	"math"
)

/* the brick a ray is currently in. */
type Ray struct {
	loc [3]float32; /* where the ray is right now */
	dir [3]float32; /* ray direction */
	active bool; /* is the ray still in the volume? */
}

func (r* Ray) advance() {
	/* if the ray already exited, don't bother. */
	if(!r.active) { return; }
	r.loc = vadd(r.loc, vsmul(r.dir, 1.0/32.0));
}

var bdims [4]uint; /* dimensions of the volume, in bricks. */
var world [3]float32; /* world space represented */
var fakeworld [3]float64; /* go annoys me. */
var filename string;
const (
	resolutionX = 640
	resolutionY = 480
)
var eye [3]float32;
var ref [3]float32;

func init() {
	flag.UintVar(&bdims[0], "bx", 0, "bricks in X");
	flag.UintVar(&bdims[1], "by", 0, "bricks in Y");
	flag.UintVar(&bdims[2], "bz", 0, "bricks in Z");
	flag.UintVar(&bdims[3], "lod", 0, "number of LODs");
	flag.Float64Var(&fakeworld[0], "X", 1.0, "world space range in X");
	flag.Float64Var(&fakeworld[1], "Y", 2.0, "world space range in y");
	flag.Float64Var(&fakeworld[2], "Z", 3.0, "world space range in z");
}

func tjfparse() {
	flag.Parse();
	world[0] = float32(fakeworld[0]);
	world[1] = float32(fakeworld[1]);
	/* reverse the world depth; looking down negative Z */
	world[2] = -float32(fakeworld[2]);
	eye = [3]float32{
		float32(world[0]/2.0),
		float32(3.0*world[1]/4.0),
		12.0,
	}
	ref = [3]float32{
		float32(world[0]/2.0),
		float32(world[1]/2.0),
		float32(world[2]/2.0),
	}
}

func active(rays []Ray) (bool) {
	for _, ray := range rays {
		if ray.active { return true; }
	}
	return false;
}

func lerp(v float32, irange [2]float32, outrange [2]uint) (float32) {
	return float32(outrange[0]) +
	       (float32(v)-irange[0]) * (float32(outrange[1]-outrange[0])) /
	                                (irange[1]-irange[0]);
}

/* god you piss me off sometimes, go. */
func lerpu(v uint, irange [2]uint, ominmax [2]float32) (float32) {
	return float32(ominmax[0]) +
	       float32(v-irange[0]) * (float32(ominmax[1]-ominmax[0])) /
	                              float32(irange[1]-irange[0])
}

/** converts a world space location to the appropriate brick */
func tobrick(loc [3]float32, world [3]float32, nbricks [3]uint) (bool, [3]uint) {
	if loc[0] < 0.0 || loc[0] > world[0] ||
	   loc[1] < 0.0 || loc[1] > world[1] ||
	   loc[2] < 0.0 || loc[2] > -world[2] {
		return false, [3]uint{}
	}

	brick_sz := [3]uint{32, 32, 32} // assume 32^3 bricks.
	nvoxels := [3]uint{
		nbricks[0] * brick_sz[0],
		nbricks[1] * brick_sz[1],
		nbricks[2] * brick_sz[2],
	};
	voxelidx := [3]float32{
                lerp(loc[0], [2]float32{0.0,world[0]}, [2]uint{0,nvoxels[0]}),
                lerp(loc[1], [2]float32{0.0,world[1]}, [2]uint{0,nvoxels[1]}),
                lerp(loc[2], [2]float32{0.0,world[2]}, [2]uint{0,nvoxels[2]}),
	}
	return true, [3]uint{
		uint(voxelidx[0]) / brick_sz[0],
		uint(voxelidx[1]) / brick_sz[1],
		uint(voxelidx[2]) / brick_sz[2],
	}
}

func in_bounds(r Ray) (bool) {
	return 0 <= r.loc[0] && r.loc[0] <= world[0] &&
	       0 <= r.loc[1] && r.loc[1] <= world[1] &&
	       0 <= r.loc[2] && r.loc[2] <= world[2];
}

/* for deriving initial ray location from pixel coordinate */
func location(x uint, y uint) ([3]float32) {
	xrange := [2]uint{uint(0), resolutionX};
	yrange := [2]uint{uint(0), resolutionY};
	const border = 4.0
	xworld := [2]float32{0.0-border, world[0]+border}
	yworld := [2]float32{0.0-border, world[1]+border}
	// we've semi-arbitrarily decided the near plane is at z=5
	return [3]float32{
		lerpu(x, xrange, xworld),
		lerpu(y, yrange, yworld),
		5.0,
	}
}

func intersect(r Ray) (bool, [3]float32) {
	bmin := [3]float32{0.0, 0.0, 0.0} /* just renaming these */
	bmax := [3]float32{world[0], world[1], world[2]}
	var t0 [3]float32;
	var t1 [3]float32;
	t0[0] = (bmin[0] - r.loc[0]) / r.dir[0];
	t0[1] = (bmin[1] - r.loc[1]) / r.dir[1];
	t0[2] = (bmin[2] - r.loc[2]) / r.dir[2];
	t1[0] = (bmax[0] - r.loc[0]) / r.dir[0];
	t1[1] = (bmax[1] - r.loc[1]) / r.dir[1];
	t1[2] = (bmax[2] - r.loc[2]) / r.dir[2];

	actualmin := vmin(t0, t1);
	actualmax := vmax(t0, t1);

/*
	fmt.Printf("loc: %f %f %f\n", r.loc[0], r.loc[1], r.loc[2]);
	fmt.Printf("actual: %f %f %f\n", actualmin[0], actualmin[1],
	           actualmin[2]);
*/
	minmax := min(min(actualmax[0], actualmax[1]), actualmax[2]);
	maxmin := max(max(actualmin[0], actualmin[1]), actualmin[2]);

	//fmt.Printf("v: %f %f\n", minmax, maxmin);

	if(minmax >= maxmin) {
		t := minmax;
		return true, [3]float32{
			r.loc[0] + t*r.dir[0],
			r.loc[1] + t*r.dir[1],
			r.loc[2] + t*r.dir[2],
		}
	}
	return false, [3]float32{}
}

func vsmul(vec [3]float32, scalar float32) ([3]float32) {
	return [3]float32{
		vec[0] * scalar,
		vec[1] * scalar,
		vec[2] * scalar,
	}
}
func vmag(vec [3]float32) (float32) {
	ssq := float64(vec[0]*vec[0]) +
	       float64(vec[1]*vec[1]) +
	       float64(vec[2]*vec[2]);
	return float32(math.Sqrt(ssq));
}
func vnormalize(vec [3]float32) ([3]float32) {
	magnitude := vmag(vec)
	return [3]float32{
		vec[0] / magnitude,
		vec[1] / magnitude,
		vec[2] / magnitude,
	}
}
func vsub(from [3]float32, to [3]float32) ([3]float32) {
	return [3]float32{
		from[0] - to[0],
		from[1] - to[1],
		from[2] - to[2],
	}
}
func vadd(a [3]float32, b [3]float32) ([3]float32) {
	return [3]float32{
		a[0] + b[0],
		a[1] + b[1],
		a[2] + b[2],
	}
}
func min(a float32, b float32) (float32) {
	if(a < b) { return a; }
	return b;
}
func max(a float32, b float32) (float32) {
	if(a < b) { return b; }
	return a;
}
func vmin(a [3]float32, b [3]float32) ([3]float32) {
	return [3]float32{
		min(a[0], b[0]),
		min(a[1], b[1]),
		min(a[2], b[2]),
	}
}
func vmax(a [3]float32, b [3]float32) ([3]float32) {
	return [3]float32{
		max(a[0], b[0]),
		max(a[1], b[1]),
		max(a[2], b[2]),
	}
}

func main() {
	tjfparse()
	rays := make([]Ray, resolutionX*resolutionY); 

	// initialize all our rays to start where they hit the box.
	for y:=uint(0); y < resolutionY; y++ {
		for x:=uint(0); x < resolutionX; x++ {
			idx := y*resolutionX + x;
			rays[idx].loc = eye;
			//rays[idx].loc = location(x,y) // pixel location
			rays[idx].dir = vnormalize(vsub(eye, location(x,y)));
			rays[idx].active, rays[idx].loc = intersect(rays[idx]);
			if hit, hitloc := intersect(rays[idx]); hit == true {
				rays[idx].loc = hitloc;
				rays[idx].active = true;
			} else {
				rays[idx].active = false;
			}
		}
	}

	// now advance the rays until they exit the volume.
	for ; active(rays) ; {
		for i:=0 ; i < len(rays) ; i++ {
                        if(rays[i].active) {
                                //fmt.Printf("%f %f %f -> ", rays[i].loc[0],
				//           rays[i].loc[1],
                                //           rays[i].loc[2]);
                                rays[i].advance()
                                //fmt.Printf("%f %f %f\n", rays[i].loc[0],
				//           rays[i].loc[1],
                                //           rays[i].loc[2]);
                                hatego := [3]uint{bdims[0], bdims[1], bdims[2]};
                                var brick [3]uint;
                                rays[i].active, brick = tobrick(rays[i].loc,
				                                world, hatego)
				// hack.
				brick[0] %= bdims[0]
				brick[1] %= bdims[1]
				brick[2] %= bdims[2]
                                if rays[i].active {
                                        fmt.Printf("%d %d %d\n", brick[0],
					           brick[1], brick[2]);
                                }
                        }
                }
	}
}
