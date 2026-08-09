package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"math"
)

const (
	brentA         = -10.0
	brentB         = 10.0
	brentTolerance = 1e-6
	brentMaxSteps  = 1000
)

// Brent minimizuje konfigurisanu benchmark funkciju koristeći Brentov metod: kombinuje zlatni presek i paraboličku
// interpolaciju za 1D minimizaciju po prvoj koordinati
// problem.Point određuje dimenzionalnost rezultata, ostale koordinate ostaju na nuli
func Brent(problem models.Problem) (result models.Result, err error) {

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}

	dims := len(problem.Point)
	if dims == 0 {
		dims = 1
	}
	if err := fn.ValidateDimension(dims); err != nil {
		return result, err
	}

	a := brentA
	if v, ok := problem.Payload["a"].(float64); ok {
		a = v
	}

	b := brentB
	if v, ok := problem.Payload["b"].(float64); ok {
		b = v
	}

	tolerance := brentTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	maxSteps := brentMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	eval := func(xVal float64) float64 {
		point := make([]float64, dims)
		point[0] = xVal
		return fn.Evaluate(point)
	}

	const goldenRatio = 0.3819660112501051

	x, w, v := (a+b)/2, (a+b)/2, (a+b)/2
	fx := eval(x)
	fw, fv := fx, fx
	d, e := 0.0, 0.0

	steps := 0
	for ; steps < maxSteps; steps++ {
		mid := (a + b) / 2
		tol1 := tolerance*math.Abs(x) + 1e-10
		tol2 := 2 * tol1

		if math.Abs(x-mid) <= tol2-0.5*(b-a) {
			break
		}

		useGolden := true
		if math.Abs(e) > tol1 {
			r := (x - w) * (fx - fv)
			q := (x - v) * (fx - fw)
			p := (x-v)*q - (x-w)*r
			q = 2 * (q - r)
			if q > 0 {
				p = -p
			}
			q = math.Abs(q)
			etemp := e
			e = d
			if math.Abs(p) < math.Abs(0.5*q*etemp) && p > q*(a-x) && p < q*(b-x) {
				d = p / q
				u := x + d
				if u-a < tol2 || b-u < tol2 {
					d = tol1 * sign(mid-x)
				}
				useGolden = false
			}
		}
		if useGolden {
			if x >= mid {
				e = a - x
			} else {
				e = b - x
			}
			d = goldenRatio * e
		}

		var u float64
		if math.Abs(d) >= tol1 {
			u = x + d
		} else {
			u = x + tol1*sign(d)
		}
		fu := eval(u)

		if fu <= fx {
			if u >= x {
				a = x
			} else {
				b = x
			}
			v, fv = w, fw
			w, fw = x, fx
			x, fx = u, fu
		} else {
			if u < x {
				a = u
			} else {
				b = u
			}
			if fu <= fw || w == x {
				v, fv = w, fw
				w, fw = u, fu
			} else if fu <= fv || v == x || v == w {
				v, fv = u, fu
			}
		}

		if includeTrajectory {
			p := make([]float64, dims)
			p[0] = x
			trajectory = recordTrajectory(trajectory, steps, p, fx, false)
		}
	}

	point := make([]float64, dims)
	point[0] = x

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fx, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "brent",
		Point:      point,
		Value:      fx,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
