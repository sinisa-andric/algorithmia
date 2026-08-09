package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	truncatedNewtonMaxSteps  = 1000
	truncatedNewtonTolerance = 1e-6
	truncatedNewtonStepClip  = 1.0
	truncatedNewtonCGIters   = 5
	truncatedNewtonHvEps     = 1e-5
)

// TruncatedNewton minimizuje konfigurisanu benchmark funkciju koristeći Truncated Newton metod: umesto računanja
// pune Hesijan matrice (kao newton.go/gauss_newton.go), rešava H*p=-grad PRIBLIŽNO sa najviše
// truncatedNewtonCGIters unutrašnjih Conjugate-Gradient koraka, gde se Hesijan-vektor proizvod aproksimira
// direkcionim izvodom gradijenta: Hv ≈ (grad(x+eps*v)-grad(x))/eps — CG se namerno prekida rano ("truncated"),
// ne rešava do kraja kao puna CG metoda
// problem.Point je početna tačka pretrage
func TruncatedNewton(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	maxSteps := truncatedNewtonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := truncatedNewtonTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)

		hessVec := func(v []float64) []float64 {
			xPlus := make([]float64, dimensions)
			for d := range xPlus {
				xPlus[d] = x[d] + truncatedNewtonHvEps*v[d]
			}
			gradPlus := numGrad(fn.Evaluate, xPlus, proximalGradEps)
			hv := make([]float64, dimensions)
			for d := range hv {
				hv[d] = (gradPlus[d] - grad[d]) / truncatedNewtonHvEps
			}
			return hv
		}

		// truncated CG: rešava H*p=-grad približno, najviše truncatedNewtonCGIters unutrašnjih koraka
		p := make([]float64, dimensions)
		r := negate(grad)
		cgDir := append([]float64(nil), r...)
		rsOld := dot(r, r)
		for iter := 0; iter < truncatedNewtonCGIters; iter++ {
			if rsOld < 1e-20 {
				break
			}
			hd := hessVec(cgDir)
			dHd := dot(cgDir, hd)
			if math.Abs(dHd) < 1e-10 {
				break
			}
			alphaCg := rsOld / dHd
			for i := range p {
				p[i] += alphaCg * cgDir[i]
			}
			for i := range r {
				r[i] -= alphaCg * hd[i]
			}
			rsNew := dot(r, r)
			if rsNew < 1e-20 {
				break
			}
			beta := rsNew / rsOld
			for i := range cgDir {
				cgDir[i] = r[i] + beta*cgDir[i]
			}
			rsOld = rsNew
		}

		alpha := backtrackAlpha(fn.Evaluate, x, p)
		s := make([]float64, dimensions)
		for i := range s {
			s[i] = alpha * p[i]
		}
		s = clipStep(s, truncatedNewtonStepClip)
		for i := range x {
			x[i] += s[i]
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "truncated_newton",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
