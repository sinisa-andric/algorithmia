package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	dfpMaxSteps  = 1000
	dfpTolerance = 1e-6
	dfpStepClip  = 1.0
)

// Dfp minimizuje konfigurisanu benchmark funkciju koristeći DFP (Davidon-Fletcher-Powell) kvazi-Newton metod:
// aproksimacija inverznog Hesijana H se ažurira DIREKTNO DFP formulom (rang-2 update preko s=alpha*p i
// y=grad_new-grad_old), za razliku od quasi_newton.go koji za istu H koristi BFGS formulu — DFP i BFGS su
// matematički "dualni" pristupi ažuriranju istog objekta različitim formulama
// problem.Point je početna tačka pretrage
func Dfp(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := dfpMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dfpTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := append([]float64(nil), problem.Point...)
	h := identityMatrix(dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		p := negate(matVec(h, grad))

		alpha := backtrackAlpha(fn.Evaluate, x, p)
		s := make([]float64, dimensions)
		for d := range s {
			s[d] = alpha * p[d]
		}
		s = clipStep(s, dfpStepClip)

		xNew := make([]float64, dimensions)
		for d := range xNew {
			xNew[d] = x[d] + s[d]
		}

		gradNew := numGrad(fn.Evaluate, xNew, proximalGradEps)
		y := make([]float64, dimensions)
		for d := range y {
			y[d] = gradNew[d] - grad[d]
		}

		sy := dot(s, y)
		hy := matVec(h, y)
		yhy := dot(y, hy)
		if math.Abs(sy) > 1e-8 && math.Abs(yhy) > 1e-8 {
			ss := outer(s, s)
			hyhy := outer(hy, hy)
			for i := range h {
				for j := range h[i] {
					h[i][j] += ss[i][j]/sy - hyhy[i][j]/yhy
				}
			}
		}

		x = xNew

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "dfp",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
