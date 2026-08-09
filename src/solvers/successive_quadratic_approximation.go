package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	successiveQuadraticApproximationMaxSteps  = 1000
	successiveQuadraticApproximationTolerance = 1e-6
	successiveQuadraticApproximationStepClip  = 1.0
	successiveQuadraticApproximationH         = 0.1
)

// SuccessiveQuadraticApproximation minimizuje konfigurisanu benchmark funkciju uzastopnim fitovanjem 1D parabole
// po SVAKOJ DIMENZIJI NEZAVISNO (3 tačke x-h,x,x+h) i skokom na teme svake parabole — razlika od
// quadratic_interpolation.go koji radi u punom prostoru, ovde je fit strogo po-dimenziji nezavisan
// problem.Point je početna tačka pretrage
func SuccessiveQuadraticApproximation(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := successiveQuadraticApproximationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := successiveQuadraticApproximationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	h := successiveQuadraticApproximationH
	if v, ok := problem.Payload["h"].(float64); ok {
		h = v
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

		y0 := fn.Evaluate(x)
		xNew := make([]float64, dimensions)
		for d := range xNew {
			xMinus := append([]float64(nil), x...)
			xMinus[d] -= h
			xPlus := append([]float64(nil), x...)
			xPlus[d] += h
			yMinus := fn.Evaluate(xMinus)
			yPlus := fn.Evaluate(xPlus)

			a := (yPlus - 2*y0 + yMinus) / (2 * h * h)
			b := (yPlus - yMinus) / (2 * h)
			xNew[d] = x[d] - b/(2*a+1e-8)
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xNew[d] - x[d]
		}
		delta = clipStep(delta, successiveQuadraticApproximationStepClip)
		for d := range x {
			x[d] += delta[d]
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
		Method:     "successive_quadratic_approximation",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
