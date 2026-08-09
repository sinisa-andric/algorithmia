package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	madgradEpsilon      = 1e-6
	madgradMomentumCoef = 0.9
	madgradMaxSteps     = 1000
	madgradTolerance    = 1e-6
	madgradStepClip     = 1.0
)

// Madgrad minimizuje konfigurisanu benchmark funkciju koristeći MADGRAD: dual-averaging sekvenca z usidrena na
// početnu tačku, sa kumulativnim gradijentnim statistikama ponderisanim TEŽINOM koja raste sa sqrt(korak) — za
// razliku od svih EMA-baziranih solvera ovde, ovaj koristi rastuću (ne opadajuću) težinu po koraku
// problem.Point je početna tačka pretrage
func Madgrad(problem models.Problem) (result models.Result, err error) {

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

	epsilon := madgradEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	momentumCoef := madgradMomentumCoef
	if v, ok := problem.Payload["momentum_coef"].(float64); ok {
		momentumCoef = v
	}

	maxSteps := madgradMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := madgradTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x0 := append([]float64(nil), problem.Point...)
	x := append([]float64(nil), x0...)
	gradSum := make([]float64, dimensions)
	s := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		w := math.Sqrt(float64(steps + 1))
		z := make([]float64, dimensions)
		for d := range z {
			gradSum[d] += w * grad[d]
			s[d] += w * grad[d] * grad[d]
			z[d] = x0[d] - gradSum[d]/(math.Sqrt(s[d])+epsilon)
		}

		xNew := make([]float64, dimensions)
		for d := range xNew {
			xNew[d] = (1-momentumCoef)*x[d] + momentumCoef*z[d]
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xNew[d] - x[d]
		}
		delta = clipStep(delta, madgradStepClip)
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
		Method:     "madgrad",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
