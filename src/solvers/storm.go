package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	stormCStorm    = 10.0
	stormEpsilon   = 1e-8
	stormMaxSteps  = 1000
	stormTolerance = 1e-6
	stormStepClip  = 1.0
)

// Storm minimizuje konfigurisanu benchmark funkciju koristeći STORM: learning rate opada sa KUBNIM korenom
// kumulativne sume kvadrata gradijenata (ne kvadratnim kao Adagrad), a momentum koeficijent zavisi od PROMENE same
// stope učenja između koraka — jedinstven mehanizam koji nijedan drugi solver u ovom modulu nema
// problem.Point je početna tačka pretrage
func Storm(problem models.Problem) (result models.Result, err error) {

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

	cStorm := stormCStorm
	if v, ok := problem.Payload["c_storm"].(float64); ok {
		cStorm = v
	}

	epsilon := stormEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := stormMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := stormTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	dPrev := make([]float64, dimensions)
	gradPrev := make([]float64, dimensions)
	sumGradSq := 0.0
	lrPrev := 0.0

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		gradNormSq := 0.0
		for _, g := range grad {
			gradNormSq += g * g
		}

		lrT := cStorm / math.Cbrt(epsilon+sumGradSq)

		ratio := 1.0
		if steps > 0 {
			ratio = lrT / lrPrev
		}

		d := make([]float64, dimensions)
		delta := make([]float64, dimensions)
		for i := range d {
			d[i] = grad[i] + (1-ratio)*(dPrev[i]-gradPrev[i])
			delta[i] = -lrT * d[i]
		}
		delta = clipStep(delta, stormStepClip)
		for i := range x {
			x[i] += delta[i]
		}

		sumGradSq += gradNormSq
		lrPrev = lrT
		dPrev = d
		gradPrev = grad

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
		Method:     "storm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
