package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	fromageLearningRate = 0.1
	fromageEpsilon      = 1e-8
	fromageMaxSteps     = 1000
	fromageTolerance    = 1e-6
	fromageStepClip     = 1.0
)

// Fromage minimizuje konfigurisanu benchmark funkciju koristeći Fromage optimizator: gradijent se skalira odnosom
// norme trenutne pozicije i norme gradijenta (bez ikakvog momentum ili EMA stanja), sa dodatnom normalizacijom
// imenioca learning rate-om — najjednostavniji solver u ovom modulu po odsustvu bilo kakvog istorijskog stanja
// problem.Point je početna tačka pretrage
func Fromage(problem models.Problem) (result models.Result, err error) {

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

	learningRate := fromageLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	epsilon := fromageEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := fromageMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := fromageTolerance
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
		xNorm := norm(x)
		gradNorm := norm(grad)
		scale := xNorm / (gradNorm + epsilon)

		delta := make([]float64, dimensions)
		for d := range delta {
			update := grad[d] * scale
			delta[d] = -learningRate * update / (1 + learningRate)
		}
		delta = clipStep(delta, fromageStepClip)
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
		Method:     "fromage",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
