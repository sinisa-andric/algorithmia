package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	fireflyAlgorithmFireflies = 20
	fireflyAlgorithmAlpha     = 0.5
	fireflyAlgorithmBeta      = 1.0
	fireflyAlgorithmGamma     = 1.0
	fireflyAlgorithmMaxSteps  = 500
	fireflyAlgorithmTolerance = 1e-6
	fireflyAlgorithmSpread    = 5.0
)

// FireflyAlgorithm minimizuje sphere funkciju koristeći firefly algoritam: tamniji svici se privlače ka svetlijima,
// pri čemu privlačenje opada sa udaljenošću uz dodat član nasumične šetnje
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func FireflyAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	fireflies := fireflyAlgorithmFireflies
	if v, ok := problem.Payload["fireflies"].(float64); ok {
		fireflies = int(v)
	}

	alpha := fireflyAlgorithmAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	beta := fireflyAlgorithmBeta
	if v, ok := problem.Payload["beta"].(float64); ok {
		beta = v
	}

	gamma := fireflyAlgorithmGamma
	if v, ok := problem.Payload["gamma"].(float64); ok {
		gamma = v
	}

	maxSteps := fireflyAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := fireflyAlgorithmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	positions := make([][]float64, fireflies)
	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = problem.Point[d] + (rand.Float64()*2-1)*fireflyAlgorithmSpread
		}
		positions[i] = position
	}

	light := make([]float64, fireflies)
	for i, position := range positions {
		light[i] = 1 / (1 + fn.Evaluate(position))
	}

	best := append([]float64(nil), positions[0]...)
	bestValue := fn.Evaluate(best)
	for _, position := range positions {
		if value := fn.Evaluate(position); value < bestValue {
			bestValue = value
			best = append([]float64(nil), position...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for i := 0; i < fireflies; i++ {
			for j := 0; j < fireflies; j++ {
				if light[j] <= light[i] {
					continue
				}

				distanceSquared := 0.0
				for d := 0; d < dimensions; d++ {
					diff := positions[j][d] - positions[i][d]
					distanceSquared += diff * diff
				}

				betaR := beta * math.Exp(-gamma*distanceSquared)

				for d := 0; d < dimensions; d++ {
					positions[i][d] += betaR*(positions[j][d]-positions[i][d]) + alpha*(rand.Float64()-0.5)
				}

				light[i] = 1 / (1 + fn.Evaluate(positions[i]))
			}
		}

		for _, position := range positions {
			if value := fn.Evaluate(position); value < bestValue {
				bestValue = value
				best = append([]float64(nil), position...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "firefly_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
