package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	bayesianOptimizationSamples       = 20
	bayesianOptimizationIterations    = 50
	bayesianOptimizationRange         = 15.0
	bayesianOptimizationTolerance     = 1e-6
	bayesianOptimizationCandidates    = 100
	bayesianOptimizationExploitFactor = 0.5
)

// BayesianOptimization minimizuje sphere funkciju koristeći pojednostavljenu Bajesovsku pretragu bez Gausovog
// procesa: povlači početnu nasumičnu populaciju, a zatim ponavljano eksploatiše trenutno najbolje uzorkovanjem
// kandidata oko njega
// problem.Point samo određuje dimenzionalnost
func BayesianOptimization(problem models.Problem) (result models.Result, err error) {

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

	samples := bayesianOptimizationSamples
	if v, ok := problem.Payload["samples"].(float64); ok {
		samples = int(v)
	}

	iterations := bayesianOptimizationIterations
	if v, ok := problem.Payload["iterations"].(float64); ok {
		iterations = int(v)
	}

	searchRange := bayesianOptimizationRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}

	tolerance := bayesianOptimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	best := make([]float64, dimensions)
	bestValue := math.Inf(1)

	for i := 0; i < samples; i++ {
		candidate := make([]float64, dimensions)
		for d := range candidate {
			candidate[d] = (rand.Float64()*2 - 1) * searchRange
		}

		if value := fn.Evaluate(candidate); value < bestValue {
			bestValue = value
			best = candidate
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, i, best, bestValue, false)
		}
	}

	exploitRadius := searchRange * bayesianOptimizationExploitFactor

	for i := 0; i < iterations && bestValue >= tolerance; i++ {

		var localBest []float64
		localBestValue := math.Inf(1)

		for c := 0; c < bayesianOptimizationCandidates; c++ {
			candidate := make([]float64, dimensions)
			for d := range candidate {
				candidate[d] = best[d] + (rand.Float64()*2-1)*exploitRadius
			}

			if value := fn.Evaluate(candidate); value < localBestValue {
				localBestValue = value
				localBest = candidate
			}
		}

		if localBestValue < bestValue {
			bestValue = localBestValue
			best = localBest
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, samples+i, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, samples+iterations, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "bayesian_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      samples + iterations,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
