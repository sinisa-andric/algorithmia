package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	footballOptimizationPopulation = 20
	footballOptimizationMaxSteps   = 500
	footballOptimizationTolerance  = 1e-6
	footballOptimizationRangeLow   = -5.0
	footballOptimizationRangeHigh  = 5.0
)

// FootballOptimization minimizuje konfigurisanu benchmark funkciju koristeći Football Game-Based Optimization:
// igrači se pozicioniraju kombinujući vuču ka trenerskoj taktici (best-u) i ka trenutnoj poziciji lopte (lokalno
// najboljem u populaciji ovog koraka), uz nezavisan šum
// problem.Point inicijalizuje populaciju igrača i određuje njenu dimenzionalnost
func FootballOptimization(problem models.Problem) (result models.Result, err error) {

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

	population := footballOptimizationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := footballOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := footballOptimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = footballOptimizationRangeLow + rand.Float64()*(footballOptimizationRangeHigh-footballOptimizationRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		ballIdx := 0
		for i, v := range values {
			if v < values[ballIdx] {
				ballIdx = i
			}
		}

		for i := range individuals {
			for d := range individuals[i] {
				pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.4
				pullBall := rand.Float64() * (individuals[ballIdx][d] - individuals[i][d]) * 0.3
				noise := (rand.Float64()*2 - 1) * 0.2
				individuals[i][d] = clamp(individuals[i][d]+pullBest+pullBall+noise, footballOptimizationRangeLow, footballOptimizationRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
			}
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
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "football_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
