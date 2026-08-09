package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	commerceAlgorithmPopulation = 20
	commerceAlgorithmMaxSteps   = 500
	commerceAlgorithmTolerance  = 1e-6
	commerceAlgorithmRangeLow   = -5.0
	commerceAlgorithmRangeHigh  = 5.0
)

// CommerceAlgorithm minimizuje konfigurisanu benchmark funkciju koristeći Commerce/Market Optimization Algorithm:
// trgovac prati tržišnu ravnotežu (best) sa jačinom određenom nasumičnom fluktuacijom ponude i tražnje svakog
// koraka
// problem.Point inicijalizuje populaciju trgovaca i određuje njenu dimenzionalnost
func CommerceAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	population := commerceAlgorithmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := commerceAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := commerceAlgorithmTolerance
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
			individual[d] = commerceAlgorithmRangeLow + rand.Float64()*(commerceAlgorithmRangeHigh-commerceAlgorithmRangeLow)
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

		for i := range individuals {
			demandFactor := rand.Float64()
			for d := range individuals[i] {
				pull := demandFactor * rand.Float64() * (best[d] - individuals[i][d]) * 0.5
				noise := (1 - demandFactor) * (rand.Float64()*2 - 1) * 0.4
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, commerceAlgorithmRangeLow, commerceAlgorithmRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
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
		Method:     "commerce_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
