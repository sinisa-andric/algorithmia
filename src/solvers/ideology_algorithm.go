package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	ideologyAlgorithmPopulation = 20
	ideologyAlgorithmMaxSteps   = 500
	ideologyAlgorithmTolerance  = 1e-6
	ideologyAlgorithmRangeLow   = -5.0
	ideologyAlgorithmRangeHigh  = 5.0
)

// IdeologyAlgorithm minimizuje konfigurisanu benchmark funkciju koristeći Ideology Algorithm: svaka jedinka
// prilagođava stav ka dominantnoj ideologiji (best-u) sa otporom ka promeni koji nasumično varira svaki korak i
// jedinku
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func IdeologyAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	population := ideologyAlgorithmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := ideologyAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := ideologyAlgorithmTolerance
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
			individual[d] = ideologyAlgorithmRangeLow + rand.Float64()*(ideologyAlgorithmRangeHigh-ideologyAlgorithmRangeLow)
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
			resistance := rand.Float64() * 0.5
			for d := range individuals[i] {
				pull := (1 - resistance) * rand.Float64() * (best[d] - individuals[i][d])
				noise := resistance * (rand.Float64()*2 - 1) * 0.4
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, ideologyAlgorithmRangeLow, ideologyAlgorithmRangeHigh)
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
		Method:     "ideology_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
