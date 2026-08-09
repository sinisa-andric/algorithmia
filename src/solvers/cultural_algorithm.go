package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	culturalAlgorithmPopulation = 20
	culturalAlgorithmMaxSteps   = 500
	culturalAlgorithmTolerance  = 1e-6
	culturalAlgorithmRangeLow   = -5.0
	culturalAlgorithmRangeHigh  = 5.0
)

// CulturalAlgorithm minimizuje konfigurisanu benchmark funkciju koristeći Cultural Algorithm: "belief space"
// (prosek najboljih 20% populacije) usmerava kretanje cele populacije, uz nezavisnu perturbaciju čija amplituda
// opada tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func CulturalAlgorithm(problem models.Problem) (result models.Result, err error) {

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

	population := culturalAlgorithmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := culturalAlgorithmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := culturalAlgorithmTolerance
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
			individual[d] = culturalAlgorithmRangeLow + rand.Float64()*(culturalAlgorithmRangeHigh-culturalAlgorithmRangeLow)
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

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		top20 := len(order) / 5
		if top20 < 1 {
			top20 = 1
		}

		belief := make([]float64, dimensions)
		for k := 0; k < top20; k++ {
			for d := range belief {
				belief[d] += individuals[order[k]][d]
			}
		}
		for d := range belief {
			belief[d] /= float64(top20)
		}

		for i := range individuals {
			for d := range individuals[i] {
				pull := rand.Float64() * (belief[d] - individuals[i][d]) * 0.5
				noise := (rand.Float64()*2 - 1) * 0.3 * (1 - float64(steps)/float64(maxSteps))
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, culturalAlgorithmRangeLow, culturalAlgorithmRangeHigh)
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
		Method:     "cultural_algorithm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
