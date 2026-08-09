package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	humanMentalSearchPopulation = 20
	humanMentalSearchMaxSteps   = 500
	humanMentalSearchTolerance  = 1e-6
	humanMentalSearchRangeLow   = -5.0
	humanMentalSearchRangeHigh  = 5.0
)

// HumanMentalSearch minimizuje konfigurisanu benchmark funkciju koristeći Human Mental Search: svaka ideja se
// pomera ka trenutno pobedničkoj ideji (best-u) mentalnom "cenom" prilagođavanja, uz nezavisnu perturbaciju čija
// amplituda opada tokom izvršavanja
// problem.Point inicijalizuje populaciju ideja i određuje njenu dimenzionalnost
func HumanMentalSearch(problem models.Problem) (result models.Result, err error) {

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

	population := humanMentalSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := humanMentalSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := humanMentalSearchTolerance
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
			individual[d] = humanMentalSearchRangeLow + rand.Float64()*(humanMentalSearchRangeHigh-humanMentalSearchRangeLow)
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
			for d := range individuals[i] {
				pull := (best[d] - individuals[i][d]) * rand.Float64() * 0.4
				noise := (rand.Float64()*2 - 1) * 0.3 * (1 - float64(steps)/float64(maxSteps))
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, humanMentalSearchRangeLow, humanMentalSearchRangeHigh)
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
		Method:     "human_mental_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
