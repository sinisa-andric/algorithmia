package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	crowSearchPopulation = 20
	crowSearchMaxSteps   = 1000
	crowSearchTolerance  = 1e-6
	crowSearchRangeLow   = -5.0
	crowSearchRangeHigh  = 5.0
	crowSearchAwareness  = 0.1
)

// CrowSearch minimizuje konfigurisanu benchmark funkciju koristeći Crow Search Algorithm: svaki gavran ili prati
// memorisanu poziciju nasumičnog gavrana ili nasumično istražuje prostor pretrage,
// u zavisnosti od svesnosti (awareness) da je praćen
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func CrowSearch(problem models.Problem) (result models.Result, err error) {

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

	population := crowSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := crowSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := crowSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	awareness := crowSearchAwareness
	if v, ok := problem.Payload["awareness"].(float64); ok {
		awareness = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	randomIndexExcept := func(exclude int) int {
		if population < 2 {
			return exclude
		}
		j := rand.IntN(population)
		for j == exclude {
			j = rand.IntN(population)
		}
		return j
	}

	crows := make([][]float64, population)
	values := make([]float64, population)
	for i := range crows {
		crow := make([]float64, dimensions)
		for d := range crow {
			crow[d] = crowSearchRangeLow + rand.Float64()*(crowSearchRangeHigh-crowSearchRangeLow)
		}
		crows[i] = crow
		values[i] = fn.Evaluate(crow)
	}

	memory := make([][]float64, population)
	memoryValues := make([]float64, population)
	for i := range crows {
		memory[i] = append([]float64(nil), crows[i]...)
		memoryValues[i] = values[i]
	}

	best := append([]float64(nil), memory[0]...)
	bestValue := memoryValues[0]
	for i, v := range memoryValues {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), memory[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range crows {
			j := randomIndexExcept(i)
			if rand.Float64() >= awareness {
				fl := 2.0
				for d := range crows[i] {
					crows[i][d] = crows[i][d] + rand.Float64()*fl*(memory[j][d]-crows[i][d])
				}
			} else {
				for d := range crows[i] {
					crows[i][d] = crowSearchRangeLow + rand.Float64()*(crowSearchRangeHigh-crowSearchRangeLow)
				}
			}
			for d := range crows[i] {
				crows[i][d] = clamp(crows[i][d], crowSearchRangeLow, crowSearchRangeHigh)
			}
			values[i] = fn.Evaluate(crows[i])

			if values[i] < memoryValues[i] {
				memory[i] = append([]float64(nil), crows[i]...)
				memoryValues[i] = values[i]
			}
		}

		for i, v := range memoryValues {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), memory[i]...)
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
		Method:     "crow_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
