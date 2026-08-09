package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	donkeyPopulation = 20
	donkeyMaxSteps   = 1000
	donkeyTolerance  = 1e-6
	donkeyRangeLow   = -5.0
	donkeyRangeHigh  = 5.0
)

// Donkey minimizuje konfigurisanu benchmark funkciju koristeći Donkey and Smuggler Optimization: svaki magarac
// kombinuje kretanje ka vođi (best) i ka nasumičnom drugom magarcu, ponderisano faktorom alpha koji opada tokom
// izvršavanja u korist vođe
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Donkey(problem models.Problem) (result models.Result, err error) {

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

	population := donkeyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := donkeyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := donkeyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
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

	donkeys := make([][]float64, population)
	values := make([]float64, population)
	for i := range donkeys {
		donkey := make([]float64, dimensions)
		for d := range donkey {
			donkey[d] = donkeyRangeLow + rand.Float64()*(donkeyRangeHigh-donkeyRangeLow)
		}
		donkeys[i] = donkey
		values[i] = fn.Evaluate(donkey)
	}

	best := append([]float64(nil), donkeys[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), donkeys[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		alpha := 2 * (1 - float64(steps)/float64(maxSteps))

		for i := range donkeys {
			r := donkeys[randomIndexExcept(i)]
			for d := range donkeys[i] {
				donkeys[i][d] = donkeys[i][d] + alpha*rand.Float64()*(best[d]-donkeys[i][d]) + (1-alpha)*rand.Float64()*(r[d]-donkeys[i][d])
				donkeys[i][d] = clamp(donkeys[i][d], donkeyRangeLow, donkeyRangeHigh)
			}
			values[i] = fn.Evaluate(donkeys[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), donkeys[i]...)
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
		Method:     "donkey",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
