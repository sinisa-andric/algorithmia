package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	flyingFoxesPopulation = 20
	flyingFoxesMaxSteps   = 1000
	flyingFoxesTolerance  = 1e-6
	flyingFoxesRangeLow   = -5.0
	flyingFoxesRangeHigh  = 5.0
)

// FlyingFoxes minimizuje konfigurisanu benchmark funkciju koristeći Flying Fox Optimization: svaka lisica kombinuje
// kretanje ka najboljem rešenju i ka nasumičnom članu jata, sa težinama koje pomera termalni stres —
// raste tokom izvršavanja i favorizuje istraživanje nad best-om
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func FlyingFoxes(problem models.Problem) (result models.Result, err error) {

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

	population := flyingFoxesPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := flyingFoxesMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := flyingFoxesTolerance
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

	foxes := make([][]float64, population)
	values := make([]float64, population)
	for i := range foxes {
		fox := make([]float64, dimensions)
		for d := range fox {
			fox[d] = flyingFoxesRangeLow + rand.Float64()*(flyingFoxesRangeHigh-flyingFoxesRangeLow)
		}
		foxes[i] = fox
		values[i] = fn.Evaluate(fox)
	}

	best := append([]float64(nil), foxes[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), foxes[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		heatStress := float64(steps) / float64(maxSteps)

		for i := range foxes {
			r := foxes[randomIndexExcept(i)]
			for d := range foxes[i] {
				foxes[i][d] = foxes[i][d] + (1-heatStress)*rand.Float64()*(best[d]-foxes[i][d]) + heatStress*rand.Float64()*(r[d]-foxes[i][d])
			}

			for d := range foxes[i] {
				foxes[i][d] = clamp(foxes[i][d], flyingFoxesRangeLow, flyingFoxesRangeHigh)
			}
			values[i] = fn.Evaluate(foxes[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), foxes[i]...)
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
		Method:     "flying_foxes",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
