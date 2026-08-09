package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	parrotPopulation = 20
	parrotMaxSteps   = 1000
	parrotTolerance  = 1e-6
	parrotRangeLow   = -5.0
	parrotRangeHigh  = 5.0
)

// Parrot minimizuje konfigurisanu benchmark funkciju koristeći Parrot Optimizer: papagaj nasumično bira između
// foraginga (ka najboljem), ostajanja (mala perturbacija), komunikacije (ka najboljem i nasumičnom peer-u) ili
// straha od stranaca (veliki nasumičan beg)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Parrot(problem models.Problem) (result models.Result, err error) {

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

	population := parrotPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := parrotMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := parrotTolerance
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

	parrots := make([][]float64, population)
	values := make([]float64, population)
	for i := range parrots {
		parrot := make([]float64, dimensions)
		for d := range parrot {
			parrot[d] = parrotRangeLow + rand.Float64()*(parrotRangeHigh-parrotRangeLow)
		}
		parrots[i] = parrot
		values[i] = fn.Evaluate(parrot)
	}

	best := append([]float64(nil), parrots[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), parrots[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range parrots {
			p := rand.Float64()
			switch {
			case p < 0.25:
				for d := range parrots[i] {
					parrots[i][d] = parrots[i][d] + rand.Float64()*(best[d]-parrots[i][d])
				}
			case p < 0.5:
				for d := range parrots[i] {
					parrots[i][d] = parrots[i][d] + (rand.Float64()*2-1)*0.1
				}
			case p < 0.75:
				r := parrots[randomIndexExcept(i)]
				for d := range parrots[i] {
					parrots[i][d] = parrots[i][d] + rand.Float64()*(best[d]-parrots[i][d]) + rand.Float64()*(r[d]-parrots[i][d])
				}
			default:
				for d := range parrots[i] {
					parrots[i][d] = parrots[i][d] + (rand.Float64()*2-1)*1.5
				}
			}

			for d := range parrots[i] {
				parrots[i][d] = clamp(parrots[i][d], parrotRangeLow, parrotRangeHigh)
			}
			values[i] = fn.Evaluate(parrots[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), parrots[i]...)
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
		Method:     "parrot",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
