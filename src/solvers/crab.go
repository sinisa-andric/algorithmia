package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	crabPopulation = 20
	crabMaxSteps   = 1000
	crabTolerance  = 1e-6
	crabRangeLow   = -5.0
	crabRangeHigh  = 5.0
)

// Crab minimizuje konfigurisanu benchmark funkciju koristeći Crab Optimization Algorithm: rak se ili bočno skače
// kombinujući kretanje ka najboljem rešenju i ka nasumičnom peer-u, ili se zakopava u pesak izvodeći malu
// odbrambenu lokalnu pretragu
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Crab(problem models.Problem) (result models.Result, err error) {

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

	population := crabPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := crabMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := crabTolerance
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

	crabs := make([][]float64, population)
	values := make([]float64, population)
	for i := range crabs {
		crab := make([]float64, dimensions)
		for d := range crab {
			crab[d] = crabRangeLow + rand.Float64()*(crabRangeHigh-crabRangeLow)
		}
		crabs[i] = crab
		values[i] = fn.Evaluate(crab)
	}

	best := append([]float64(nil), crabs[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), crabs[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range crabs {
			if rand.Float64() < 0.6 {
				r := crabs[randomIndexExcept(i)]
				for d := range crabs[i] {
					crabs[i][d] = crabs[i][d] + rand.Float64()*(best[d]-crabs[i][d]) + rand.Float64()*(r[d]-crabs[i][d])*0.3
				}
			} else {
				for d := range crabs[i] {
					crabs[i][d] = crabs[i][d] + (rand.Float64()*2-1)*0.15
				}
			}

			for d := range crabs[i] {
				crabs[i][d] = clamp(crabs[i][d], crabRangeLow, crabRangeHigh)
			}
			values[i] = fn.Evaluate(crabs[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), crabs[i]...)
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
		Method:     "crab",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
