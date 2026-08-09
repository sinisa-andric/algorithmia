package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sailfishPopulation = 20
	sailfishMaxSteps   = 1000
	sailfishTolerance  = 1e-6
	sailfishRangeLow   = -5.0
	sailfishRangeHigh  = 5.0
)

// Sailfish minimizuje konfigurisanu benchmark funkciju koristeći Sailfish Optimizer: jedrenjaci lovе sardine krećući
// se ka najboljem rešenju usrednjenom sa nasumičnom sardinom, dok se sardine kreću ka najboljem rešenju,
// simulirajući lov u paru populacija
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Sailfish(problem models.Problem) (result models.Result, err error) {

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

	population := sailfishPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sailfishMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sailfishTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sailfish := make([][]float64, population)
	sailfishValues := make([]float64, population)
	sardines := make([][]float64, population)
	sardineValues := make([]float64, population)
	for i := range population {
		sf := make([]float64, dimensions)
		sd := make([]float64, dimensions)
		for d := range sf {
			sf[d] = sailfishRangeLow + rand.Float64()*(sailfishRangeHigh-sailfishRangeLow)
			sd[d] = sailfishRangeLow + rand.Float64()*(sailfishRangeHigh-sailfishRangeLow)
		}
		sailfish[i] = sf
		sardines[i] = sd
		sailfishValues[i] = fn.Evaluate(sf)
		sardineValues[i] = fn.Evaluate(sd)
	}

	best := append([]float64(nil), sailfish[0]...)
	bestValue := sailfishValues[0]
	for i, v := range sailfishValues {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), sailfish[i]...)
		}
	}
	for i, v := range sardineValues {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), sardines[i]...)
		}
	}

	const pp = 0.5

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range sailfish {
			s := sardines[rand.IntN(population)]
			for d := range sailfish[i] {
				sailfish[i][d] = best[d] - pp*(rand.Float64()*(best[d]+s[d])/2-sailfish[i][d])
				sailfish[i][d] = clamp(sailfish[i][d], sailfishRangeLow, sailfishRangeHigh)
			}
			sailfishValues[i] = fn.Evaluate(sailfish[i])
		}

		for i := range sardines {
			for d := range sardines[i] {
				sardines[i][d] = rand.Float64()*(best[d]-sardines[i][d]) + sardines[i][d]
				sardines[i][d] = clamp(sardines[i][d], sailfishRangeLow, sailfishRangeHigh)
			}
			sardineValues[i] = fn.Evaluate(sardines[i])
		}

		for i, v := range sailfishValues {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), sailfish[i]...)
			}
		}
		for i, v := range sardineValues {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), sardines[i]...)
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
		Method:     "sailfish",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
