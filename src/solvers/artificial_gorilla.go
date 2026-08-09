package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	artificialGorillaPopulation  = 20
	artificialGorillaMaxSteps    = 1000
	artificialGorillaTolerance   = 1e-6
	artificialGorillaRangeLow    = -5.0
	artificialGorillaRangeHigh   = 5.0
	artificialGorillaMigrateProb = 0.03
)

// ArtificialGorilla minimizuje konfigurisanu benchmark funkciju koristeći Artificial Gorilla Troops Optimizer: svaka
// gorila retko migrira ka potpuno nasumičnoj poziciji, inače se kreće bilo ka nasumičnom drugom članu čopora bilo ka
// silverback-u (trenutno najboljem rešenju) uz kohezioni član koji ih približava
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ArtificialGorilla(problem models.Problem) (result models.Result, err error) {

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

	population := artificialGorillaPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := artificialGorillaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := artificialGorillaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	gorillas := make([][]float64, population)
	values := make([]float64, population)
	for i := range gorillas {
		gorilla := make([]float64, dimensions)
		for d := range gorilla {
			gorilla[d] = artificialGorillaRangeLow + rand.Float64()*(artificialGorillaRangeHigh-artificialGorillaRangeLow)
		}
		gorillas[i] = gorilla
		values[i] = fn.Evaluate(gorilla)
	}

	best := append([]float64(nil), gorillas[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), gorillas[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range gorillas {
			r := rand.Float64()

			switch {
			case r < artificialGorillaMigrateProb:
				for d := range gorillas[i] {
					gorillas[i][d] = artificialGorillaRangeLow + rand.Float64()*(artificialGorillaRangeHigh-artificialGorillaRangeLow)
				}
			case rand.Float64() >= 0.5:
				other := gorillas[rand.IntN(population)]
				for d := range gorillas[i] {
					gorillas[i][d] = gorillas[i][d] - rand.Float64()*gorillas[i][d] + rand.Float64()*other[d]
				}
			default:
				c := rand.Float64()
				for d := range gorillas[i] {
					gorillas[i][d] = best[d] + rand.Float64()*(gorillas[i][d]-rand.Float64()*best[d]) + c*(gorillas[i][d]-best[d])
				}
			}

			for d := range gorillas[i] {
				gorillas[i][d] = clamp(gorillas[i][d], artificialGorillaRangeLow, artificialGorillaRangeHigh)
			}

			values[i] = fn.Evaluate(gorillas[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), gorillas[i]...)
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
		Method:     "artificial_gorilla",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
