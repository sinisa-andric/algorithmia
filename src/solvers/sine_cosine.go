package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sineCosinePopulation = 20
	sineCosineMaxSteps   = 500
	sineCosineAParam     = 2.0
	sineCosineTolerance  = 1e-6
	sineCosineRangeLow   = -5.0
	sineCosineRangeHigh  = 5.0
)

// SineCosine minimizuje konfigurisanu benchmark funkciju koristeći Sine Cosine Algorithm: svaka jedinka se kreće
// duž sinusne ili kosinusne talasne putanje ka najboljem rešenju, sa amplitudom koja linearno opada tokom
// izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SineCosine(problem models.Problem) (result models.Result, err error) {

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

	population := sineCosinePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sineCosineMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	aParam := sineCosineAParam
	if v, ok := problem.Payload["a_param"].(float64); ok {
		aParam = v
	}

	tolerance := sineCosineTolerance
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
			individual[d] = sineCosineRangeLow + rand.Float64()*(sineCosineRangeHigh-sineCosineRangeLow)
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

		a := aParam * (1 - float64(steps)/float64(maxSteps))

		for i := range individuals {
			for d := range individuals[i] {
				r2 := rand.Float64() * 2 * math.Pi
				r3 := rand.Float64() * 2
				r4 := rand.Float64()
				target := math.Abs(r3*best[d] - individuals[i][d])

				if r4 < 0.5 {
					individuals[i][d] = clamp(individuals[i][d]+a*math.Sin(r2)*target, sineCosineRangeLow, sineCosineRangeHigh)
				} else {
					individuals[i][d] = clamp(individuals[i][d]+a*math.Cos(r2)*target, sineCosineRangeLow, sineCosineRangeHigh)
				}
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
		Method:     "sine_cosine",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
