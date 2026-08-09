package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	cockroachPopulation = 20
	cockroachMaxSteps   = 1000
	cockroachTolerance  = 1e-6
	cockroachRangeLow   = -5.0
	cockroachRangeHigh  = 5.0
)

// Cockroach minimizuje konfigurisanu benchmark funkciju koristeći Cockroach Swarm Optimization: bubašvaba se
// najčešće juri ka najboljem rešenju, a povremeno se uznemiri i izvede veliki nasumičan skok (raspršivanje)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Cockroach(problem models.Problem) (result models.Result, err error) {

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

	population := cockroachPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := cockroachMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := cockroachTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	roaches := make([][]float64, population)
	values := make([]float64, population)
	for i := range roaches {
		roach := make([]float64, dimensions)
		for d := range roach {
			roach[d] = cockroachRangeLow + rand.Float64()*(cockroachRangeHigh-cockroachRangeLow)
		}
		roaches[i] = roach
		values[i] = fn.Evaluate(roach)
	}

	best := append([]float64(nil), roaches[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), roaches[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range roaches {
			if rand.Float64() < 0.15 {
				for d := range roaches[i] {
					roaches[i][d] = roaches[i][d] + (rand.Float64()*2-1)*1.5
				}
			} else {
				for d := range roaches[i] {
					roaches[i][d] = roaches[i][d] + rand.Float64()*(best[d]-roaches[i][d])
				}
			}

			for d := range roaches[i] {
				roaches[i][d] = clamp(roaches[i][d], cockroachRangeLow, cockroachRangeHigh)
			}
			values[i] = fn.Evaluate(roaches[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), roaches[i]...)
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
		Method:     "cockroach",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
