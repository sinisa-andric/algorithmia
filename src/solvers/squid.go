package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	squidPopulation = 20
	squidMaxSteps   = 1000
	squidTolerance  = 1e-6
	squidRangeLow   = -5.0
	squidRangeHigh  = 5.0
)

// Squid minimizuje konfigurisanu benchmark funkciju koristeći Squid Optimization Algorithm: lignja najčešće
// koristi mlazni pogon i brzo se kreće ka najboljem rešenju, a povremeno beži ispuštajući oblak mastila — nagli
// veliki nasumičan pomeraj
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Squid(problem models.Problem) (result models.Result, err error) {

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

	population := squidPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := squidMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := squidTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	squids := make([][]float64, population)
	values := make([]float64, population)
	for i := range squids {
		squid := make([]float64, dimensions)
		for d := range squid {
			squid[d] = squidRangeLow + rand.Float64()*(squidRangeHigh-squidRangeLow)
		}
		squids[i] = squid
		values[i] = fn.Evaluate(squid)
	}

	best := append([]float64(nil), squids[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), squids[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range squids {
			if rand.Float64() < 0.75 {
				for d := range squids[i] {
					squids[i][d] = squids[i][d] + rand.Float64()*(best[d]-squids[i][d])*1.1
				}
			} else {
				for d := range squids[i] {
					squids[i][d] = squids[i][d] + (rand.Float64()*2-1)*1.4
				}
			}

			for d := range squids[i] {
				squids[i][d] = clamp(squids[i][d], squidRangeLow, squidRangeHigh)
			}
			values[i] = fn.Evaluate(squids[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), squids[i]...)
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
		Method:     "squid",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
