package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	tarantulaHawkPopulation = 20
	tarantulaHawkMaxSteps   = 1000
	tarantulaHawkTolerance  = 1e-6
	tarantulaHawkRangeLow   = -5.0
	tarantulaHawkRangeHigh  = 5.0
)

// TarantulaHawk minimizuje konfigurisanu benchmark funkciju koristeći Tarantula Hawk Wasp Optimizer: osa ili traži
// tarantulu Levy letom (istraživanje) ili je napada i paralizuje krećući se direktno ka najboljem rešenju sa
// intenzitetom koji raste tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func TarantulaHawk(problem models.Problem) (result models.Result, err error) {

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

	population := tarantulaHawkPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := tarantulaHawkMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := tarantulaHawkTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	wasps := make([][]float64, population)
	values := make([]float64, population)
	for i := range wasps {
		wasp := make([]float64, dimensions)
		for d := range wasp {
			wasp[d] = tarantulaHawkRangeLow + rand.Float64()*(tarantulaHawkRangeHigh-tarantulaHawkRangeLow)
		}
		wasps[i] = wasp
		values[i] = fn.Evaluate(wasp)
	}

	best := append([]float64(nil), wasps[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), wasps[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range wasps {
			if rand.Float64() < 0.5 {
				l := mantegnaLevy(1.5)
				for d := range wasps[i] {
					wasps[i][d] = wasps[i][d] + l*(rand.Float64()*2-1)
				}
			} else {
				for d := range wasps[i] {
					wasps[i][d] = wasps[i][d] + rand.Float64()*(best[d]-wasps[i][d])*(1+float64(steps)/float64(maxSteps))
				}
			}

			for d := range wasps[i] {
				wasps[i][d] = clamp(wasps[i][d], tarantulaHawkRangeLow, tarantulaHawkRangeHigh)
			}
			values[i] = fn.Evaluate(wasps[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), wasps[i]...)
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
		Method:     "tarantula_hawk",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
