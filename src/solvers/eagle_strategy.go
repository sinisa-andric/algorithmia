package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	eagleStrategyPopulation = 20
	eagleStrategyMaxSteps   = 1000
	eagleStrategyTolerance  = 1e-6
	eagleStrategyRangeLow   = -5.0
	eagleStrategyRangeHigh  = 5.0
)

// EagleStrategy minimizuje konfigurisanu benchmark funkciju koristeći Eagle Strategy: orao naizmenično bira globalnu
// eksploraciju Levy letom ili lokalnu intenzifikaciju finom pretragom oko najboljeg rešenja amplitudom koja opada
// tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func EagleStrategy(problem models.Problem) (result models.Result, err error) {

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

	population := eagleStrategyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := eagleStrategyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := eagleStrategyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	eagles := make([][]float64, population)
	values := make([]float64, population)
	for i := range eagles {
		eagle := make([]float64, dimensions)
		for d := range eagle {
			eagle[d] = eagleStrategyRangeLow + rand.Float64()*(eagleStrategyRangeHigh-eagleStrategyRangeLow)
		}
		eagles[i] = eagle
		values[i] = fn.Evaluate(eagle)
	}

	best := append([]float64(nil), eagles[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), eagles[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range eagles {
			if rand.Float64() < 0.5 {
				l := mantegnaLevy(1.5)
				for d := range eagles[i] {
					eagles[i][d] = eagles[i][d] + l*(rand.Float64()*2-1)
				}
			} else {
				for d := range eagles[i] {
					eagles[i][d] = best[d] + (rand.Float64()*2-1)*0.2*(1-float64(steps)/float64(maxSteps))
				}
			}

			for d := range eagles[i] {
				eagles[i][d] = clamp(eagles[i][d], eagleStrategyRangeLow, eagleStrategyRangeHigh)
			}
			values[i] = fn.Evaluate(eagles[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), eagles[i]...)
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
		Method:     "eagle_strategy",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
