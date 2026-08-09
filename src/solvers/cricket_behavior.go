package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	cricketBehaviorPopulation = 20
	cricketBehaviorMaxSteps   = 500
	cricketBehaviorTolerance  = 1e-6
	cricketBehaviorRangeLow   = -5.0
	cricketBehaviorRangeHigh  = 5.0
)

// CricketBehavior minimizuje konfigurisanu benchmark funkciju koristeći Cricket Behavior Optimization: glasnoća
// zvuka jača što je cvrčak bliži najglasnijem (best-u) i privlači ga, dok slabiji zvuk ostavlja više prostora za
// nasumično lutanje
// problem.Point inicijalizuje populaciju cvrčaka i određuje njenu dimenzionalnost
func CricketBehavior(problem models.Problem) (result models.Result, err error) {

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

	population := cricketBehaviorPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := cricketBehaviorMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := cricketBehaviorTolerance
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
			individual[d] = cricketBehaviorRangeLow + rand.Float64()*(cricketBehaviorRangeHigh-cricketBehaviorRangeLow)
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

		for i := range individuals {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = individuals[i][d] - best[d]
			}
			loudness := 1 / (1 + norm(diff))

			for d := range individuals[i] {
				pull := loudness * rand.Float64() * (best[d] - individuals[i][d])
				// loudness teži 1 kako se cvrčak približava best-u, a (1-loudness) tada teži 0 zajedno sa pull
				// članom (koji takođe teži 0 jer best-x teži 0) — bez donje granice bi cvrčak koji se približi
				// best-u potpuno prestao da se pomera; mala nezavisna perturbacija to sprečava
				noise := (1-loudness)*(rand.Float64()*2-1)*0.5 + (rand.Float64()*2-1)*0.02
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, cricketBehaviorRangeLow, cricketBehaviorRangeHigh)
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
		Method:     "cricket_behavior",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
