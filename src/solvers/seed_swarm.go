package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seedSwarmPopulation = 20
	seedSwarmMaxSteps   = 500
	seedSwarmTolerance  = 1e-6
	seedSwarmRangeLow   = -5.0
	seedSwarmRangeHigh  = 5.0
)

// SeedSwarm minimizuje konfigurisanu benchmark funkciju koristeći Seed Swarm Optimization: teža (bolja) semenka
// pada bliže najboljem rešenju dok je lakša (lošija) semenka vetar nosi dalje u širu nasumičnu eksploraciju
// problem.Point inicijalizuje populaciju semenki i određuje njenu dimenzionalnost
func SeedSwarm(problem models.Problem) (result models.Result, err error) {

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

	population := seedSwarmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seedSwarmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seedSwarmTolerance
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
			individual[d] = seedSwarmRangeLow + rand.Float64()*(seedSwarmRangeHigh-seedSwarmRangeLow)
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

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		for i := range individuals {
			weight := (worstValue - values[i]) / (worstValue - bestValue + 1e-10)
			windSpread := (1-weight)*1.5 + 0.1

			for d := range individuals[i] {
				pull := weight * rand.Float64() * (best[d] - individuals[i][d])
				noise := (rand.Float64()*2 - 1) * windSpread * 0.3
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, seedSwarmRangeLow, seedSwarmRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "seed_swarm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
