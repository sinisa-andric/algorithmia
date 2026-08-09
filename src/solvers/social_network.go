package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	socialNetworkPopulation = 20
	socialNetworkMaxSteps   = 500
	socialNetworkTolerance  = 1e-6
	socialNetworkRangeLow   = -5.0
	socialNetworkRangeHigh  = 5.0
)

// SocialNetwork minimizuje konfigurisanu benchmark funkciju koristeći Social Network Optimization: jedinke se
// ugledaju na uticajni čvor izabran rulet-selekcijom ponderisanom fitnesom i istovremeno teže ka globalno
// najuticajnijem čvoru (best-u), uz nezavisan šum
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SocialNetwork(problem models.Problem) (result models.Result, err error) {

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

	population := socialNetworkPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := socialNetworkMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := socialNetworkTolerance
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
			individual[d] = socialNetworkRangeLow + rand.Float64()*(socialNetworkRangeHigh-socialNetworkRangeLow)
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

		weights := make([]float64, len(individuals))
		totalWeight := 0.0
		for j, v := range values {
			weights[j] = 1 / (1 + v)
			totalWeight += weights[j]
		}
		pickInfluencer := func() int {
			target := rand.Float64() * totalWeight
			cumulative := 0.0
			for j, w := range weights {
				cumulative += w
				if target <= cumulative {
					return j
				}
			}
			return len(weights) - 1
		}

		for i := range individuals {
			influencer := pickInfluencer()
			for d := range individuals[i] {
				pullInfluencer := rand.Float64() * (individuals[influencer][d] - individuals[i][d]) * 0.4
				pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
				noise := (rand.Float64()*2 - 1) * 0.15
				individuals[i][d] = clamp(individuals[i][d]+pullInfluencer+pullBest+noise, socialNetworkRangeLow, socialNetworkRangeHigh)
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
		Method:     "social_network",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
