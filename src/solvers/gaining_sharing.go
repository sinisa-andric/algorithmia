package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	gainingSharingPopulation = 20
	gainingSharingMaxSteps   = 500
	gainingSharingTolerance  = 1e-6
	gainingSharingRangeLow   = -5.0
	gainingSharingRangeHigh  = 5.0
)

// GainingSharing minimizuje konfigurisanu benchmark funkciju koristeći Gaining Sharing Knowledge Algorithm:
// mlađi (slabije rangirani) uče kombinacijom dve nasumične starije jedinke uz vezu ka best-u, dok stariji dele
// znanje direktno sa best-om
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GainingSharing(problem models.Problem) (result models.Result, err error) {

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

	population := gainingSharingPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := gainingSharingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gainingSharingTolerance
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
			individual[d] = gainingSharingRangeLow + rand.Float64()*(gainingSharingRangeHigh-gainingSharingRangeLow)
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

		order := make([]int, len(individuals))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		half := len(order) / 2
		if half < 1 {
			half = 1
		}
		if half >= len(order) {
			half = len(order) - 1
		}

		for _, idx := range order[half:] {
			i1 := order[rand.IntN(half)]
			i2 := order[rand.IntN(half)]
			for d := range individuals[idx] {
				pull := rand.Float64() * (individuals[i1][d] - individuals[i2][d])
				pullBest := rand.Float64() * (best[d] - individuals[idx][d]) * 0.2
				individuals[idx][d] = clamp(individuals[idx][d]+pull+pullBest, gainingSharingRangeLow, gainingSharingRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
		}

		for _, idx := range order[:half] {
			for d := range individuals[idx] {
				pull := rand.Float64() * (best[d] - individuals[idx][d]) * 0.5
				noise := (rand.Float64()*2 - 1) * 0.2
				individuals[idx][d] = clamp(individuals[idx][d]+pull+noise, gainingSharingRangeLow, gainingSharingRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
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
		Method:     "gaining_sharing",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
