package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	waterEvaporationPopulation = 20
	waterEvaporationMaxSteps   = 500
	waterEvaporationTolerance  = 1e-6
	waterEvaporationRangeLow   = -5.0
	waterEvaporationRangeHigh  = 5.0
)

// WaterEvaporation minimizuje konfigurisanu benchmark funkciju koristeći Water Evaporation Optimization: u prvoj
// polovini izvršavanja molekuli pretežno isparavaju (šum uz blagu vezu ka best-u), a u drugoj pretežno kondenzuju
// ka best-u uz preostali šum
// problem.Point inicijalizuje populaciju molekula i određuje njenu dimenzionalnost
func WaterEvaporation(problem models.Problem) (result models.Result, err error) {

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

	population := waterEvaporationPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := waterEvaporationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := waterEvaporationTolerance
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
			individual[d] = waterEvaporationRangeLow + rand.Float64()*(waterEvaporationRangeHigh-waterEvaporationRangeLow)
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

		phaseOne := steps < maxSteps/2

		for i := range individuals {
			if phaseOne {
				for d := range individuals[i] {
					noise := (rand.Float64()*2 - 1) * 0.6 * (1 - float64(steps)/float64(maxSteps))
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.2
					individuals[i][d] = clamp(individuals[i][d]+noise+pull, waterEvaporationRangeLow, waterEvaporationRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.6
					noise := (rand.Float64()*2 - 1) * 0.1
					individuals[i][d] = clamp(individuals[i][d]+pull+noise, waterEvaporationRangeLow, waterEvaporationRangeHigh)
				}
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
		Method:     "water_evaporation",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
