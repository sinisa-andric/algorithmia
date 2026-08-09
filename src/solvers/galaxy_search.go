package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	galaxySearchPopulation = 20
	galaxySearchMaxSteps   = 500
	galaxySearchTolerance  = 1e-6
	galaxySearchRangeLow   = -5.0
	galaxySearchRangeHigh  = 5.0
)

// GalaxySearch minimizuje konfigurisanu benchmark funkciju koristeći Gravitational-based Search Algorithm: svaka
// zvezda kruži oko centra galaksije (best-a) po spiralnoj putanji čiji se radijus sužava tokom izvršavanja, uz
// dodatni šum koji predstavlja nepravilnosti orbite
// problem.Point inicijalizuje populaciju zvezda i određuje njenu dimenzionalnost
func GalaxySearch(problem models.Problem) (result models.Result, err error) {

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

	population := galaxySearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := galaxySearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := galaxySearchTolerance
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
			individual[d] = galaxySearchRangeLow + rand.Float64()*(galaxySearchRangeHigh-galaxySearchRangeLow)
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

		spiralFactor := 0.3 * (1 - float64(steps)/float64(maxSteps))

		for i := range individuals {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = individuals[i][d] - best[d]
			}
			radius := norm(diff)
			angle := rand.Float64() * 2 * math.Pi

			for d := range individuals[i] {
				individuals[i][d] = clamp(best[d]+radius*(1-spiralFactor)*math.Cos(angle+float64(d))+(rand.Float64()*2-1)*0.1, galaxySearchRangeLow, galaxySearchRangeHigh)
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
		Method:     "galaxy_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
