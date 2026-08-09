package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	colonialCompetitivePopulation = 20
	colonialCompetitiveMaxSteps   = 500
	colonialCompetitiveTolerance  = 1e-6
	colonialCompetitiveRangeLow   = -5.0
	colonialCompetitiveRangeHigh  = 5.0
)

// ColonialCompetitive minimizuje konfigurisanu benchmark funkciju koristeći Colonial Competitive Algorithm: za
// razliku od formalne imperijske hijerarhije, svaka jedinka nezavisno migrira ka svojoj najbližoj boljoj jedinki
// (ili ka best-u ako takva ne postoji), bez fiksne raspodele kolonija ili takmičenja imperija
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ColonialCompetitive(problem models.Problem) (result models.Result, err error) {

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

	population := colonialCompetitivePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := colonialCompetitiveMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := colonialCompetitiveTolerance
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
			individual[d] = colonialCompetitiveRangeLow + rand.Float64()*(colonialCompetitiveRangeHigh-colonialCompetitiveRangeLow)
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
			nearestIdx := -1
			nearestDist := math.Inf(1)
			for j := range individuals {
				if j == i || values[j] >= values[i] {
					continue
				}
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = individuals[j][d] - individuals[i][d]
				}
				dist := norm(diff)
				if dist < nearestDist {
					nearestDist = dist
					nearestIdx = j
				}
			}

			target := best
			if nearestIdx >= 0 {
				target = individuals[nearestIdx]
			}

			for d := range individuals[i] {
				pull := rand.Float64() * (target[d] - individuals[i][d]) * 0.5
				noise := (rand.Float64()*2 - 1) * 0.2
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, colonialCompetitiveRangeLow, colonialCompetitiveRangeHigh)
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
		Method:     "colonial_competitive",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
