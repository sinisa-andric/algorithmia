package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	queenBeeEvolutionPopulation = 20
	queenBeeEvolutionMaxSteps   = 500
	queenBeeEvolutionTolerance  = 1e-6
	queenBeeEvolutionRangeLow   = -5.0
	queenBeeEvolutionRangeHigh  = 5.0
)

// QueenBeeEvolution minimizuje konfigurisanu benchmark funkciju koristeći Queen Bee Evolution: svaka radnica se
// ukršta sa kraljicom (best-om) usrednjavanjem pozicija, uz mutacionu perturbaciju čija amplituda opada tokom
// izvršavanja
// problem.Point inicijalizuje populaciju pčela i određuje njenu dimenzionalnost
func QueenBeeEvolution(problem models.Problem) (result models.Result, err error) {

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

	population := queenBeeEvolutionPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := queenBeeEvolutionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := queenBeeEvolutionTolerance
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
			individual[d] = queenBeeEvolutionRangeLow + rand.Float64()*(queenBeeEvolutionRangeHigh-queenBeeEvolutionRangeLow)
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
			for d := range individuals[i] {
				individuals[i][d] = clamp(0.5*(individuals[i][d]+best[d])+(rand.Float64()*2-1)*0.3*(1-float64(steps)/float64(maxSteps)), queenBeeEvolutionRangeLow, queenBeeEvolutionRangeHigh)
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
		Method:     "queen_bee_evolution",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
