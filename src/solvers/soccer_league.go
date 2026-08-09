package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	soccerLeaguePopulation = 20
	soccerLeagueMaxSteps   = 500
	soccerLeagueTolerance  = 1e-6
	soccerLeagueRangeLow   = -5.0
	soccerLeagueRangeHigh  = 5.0
)

// SoccerLeague minimizuje konfigurisanu benchmark funkciju koristeći Soccer League Competition: svaki tim igra
// meč protiv nasumičnog protivnika — pobeda donosi blagu korekciju ka šampionu (best-u) uz malu perturbaciju,
// poraz donosi veći trening pomak uz širu perturbaciju
// problem.Point inicijalizuje populaciju timova i određuje njenu dimenzionalnost
func SoccerLeague(problem models.Problem) (result models.Result, err error) {

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

	population := soccerLeaguePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := soccerLeagueMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := soccerLeagueTolerance
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
			individual[d] = soccerLeagueRangeLow + rand.Float64()*(soccerLeagueRangeHigh-soccerLeagueRangeLow)
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
			opponent := i
			if len(individuals) > 1 {
				opponent = rand.IntN(len(individuals))
				for opponent == i {
					opponent = rand.IntN(len(individuals))
				}
			}

			if values[i] < values[opponent] {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
					noise := (rand.Float64()*2 - 1) * 0.05
					individuals[i][d] = clamp(individuals[i][d]+pull+noise, soccerLeagueRangeLow, soccerLeagueRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.6
					noise := (rand.Float64()*2 - 1) * 0.3
					individuals[i][d] = clamp(individuals[i][d]+pull+noise, soccerLeagueRangeLow, soccerLeagueRangeHigh)
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
		Method:     "soccer_league",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
