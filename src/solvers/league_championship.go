package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	leagueChampionshipPopulation = 20
	leagueChampionshipMaxSteps   = 500
	leagueChampionshipTolerance  = 1e-6
	leagueChampionshipRangeLow   = -5.0
	leagueChampionshipRangeHigh  = 5.0
)

// LeagueChampionship minimizuje konfigurisanu benchmark funkciju koristeći League Championship Algorithm: timovi
// igraju round-robin raspored protiv protivnika određenog po koraku, pobednik zadržava taktiku uz malu
// perturbaciju dok gubitnik menja taktiku pomakom ka prvaku
// problem.Point inicijalizuje populaciju timova i određuje njenu dimenzionalnost
func LeagueChampionship(problem models.Problem) (result models.Result, err error) {

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

	population := leagueChampionshipPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := leagueChampionshipMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := leagueChampionshipTolerance
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
			individual[d] = leagueChampionshipRangeLow + rand.Float64()*(leagueChampionshipRangeHigh-leagueChampionshipRangeLow)
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

		n := len(individuals)
		for i := range individuals {
			opponent := (i + steps + 1) % n
			if opponent == i {
				opponent = (opponent + 1) % n
			}

			if values[i] < values[opponent] {
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.2, leagueChampionshipRangeLow, leagueChampionshipRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*(best[d]-individuals[i][d])*0.6, leagueChampionshipRangeLow, leagueChampionshipRangeHigh)
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
		Method:     "league_championship",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
