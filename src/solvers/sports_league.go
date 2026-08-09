package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sportsLeaguePopulation   = 20
	sportsLeagueMaxSteps     = 500
	sportsLeagueSeasonLength = 10
	sportsLeagueTolerance    = 1e-6
	sportsLeagueRangeLow     = -5.0
	sportsLeagueRangeHigh    = 5.0
)

// SportsLeague minimizuje konfigurisanu benchmark funkciju koristeći Sports League Optimization: timovi se kreću
// ka prvaku lige (best-u) uz nezavisan šum, a na kraju svake sezone (bloka koraka) najgori tim se transferiše u
// nasumičnu okolinu prvaka
// problem.Point inicijalizuje populaciju timova i određuje njenu dimenzionalnost
func SportsLeague(problem models.Problem) (result models.Result, err error) {

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

	population := sportsLeaguePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sportsLeagueMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	seasonLength := sportsLeagueSeasonLength
	if v, ok := problem.Payload["season_length"].(float64); ok {
		seasonLength = int(v)
	}

	tolerance := sportsLeagueTolerance
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
			individual[d] = sportsLeagueRangeLow + rand.Float64()*(sportsLeagueRangeHigh-sportsLeagueRangeLow)
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
				pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.4
				noise := (rand.Float64()*2 - 1) * 0.3
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, sportsLeagueRangeLow, sportsLeagueRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		if seasonLength > 0 && (steps+1)%seasonLength == 0 {
			worstIdx := 0
			for i, v := range values {
				if v > values[worstIdx] {
					worstIdx = i
				}
			}
			for d := range individuals[worstIdx] {
				individuals[worstIdx][d] = clamp(best[d]+(rand.Float64()*2-1)*0.8, sportsLeagueRangeLow, sportsLeagueRangeHigh)
			}
			values[worstIdx] = fn.Evaluate(individuals[worstIdx])
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
		Method:     "sports_league",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
