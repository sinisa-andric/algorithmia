package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	antlionPopulation = 20
	antlionMaxSteps   = 1000
	antlionTolerance  = 1e-6
	antlionRangeLow   = -5.0
	antlionRangeHigh  = 5.0
)

// Antlion minimizuje konfigurisanu benchmark funkciju koristeći Antlion Optimizer: svaki mrav nasumično hoda oko
// elitnog i nasumično izabranog mravljeg lava unutar opsega koji se sužava tokom izvršavanja,
// a mravlji lav biva zamenjen ako ga mrav nadmaši
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Antlion(problem models.Problem) (result models.Result, err error) {

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

	population := antlionPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := antlionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := antlionTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	antlions := make([][]float64, population)
	values := make([]float64, population)
	for i := range antlions {
		antlion := make([]float64, dimensions)
		for d := range antlion {
			antlion[d] = antlionRangeLow + rand.Float64()*(antlionRangeHigh-antlionRangeLow)
		}
		antlions[i] = antlion
		values[i] = fn.Evaluate(antlion)
	}

	ants := make([][]float64, population)
	for i := range antlions {
		ants[i] = append([]float64(nil), antlions[i]...)
	}

	best := append([]float64(nil), antlions[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), antlions[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		lb := antlionRangeLow / float64(steps+1)
		ub := antlionRangeHigh / float64(steps+1)

		for i := range ants {
			// elite (best) i nasumično izabrani antlion oboje vode nasumičnu šetnju mrava,
			// nova pozicija mrava je njihov prosek
			j := rand.IntN(population)

			for d := range ants[i] {
				randWalkAntlion := lb + rand.Float64()*(ub-lb)
				randWalkElite := lb + rand.Float64()*(ub-lb)
				ants[i][d] = (randWalkAntlion + randWalkElite) / 2
				ants[i][d] = clamp(ants[i][d], antlionRangeLow, antlionRangeHigh)
			}

			antValue := fn.Evaluate(ants[i])
			if antValue < values[j] {
				antlions[j] = append([]float64(nil), ants[i]...)
				values[j] = antValue
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), antlions[i]...)
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
		Method:     "antlion",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
